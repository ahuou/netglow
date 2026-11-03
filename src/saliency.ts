// Saliency heatmap on the image input: gradient × input of the chosen output, per pixel.
//
// onnxruntime-web has no autograd, so the gradient is computed here: a reverse pass over the nodes that lie on the
// path from the image input to the pre-softmax outputs, using the weights from the ONNX file and the real
// intermediate tensors onnxruntime returned for each tick (ReLU masks, max-pool winners, tanh outputs ...). For a
// network of the supported ops this is the exact gradient of the forward pass that was run, not an approximation.
//
// Vocabulary:
//   path       - the nodes both reachable from the image input and needed for the outputs, in graph order
//   gradient   - d(chosen output's logit) / d(each input value), at this tick's activations
//   grad×input - gradient times the input value: how much each input value pushes the logit (first-order, and
//                exact for piecewise-linear nets with zero biases); summed over the channels of the frame stack
//   heat       - that sum per pixel, signed: warm pushes the chosen output up, cool pushes it down

import type { ModelInfo, NodeProto } from "./onnx/proto";

const IDENTITY = new Set(["Reshape", "Flatten", "Squeeze", "Unsqueeze", "Identity", "Dropout", "Cast"]);
const SUPPORTED = new Set([
  ...IDENTITY, "Gemm", "MatMul", "Conv", "Relu", "LeakyRelu", "Tanh", "Sigmoid", "Add", "Sub", "Mul", "Div", "Neg",
  "Concat", "MaxPool", "AveragePool", "GlobalAveragePool", "BatchNormalization",
]);

export interface SaliencyPath {
  /** the image model input */
  input: string;
  /** nodes on the path, graph order */
  nodes: NodeProto[];
  /** tensors the reverse pass reads (node outputs, and inputs of max-pools) */
  fetch: string[];
  /** set when the path holds an op the reverse pass cannot handle */
  error: string | null;
}

/** find the image path of a model; outputTensor = the pre-softmax output netglow shows */
export function saliencyPath(m: ModelInfo, input: string, outputTensor: string): SaliencyPath {
  const fromInput = new Set<string>([input]);
  for (const n of m.nodes) if (n.inputs.some((i) => fromInput.has(i))) n.outputs.forEach((o) => fromInput.add(o));
  const producer = new Map<string, NodeProto>();
  for (const n of m.nodes) n.outputs.forEach((o) => producer.set(o, n));
  const need = new Set<string>([outputTensor]);
  const onPath = new Set<NodeProto>();
  for (let i = m.nodes.length - 1; i >= 0; i--) {
    const n = m.nodes[i];
    if (!n.outputs.some((o) => need.has(o)) || !n.outputs.some((o) => fromInput.has(o))) continue;
    onPath.add(n);
    n.inputs.forEach((x) => x && fromInput.has(x) && need.add(x));
  }
  const nodes = m.nodes.filter((n) => onPath.has(n));
  const bad = nodes.find((n) => !SUPPORTED.has(n.opType));
  const fetch: string[] = [];
  for (const n of nodes) {
    fetch.push(n.outputs[0]);
    if (n.opType === "MaxPool") fetch.push(n.inputs[0]);
  }
  return {
    input, nodes, fetch: fetch.filter((t) => t !== input),
    error: bad ? `Saliency is not available: the image path runs through a ${bad.opType} (node "${bad.name || bad.outputs[0]}").` : nodes.length ? null : "Saliency is not available: no path from the image input to the outputs.",
  };
}

interface T3 {
  dims: number[];
  size: number;
  data: Float32Array;
}

export interface SaliencyMap {
  H: number;
  W: number;
  /** T x H x W signed grad×input, summed over channels */
  heat: Float32Array;
  /** 99th percentile of |heat| over the recording: the colour scale */
  scale: number;
  /** which output each tick explains */
  target: Int32Array;
}

/**
 * Reverse pass for every tick. tensors: the trace (per-tick dims, batch removed); x: the image input as floats
 * (T x C x H x W); target[t]: the output unit to explain at tick t.
 */
export function computeSaliency(m: ModelInfo, path: SaliencyPath, outputTensor: string, tensors: Map<string, T3>, x: Float32Array, dims: number[], target: Int32Array): SaliencyMap {
  const T = target.length;
  const [C, H, W] = dims;
  const inSize = C * H * W;
  const heat = new Float32Array(T * H * W);
  const sizeOf = (t: string) => (t === path.input ? inSize : tensors.get(t)?.size ?? -1);
  const dimsOf = (t: string) => (t === path.input ? dims : tensors.get(t)!.dims);
  const val = (t: string, k: number) => (t === path.input ? x.subarray(k * inSize, (k + 1) * inSize) : tensors.get(t)!.data.subarray(k * tensors.get(t)!.size, (k + 1) * tensors.get(t)!.size));
  const constOf = (name: string) => m.initializers.get(name);
  const onPath = (t: string) => t === path.input || path.nodes.some((n) => n.outputs[0] === t);
  for (let k = 0; k < T; k++) {
    const g = new Map<string, Float32Array>();
    const go = new Float32Array(sizeOf(outputTensor));
    go[target[k]] = 1;
    g.set(outputTensor, go);
    const acc = (t: string, d: Float32Array) => {
      const e = g.get(t);
      if (!e) g.set(t, d);
      else for (let i = 0; i < d.length; i++) e[i] += d[i];
    };
    for (let ni = path.nodes.length - 1; ni >= 0; ni--) {
      const n = path.nodes[ni];
      const gy = g.get(n.outputs[0]);
      if (!gy) continue;
      const xin = n.inputs[0];
      const op = n.opType;
      if (IDENTITY.has(op)) acc(xin, gy);
      else if (op === "Relu" || op === "LeakyRelu" || op === "Tanh" || op === "Sigmoid") {
        const y = val(n.outputs[0], k);
        const a = n.attrs.get("alpha")?.f ?? 0.01;
        const d = new Float32Array(gy.length);
        for (let i = 0; i < d.length; i++)
          d[i] = gy[i] * (op === "Relu" ? (y[i] > 0 ? 1 : 0) : op === "LeakyRelu" ? (y[i] > 0 ? 1 : a) : op === "Tanh" ? 1 - y[i] * y[i] : y[i] * (1 - y[i]));
        acc(xin, d);
      } else if (op === "Gemm" || op === "MatMul") {
        const B = constOf(n.inputs[1])!;
        const transB = op === "Gemm" && (n.attrs.get("transB")?.i ?? 0) === 1;
        const alpha = op === "Gemm" ? (n.attrs.get("alpha")?.f ?? 1) : 1;
        const [r0, c0] = B.dims;
        const nIn = transB ? c0 : r0;
        const nOut = transB ? r0 : c0;
        const d = new Float32Array(nIn);
        for (let j = 0; j < nOut; j++) {
          const v = gy[j] * alpha;
          if (!v) continue;
          if (transB) for (let i = 0; i < nIn; i++) d[i] += v * B.data[j * nIn + i];
          else for (let i = 0; i < nIn; i++) d[i] += v * B.data[i * nOut + j];
        }
        acc(xin, d);
      } else if (op === "Conv") {
        const w = constOf(n.inputs[1])!;
        const [Co, Ci, kh, kw] = w.dims;
        const [, Hi, Wi] = dimsOf(xin);
        const [, Ho, Wo] = dimsOf(n.outputs[0]);
        const st = n.attrs.get("strides")?.ints ?? [1, 1];
        const pads = n.attrs.get("pads")?.ints ?? [0, 0, 0, 0];
        const dil = n.attrs.get("dilations")?.ints ?? [1, 1];
        const d = new Float32Array(Ci * Hi * Wi);
        for (let o = 0; o < Co; o++)
          for (let oy = 0; oy < Ho; oy++)
            for (let ox = 0; ox < Wo; ox++) {
              const v = gy[(o * Ho + oy) * Wo + ox];
              if (!v) continue;
              for (let c = 0; c < Ci; c++)
                for (let ky = 0; ky < kh; ky++) {
                  const iy = oy * st[0] - pads[0] + ky * dil[0];
                  if (iy < 0 || iy >= Hi) continue;
                  const wo = ((o * Ci + c) * kh + ky) * kw;
                  const io = (c * Hi + iy) * Wi;
                  for (let kx = 0; kx < kw; kx++) {
                    const ix = ox * st[1] - pads[1] + kx * dil[1];
                    if (ix >= 0 && ix < Wi) d[io + ix] += v * w.data[wo + kx];
                  }
                }
            }
        acc(xin, d);
      } else if (op === "Add" || op === "Sub" || op === "Mul" || op === "Div") {
        const ins = n.inputs.filter((i) => onPath(i));
        if (ins.length === 2) {
          // a residual sum (Mul / Div of two signals is not on any supported path)
          acc(ins[0], gy);
          acc(ins[1], op === "Sub" ? gy.map((v) => -v) : gy);
          continue;
        }
        const other = n.inputs.find((i) => i !== ins[0])!;
        const c = constOf(other)?.data;
        const sc = c && c.length === 1 ? c[0] : null;
        if ((op === "Mul" || op === "Div") && sc === null) throw new Error(`Saliency: ${op} "${n.name}" by a non-scalar constant is not supported.`);
        const f = op === "Mul" ? sc! : op === "Div" ? 1 / sc! : op === "Sub" && ins[0] !== n.inputs[0] ? -1 : 1;
        acc(ins[0], f === 1 ? gy : gy.map((v) => v * f));
      } else if (op === "Neg") acc(xin, gy.map((v) => -v));
      else if (op === "Concat") {
        let off = 0;
        for (const i of n.inputs) {
          const sz = sizeOf(i) > 0 ? sizeOf(i) : constOf(i)?.data.length ?? 0;
          if (onPath(i)) acc(i, gy.slice(off, off + sz));
          off += sz;
        }
      } else if (op === "MaxPool" || op === "AveragePool") {
        const [Cc, Hi, Wi] = dimsOf(xin);
        const [, Ho, Wo] = dimsOf(n.outputs[0]);
        const ks = n.attrs.get("kernel_shape")!.ints!;
        const st = n.attrs.get("strides")?.ints ?? ks;
        const pads = n.attrs.get("pads")?.ints ?? [0, 0, 0, 0];
        const xi = op === "MaxPool" ? val(xin, k) : null;
        const d = new Float32Array(Cc * Hi * Wi);
        for (let c = 0; c < Cc; c++)
          for (let oy = 0; oy < Ho; oy++)
            for (let ox = 0; ox < Wo; ox++) {
              const v = gy[(c * Ho + oy) * Wo + ox];
              let best = -1;
              let bv = -Infinity;
              const cells: number[] = [];
              for (let ky = 0; ky < ks[0]; ky++)
                for (let kx = 0; kx < ks[1]; kx++) {
                  const iy = oy * st[0] - pads[0] + ky;
                  const ix = ox * st[1] - pads[1] + kx;
                  if (iy < 0 || iy >= Hi || ix < 0 || ix >= Wi) continue;
                  const p = (c * Hi + iy) * Wi + ix;
                  cells.push(p);
                  if (xi && xi[p] > bv) {
                    bv = xi[p];
                    best = p;
                  }
                }
              if (xi) {
                if (best >= 0) d[best] += v;
              } else for (const p of cells) d[p] += v / (ks[0] * ks[1]);
            }
        acc(xin, d);
      } else if (op === "GlobalAveragePool") {
        const [Cc, Hi, Wi] = dimsOf(xin);
        const d = new Float32Array(Cc * Hi * Wi);
        for (let c = 0; c < Cc; c++) for (let p = 0; p < Hi * Wi; p++) d[c * Hi * Wi + p] = gy[c] / (Hi * Wi);
        acc(xin, d);
      } else if (op === "BatchNormalization") {
        const [Cc, Hi = 1, Wi = 1] = dimsOf(xin);
        const sc = constOf(n.inputs[1])!.data;
        const vr = constOf(n.inputs[4])!.data;
        const eps = n.attrs.get("epsilon")?.f ?? 1e-5;
        const hw = Hi * Wi;
        const d = new Float32Array(gy.length);
        for (let c = 0; c < Cc; c++) for (let p = 0; p < hw; p++) d[c * hw + p] = (gy[c * hw + p] * sc[c]) / Math.sqrt(vr[c] + eps);
        acc(xin, d);
      }
    }
    const gx = g.get(path.input);
    if (!gx) continue;
    const xi = x.subarray(k * inSize, (k + 1) * inSize);
    const o = k * H * W;
    for (let c = 0; c < C; c++) for (let p = 0; p < H * W; p++) heat[o + p] += gx[c * H * W + p] * xi[c * H * W + p];
  }
  // colour scale: 99th percentile of |heat| over the recording (sampled)
  const step = Math.max(1, Math.floor(heat.length / 200000));
  const s: number[] = [];
  for (let i = 0; i < heat.length; i += step) s.push(Math.abs(heat[i]));
  s.sort((a, b) => a - b);
  const scale = Math.max(s[Math.floor(0.99 * (s.length - 1))] ?? 0, 1e-12);
  return { H, W, heat, scale, target };
}
