// Turns an ONNX graph into a "plan": the straight chain of weighted layers from the input to the output,
// which tensors to expose for each layer, and the weights needed for attribution.
//
// Vocabulary:
//   weighted layer - a node that multiplies by learned weights: Gemm / MatMul (dense) or Conv
//   activation     - the tensor a layer hands to the next weighted layer (after its bias, nonlinearity, pooling)
//   data input     - the input of a node that carries the signal, as opposed to its constant weights / shape args
//   chain          - v0 supports graphs where every node has exactly one data input (MLPs, CNN encoder + MLP head)

import { ELEM, type ModelInfo, type NodeProto, type TensorProto } from "./onnx/proto";

export interface ConvGeom {
  kh: number;
  kw: number;
  sh: number;
  sw: number;
  ph: number; // top / left pad (ONNX pads = [top, left, bottom, right])
  pw: number;
  dh: number;
  dw: number;
}

export interface Layer {
  kind: "dense" | "conv";
  node: NodeProto;
  /** the tensor feeding this layer (== previous layer's actTensor up to a reshape, or the model input) */
  inTensor: string;
  /** the tensor shown as this layer's nodes */
  actTensor: string;
  /** dense: [out][in] row-major; conv: [Cout][Cin][kh][kw] */
  W: Float32Array;
  out: number;
  in: number;
  conv?: ConvGeom;
  label: string;
}

export interface Plan {
  inputName: string;
  inputElemType: number;
  inputShape: (number | string)[];
  layers: Layer[]; // the last one produces the outputs
  /** pre-softmax output tensor (== last layer's actTensor) */
  outputTensor: string;
  graphOutput: string;
  softmaxInGraph: boolean;
  inputIsImage: boolean;
  warnings: string[];
  /** extra tensors to request from onnxruntime */
  fetch: string[];
}

const RESHAPE = new Set(["Reshape", "Flatten", "Squeeze", "Unsqueeze", "Identity", "Dropout"]);
const UNARY = new Set([
  ...RESHAPE, "Transpose", "Cast", "Slice", "Pad", "Relu", "LeakyRelu", "PRelu", "Elu", "Selu", "Celu", "Sigmoid", "Tanh",
  "Softplus", "Softsign", "HardSigmoid", "HardSwish", "Gelu", "Mish", "Clip", "Softmax", "LogSoftmax", "BatchNormalization",
  "InstanceNormalization", "LayerNormalization", "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool", "ReduceMean",
  "ReduceMax", "ReduceSum", "Neg", "Abs", "Exp", "Log", "Sqrt", "Erf", "Floor", "Ceil", "Round", "Gemm", "Conv", "LpNormalization",
]);
const BINARY = new Set(["Add", "Sub", "Mul", "Div", "MatMul", "Pow", "Max", "Min"]);

export class UnsupportedGraph extends Error {}

export function buildPlan(m: ModelInfo): Plan {
  const warnings: string[] = [];
  const consts = new Set(m.initializers.keys());
  for (const n of m.nodes) {
    const ins = n.inputs.filter(Boolean);
    if (n.opType === "Constant" || (ins.length && ins.every((i) => consts.has(i)) && n.opType !== "Shape")) n.outputs.forEach((o) => consts.add(o));
  }
  const realInputs = m.inputs.filter((i) => !m.initializers.has(i.name));
  if (realInputs.length !== 1)
    throw new UnsupportedGraph(`netglow v0 needs a model with exactly one input tensor; this one has ${realInputs.length} (${realInputs.map((i) => i.name).join(", ")}).`);
  const input = realInputs[0];
  if (!m.outputs.length) throw new UnsupportedGraph("The model declares no outputs.");
  const graphOutput = m.outputs[0].name;
  if (m.outputs.length > 1) warnings.push(`The model has ${m.outputs.length} outputs; netglow shows the first one (${graphOutput}).`);

  const producer = new Map<string, NodeProto>();
  for (const n of m.nodes) n.outputs.forEach((o) => producer.set(o, n));

  const dataInputs = (n: NodeProto): string[] => {
    const ins = n.inputs.filter(Boolean);
    if (UNARY.has(n.opType)) return ins[0] && !consts.has(ins[0]) ? [ins[0]] : [];
    if (BINARY.has(n.opType)) return ins.filter((i) => !consts.has(i));
    throw new UnsupportedGraph(`netglow v0 does not support the "${n.opType}" op on the signal path (node "${n.name || n.outputs[0]}"). Supported: MLPs and CNN encoder + MLP head chains.`);
  };

  // walk back from the output along the data path
  const chain: NodeProto[] = [];
  let t = graphOutput;
  const seen = new Set<string>();
  while (t !== input.name) {
    if (seen.has(t)) throw new UnsupportedGraph("Cycle in graph.");
    seen.add(t);
    const n = producer.get(t);
    if (!n) throw new UnsupportedGraph(`Tensor "${t}" has no producer and is not the model input.`);
    const d = dataInputs(n);
    if (d.length !== 1)
      throw new UnsupportedGraph(`netglow v0 supports chain-shaped networks only. Node "${n.name || n.outputs[0]}" (${n.opType}) ${d.length ? "joins two branches (a skip connection, concat or gating)" : "does not depend on the input"}.`);
    chain.push(n);
    t = d[0];
  }
  chain.reverse();

  const isWeighted = (n: NodeProto) =>
    n.opType === "Conv" || n.opType === "Gemm" || (n.opType === "MatMul" && n.inputs.some((i) => m.initializers.has(i)));
  const wIdx = chain.map((n, i) => (isWeighted(n) ? i : -1)).filter((i) => i >= 0);
  if (!wIdx.length) throw new UnsupportedGraph("No weighted layers (Gemm, MatMul or Conv) found on the signal path.");

  const dataIn = (n: NodeProto) => dataInputs(n)[0];
  const layers: Layer[] = [];
  let softmaxInGraph = false;
  for (let k = 0; k < wIdx.length; k++) {
    const n = chain[wIdx[k]];
    const end = k + 1 < wIdx.length ? wIdx[k + 1] : chain.length;
    // shown tensor: the last non-reshape node before the next weighted one (for the last layer, before a softmax)
    let a = wIdx[k];
    for (let i = wIdx[k] + 1; i < end; i++) {
      const op = chain[i].opType;
      if (k + 1 === wIdx.length && (op === "Softmax" || op === "LogSoftmax")) {
        softmaxInGraph = true;
        break;
      }
      if (op === "Transpose" && k + 1 < wIdx.length) warnings.push(`A Transpose sits between layer ${k + 1} and ${k + 2}; edge attribution there assumes the original element order.`);
      if (!RESHAPE.has(op)) a = i;
    }
    const actTensor = chain[a].outputs[0];
    layers.push(weightedLayer(n, dataIn(n), actTensor, m, k, wIdx.length));
  }
  const last = layers[layers.length - 1];
  if (last.kind !== "dense") throw new UnsupportedGraph("The last weighted layer is a convolution; netglow v0 expects a dense output head.");
  for (let k = 0; k + 1 < layers.length; k++) {
    const next = layers[k + 1];
    if (layers[k].kind === "dense" && next.kind === "conv") throw new UnsupportedGraph("A dense layer feeds a convolution (decoder-style); not supported in v0.");
  }
  const inputIsImage = input.shape?.length === 4;
  if (input.shape && input.shape.length > 4) throw new UnsupportedGraph(`Input rank ${input.shape.length} is not supported (use [N, features] or [N, C, H, W]).`);
  if (input.shape?.length === 3) throw new UnsupportedGraph("Sequence inputs [N, T, F] are not supported in v0.");
  if (inputIsImage && layers[0].kind !== "conv" && layers[0].kind !== "dense") warnings.push("Image input without a leading conv.");
  if (![ELEM.FLOAT, ELEM.UINT8, ELEM.INT8, ELEM.DOUBLE, ELEM.INT64, ELEM.INT32].includes(input.elemType as never))
    throw new UnsupportedGraph(`Input element type ${input.elemType} is not supported.`);

  const fetch = layers.map((l) => l.actTensor);
  if (layers[0].inTensor !== input.name) fetch.push(layers[0].inTensor);
  if (!fetch.includes(graphOutput)) fetch.push(graphOutput);
  return {
    inputName: input.name,
    inputElemType: input.elemType,
    inputShape: input.shape ?? [],
    layers,
    outputTensor: last.actTensor,
    graphOutput,
    softmaxInGraph,
    inputIsImage,
    warnings,
    fetch,
  };
}

function weightedLayer(n: NodeProto, inTensor: string, actTensor: string, m: ModelInfo, k: number, nLayers: number): Layer {
  const lastLayer = k + 1 === nLayers;
  const need = (name: string): TensorProto => {
    const t = m.initializers.get(name);
    if (!t || !t.data.length) throw new UnsupportedGraph(`Weights "${name}" of node "${n.name}" are not a readable constant.`);
    return t;
  };
  if (n.opType === "Conv") {
    const w = need(n.inputs[1]);
    const [cout, cinG, kh, kw] = w.dims;
    const group = n.attrs.get("group")?.i ?? 1;
    if (group !== 1) throw new UnsupportedGraph(`Grouped / depthwise convolution (group=${group}) is not supported in v0.`);
    if (n.attrs.get("auto_pad")?.s && n.attrs.get("auto_pad")!.s !== "NOTSET")
      throw new UnsupportedGraph("Conv auto_pad is not supported; export with explicit pads.");
    const st = n.attrs.get("strides")?.ints ?? [1, 1];
    const pads = n.attrs.get("pads")?.ints ?? [0, 0, 0, 0];
    const dil = n.attrs.get("dilations")?.ints ?? [1, 1];
    return {
      kind: "conv", node: n, inTensor, actTensor, W: w.data, out: cout, in: cinG,
      conv: { kh, kw, sh: st[0], sw: st[1], ph: pads[0], pw: pads[1], dh: dil[0], dw: dil[1] },
      label: lastLayer ? "output" : `conv ${k + 1}`,
    };
  }
  let W: Float32Array;
  let out: number;
  let inn: number;
  if (n.opType === "Gemm") {
    if ((n.attrs.get("transA")?.i ?? 0) !== 0) throw new UnsupportedGraph("Gemm with transA is not supported.");
    const b = need(n.inputs[1]);
    const transB = (n.attrs.get("transB")?.i ?? 0) === 1;
    const alpha = n.attrs.get("alpha")?.f ?? 1;
    if (transB) {
      [out, inn] = b.dims;
      W = alpha === 1 ? b.data : b.data.map((x) => x * alpha);
    } else {
      [inn, out] = b.dims;
      W = transpose(b.data, inn, out, alpha);
    }
  } else {
    // MatMul(x, W) with W [in, out]
    const wi = n.inputs.findIndex((i) => m.initializers.has(i));
    if (wi !== 1) throw new UnsupportedGraph("MatMul with the weight on the left is not supported.");
    const b = need(n.inputs[1]);
    if (b.dims.length !== 2) throw new UnsupportedGraph("Batched MatMul weights are not supported.");
    [inn, out] = b.dims;
    W = transpose(b.data, inn, out, 1);
  }
  return { kind: "dense", node: n, inTensor, actTensor, W, out, in: inn, label: lastLayer ? "output" : `dense ${k + 1}` };
}

function transpose(src: Float32Array, rows: number, cols: number, scale: number): Float32Array {
  const o = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) o[c * rows + r] = src[r * cols + c] * scale;
  return o;
}
