// Turns an ONNX graph into a "plan": the weighted layers from the input(s) to the output,
// which tensors to expose for each layer, and the weights needed for attribution.
//
// Vocabulary:
//   weighted layer - a node that multiplies by learned weights: Gemm / MatMul (dense) or Conv
//   activation     - the tensor a layer hands to the next weighted layer (after its bias, nonlinearity, pooling)
//   data input     - the input of a node that carries the signal, as opposed to its constant weights / shape args
//   chain          - a path where every node has exactly one data input (MLPs, CNN encoder + MLP head)
//   branch / merge - several model inputs, each with its own chain (a conv stack, dense layers, or nothing),
//                    joined by one Concat whose output runs through a trunk chain of dense layers to the outputs

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

export interface PlanInput {
  name: string;
  elemType: number;
  shape: (number | string)[];
  /** [N, C, H, W] */
  isImage: boolean;
}

/** the path from one model input to the merge (or, for chain models, the input alone) */
export interface Branch {
  /** index into plan.inputs */
  input: number;
  /** weighted layers between the input and the merge (empty when the input goes straight in) */
  layers: Layer[];
  /** the tensor this branch hands to the Concat (null for chain models) */
  concatIn: string | null;
}

export interface Plan {
  inputs: PlanInput[];
  /** one per model input, in Concat order (chain models: one branch with no layers) */
  branches: Branch[];
  /** the Concat that joins the branches; null for chain models */
  merge: { node: NodeProto; name: string } | null;
  /** weighted layers after the merge (chain models: all of them); the last one produces the outputs */
  trunk: Layer[];
  /** every weighted layer: branch layers in branch order, then the trunk */
  layers: Layer[];
  /** pre-softmax output tensor (== last layer's actTensor) */
  outputTensor: string;
  graphOutput: string;
  softmaxInGraph: boolean;
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
const BINARY = new Set(["Add", "Sub", "Mul", "Div", "MatMul", "Pow", "Max", "Min", "Concat"]);

export class UnsupportedGraph extends Error {}

export function buildPlan(m: ModelInfo): Plan {
  const warnings: string[] = [];
  const consts = new Set(m.initializers.keys());
  for (const n of m.nodes) {
    const ins = n.inputs.filter(Boolean);
    if (n.opType === "Constant" || (ins.length && ins.every((i) => consts.has(i)) && n.opType !== "Shape")) n.outputs.forEach((o) => consts.add(o));
  }
  const realInputs = m.inputs.filter((i) => !m.initializers.has(i.name));
  if (!realInputs.length) throw new UnsupportedGraph("The model declares no input tensor.");
  if (!m.outputs.length) throw new UnsupportedGraph("The model declares no outputs.");
  const graphOutput = m.outputs[0].name;
  if (m.outputs.length > 1) warnings.push(`The model has ${m.outputs.length} outputs; netglow shows the first one (${graphOutput}).`);
  const inputIdx = new Map(realInputs.map((v, i) => [v.name, i]));

  const producer = new Map<string, NodeProto>();
  for (const n of m.nodes) n.outputs.forEach((o) => producer.set(o, n));

  const dataInputs = (n: NodeProto): string[] => {
    const ins = n.inputs.filter(Boolean);
    if (UNARY.has(n.opType)) return ins[0] && !consts.has(ins[0]) ? [ins[0]] : [];
    if (BINARY.has(n.opType)) return ins.filter((i) => !consts.has(i));
    throw new UnsupportedGraph(`netglow does not support the "${n.opType}" op on the signal path (node "${n.name || n.outputs[0]}"). Supported: MLPs, CNN encoder + MLP head chains, and several inputs joined by one Concat.`);
  };
  const nodeName = (n: NodeProto) => n.name || n.outputs[0];

  // walk back from tensor t along the data path until a model input or a node that joins several signals
  const seen = new Set<string>();
  const walk = (t: string): { chain: NodeProto[]; input: string | null; join: NodeProto | null } => {
    const chain: NodeProto[] = [];
    while (!inputIdx.has(t)) {
      if (seen.has(t)) throw new UnsupportedGraph("Cycle in graph, or one tensor feeds two branches.");
      seen.add(t);
      const n = producer.get(t);
      if (!n) throw new UnsupportedGraph(`Tensor "${t}" has no producer and is not a model input.`);
      const d = dataInputs(n);
      if (!d.length) throw new UnsupportedGraph(`Node "${nodeName(n)}" (${n.opType}) does not depend on any input.`);
      if (d.length > 1) return { chain: chain.reverse(), input: null, join: n };
      chain.push(n);
      t = d[0];
    }
    return { chain: chain.reverse(), input: t, join: null };
  };

  const top = walk(graphOutput);
  const chains: { nodes: NodeProto[]; input: number; concatIn: string | null }[] = [];
  let merge: Plan["merge"] = null;
  if (top.join) {
    const j = top.join;
    if (j.opType !== "Concat")
      throw new UnsupportedGraph(`netglow supports chain-shaped networks, or several inputs joined by one Concat. Node "${nodeName(j)}" (${j.opType}) joins two branches (a skip connection or gating).`);
    const axis = j.attrs.get("axis")?.i ?? 1;
    if (axis !== 1 && axis !== -1) throw new UnsupportedGraph(`Concat "${nodeName(j)}" joins along axis ${axis}; netglow expects flat features joined along axis 1.`);
    merge = { node: j, name: nodeName(j) };
    for (const t of dataInputs(j)) {
      const b = walk(t);
      if (b.join) throw new UnsupportedGraph(`Node "${nodeName(b.join)}" (${b.join.opType}) joins signals inside a branch; netglow supports one Concat that joins whole inputs.`);
      chains.push({ nodes: b.chain, input: inputIdx.get(b.input!)!, concatIn: t });
    }
    const used = new Set(chains.map((c) => c.input));
    if (used.size !== chains.length) throw new UnsupportedGraph("One input feeds the Concat twice; netglow expects one branch per input.");
    if (used.size !== realInputs.length) throw new UnsupportedGraph(`The model has ${realInputs.length} inputs but only ${used.size} reach the output.`);
  } else {
    if (realInputs.length !== 1)
      throw new UnsupportedGraph(`The model has ${realInputs.length} inputs but only "${top.input}" reaches the output.`);
    chains.push({ nodes: [], input: 0, concatIn: null });
  }
  const trunkNodes = top.chain;

  const isWeighted = (n: NodeProto) =>
    n.opType === "Conv" || n.opType === "Gemm" || (n.opType === "MatMul" && n.inputs.some((i) => m.initializers.has(i)));
  const dataIn = (n: NodeProto) => dataInputs(n)[0];
  const nTotal = [...chains.map((c) => c.nodes), trunkNodes].reduce((a, ns) => a + ns.filter(isWeighted).length, 0);
  let k = 0;
  let softmaxInGraph = false;
  // weighted layers of one chain; shown tensor = the last non-reshape node before the next weighted one
  // (for the very last layer: before a softmax)
  const layersOf = (chain: NodeProto[], isTrunk: boolean): Layer[] => {
    const wIdx = chain.map((n, i) => (isWeighted(n) ? i : -1)).filter((i) => i >= 0);
    const out: Layer[] = [];
    for (let q = 0; q < wIdx.length; q++) {
      const n = chain[wIdx[q]];
      const end = q + 1 < wIdx.length ? wIdx[q + 1] : chain.length;
      const lastOverall = isTrunk && q + 1 === wIdx.length;
      let a = wIdx[q];
      for (let i = wIdx[q] + 1; i < end; i++) {
        const op = chain[i].opType;
        if (lastOverall && (op === "Softmax" || op === "LogSoftmax")) {
          softmaxInGraph = true;
          break;
        }
        if (op === "Transpose" && q + 1 < wIdx.length) warnings.push(`A Transpose sits between layer ${k + 1} and ${k + 2}; edge attribution there assumes the original element order.`);
        if (!RESHAPE.has(op)) a = i;
      }
      out.push(weightedLayer(n, dataIn(n), chain[a].outputs[0], m, k++, nTotal));
    }
    for (let q = 0; q + 1 < out.length; q++)
      if (out[q].kind === "dense" && out[q + 1].kind === "conv") throw new UnsupportedGraph("A dense layer feeds a convolution (decoder-style); not supported.");
    return out;
  };
  const branches: Branch[] = chains.map((c) => ({ input: c.input, layers: layersOf(c.nodes, false), concatIn: c.concatIn }));
  const trunk = layersOf(trunkNodes, true);
  if (!trunk.length) throw new UnsupportedGraph(merge ? "The Concat must feed a dense layer (Gemm or MatMul)." : "No weighted layers (Gemm, MatMul or Conv) found on the signal path.");
  const last = trunk[trunk.length - 1];
  if (last.kind !== "dense") throw new UnsupportedGraph("The last weighted layer is a convolution; netglow expects a dense output head.");
  if (merge && trunk[0].kind !== "dense") throw new UnsupportedGraph("A convolution right after the Concat is not supported; join flat features into a dense layer.");
  if (merge && trunkNodes.slice(0, trunkNodes.indexOf(trunk[0].node)).some((n) => !RESHAPE.has(n.opType)))
    warnings.push("An elementwise op sits between the Concat and the first dense layer; edges into it use that op's output.");
  for (const b of branches) {
    const l = b.layers.at(-1);
    if (l && l.kind === "dense" && trunk[0].kind === "conv") throw new UnsupportedGraph("A dense layer feeds a convolution (decoder-style); not supported.");
  }

  const inputs: PlanInput[] = realInputs.map((v) => {
    if (v.shape && v.shape.length > 4) throw new UnsupportedGraph(`Input "${v.name}" has rank ${v.shape.length}; use [N, features] or [N, C, H, W].`);
    if (v.shape?.length === 3) throw new UnsupportedGraph(`Input "${v.name}": sequence inputs [N, T, F] are not supported.`);
    if (![ELEM.FLOAT, ELEM.UINT8, ELEM.INT8, ELEM.DOUBLE, ELEM.INT64, ELEM.INT32].includes(v.elemType as never))
      throw new UnsupportedGraph(`Input "${v.name}" has element type ${v.elemType}, which is not supported.`);
    return { name: v.name, elemType: v.elemType, shape: v.shape ?? [], isImage: v.shape?.length === 4 };
  });
  if (inputs.filter((i) => i.isImage).length > 1) throw new UnsupportedGraph("netglow shows one image input per model; this one has several.");

  const layers = [...branches.flatMap((b) => b.layers), ...trunk];
  const fetch = layers.map((l) => l.actTensor);
  const add = (t: string | null) => t && !inputIdx.has(t) && !fetch.includes(t) && fetch.push(t);
  for (const b of branches) {
    add(b.layers[0]?.inTensor ?? null);
    add(b.concatIn);
  }
  add(trunk[0].inTensor);
  add(graphOutput);
  return { inputs, branches, merge, trunk, layers, outputTensor: last.actTensor, graphOutput, softmaxInGraph, warnings, fetch };
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
