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
//   skip / block   - a residual connection: Add(block(x), x) or Add(block(x), P(x)). The block is the main path
//                    (one or more weighted layers); the skip is the identity, or one projection layer P
//   recurrent cell - an LSTM or GRU node stepped once per tick; its hidden state comes in as a model input
//                    (initial_h, initial_c) and goes out as a graph output, and the run threads it from tick to tick

import { ELEM, type ModelInfo, type NodeProto, type TensorProto } from "./onnx/proto";
import { saliencyPath, type SaliencyPath } from "./saliency";

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
  /** rnn = an LSTM / GRU cell; W then holds the input weights of its candidate (new-content) gate */
  kind: "dense" | "conv" | "rnn";
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
  rnn?: RnnCell;
  label: string;
}

/** an LSTM / GRU cell, weights in ONNX gate order (LSTM: i o f c, GRU: z r h), one direction */
export interface RnnCell {
  type: "lstm" | "gru";
  H: number;
  I: number;
  gates: number;
  /** [gates*H][I] input weights, [gates*H][H] recurrent weights, [2*gates*H] biases (Wb then Rb) */
  Wx: Float32Array;
  R: Float32Array;
  B: Float32Array;
  linearBeforeReset: boolean;
  /** index of the candidate gate in the gate order (LSTM c = 3, GRU h = 2) */
  cand: number;
  /** recurrent weights of the candidate gate, [H][H] */
  Rc: Float32Array;
  /** the model inputs that feed the initial state, and the graph outputs that hand the next state back */
  hIn: string;
  hOut: string;
  cIn: string | null;
  cOut: string | null;
}

/** a residual connection: the skip leaves column `from` (null = the branch input) and joins column `to` */
export interface Skip {
  add: NodeProto;
  to: Layer;
  from: Layer | null;
  /** branch index for skips inside a branch, -1 for the trunk (and chain models) */
  branch: number;
  /** the tensor the skip reads: the block input, or the projection's input */
  src: string;
  /** a projection layer on the skip path; null = identity */
  proj: Layer | null;
}

/** a recurrent state threaded across ticks: model input <- graph output of the previous tick */
export interface StateIO {
  input: string;
  output: string;
  /** per-tick feed shape with the batch axis set to 1, e.g. [1, 1, H] */
  dims: number[];
  size: number;
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
  /** residual connections, in graph order */
  skips: Skip[];
  /** recurrent states the run threads from tick to tick (empty for feed-forward models) */
  states: StateIO[];
  /** per-tick dims for tensors whose batch axis is not the first one (an LSTM's Y is [seq, dirs, N, H]) */
  tickDims: Map<string, number[]>;
  /** image models: the path the saliency reverse pass walks (null without an image input) */
  saliency: SaliencyPath | null;
}

const RESHAPE = new Set(["Reshape", "Flatten", "Squeeze", "Unsqueeze", "Identity", "Dropout"]);
const RNN = new Set(["LSTM", "GRU"]);
const UNARY = new Set([
  ...RESHAPE, "Transpose", "Cast", "Slice", "Pad", "Relu", "LeakyRelu", "PRelu", "Elu", "Selu", "Celu", "Sigmoid", "Tanh",
  "Softplus", "Softsign", "HardSigmoid", "HardSwish", "Gelu", "Mish", "Clip", "Softmax", "LogSoftmax", "BatchNormalization",
  "InstanceNormalization", "LayerNormalization", "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool", "ReduceMean",
  "ReduceMax", "ReduceSum", "Neg", "Abs", "Exp", "Log", "Sqrt", "Erf", "Floor", "Ceil", "Round", "Gemm", "Conv", "LpNormalization",
  "Gather", ...RNN,
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
  const nodeName = (n: NodeProto) => n.name || n.outputs[0];
  const producer = new Map<string, NodeProto>();
  for (const n of m.nodes) n.outputs.forEach((o) => producer.set(o, n));
  const declared = m.inputs.filter((i) => !m.initializers.has(i.name));
  if (!m.outputs.length) throw new UnsupportedGraph("The model declares no outputs.");
  const graphOutput = m.outputs[0].name;

  // ---- recurrent cells: their initial-state inputs are model inputs that carry state, not observations
  const declaredNames = new Set(declared.map((v) => v.name));
  const passThrough = (t: string, dir: "up" | "down"): string => {
    // follow Identity-like nodes from a cell's state slot to the model input (up) or graph output (down)
    for (let i = 0; i < 8; i++) {
      if (dir === "up") {
        if (declaredNames.has(t)) return t;
        const p = producer.get(t);
        if (!p || !RESHAPE.has(p.opType)) return t;
        t = p.inputs[0];
      } else {
        if (m.outputs.some((o) => o.name === t)) return t;
        const c = m.nodes.find((n) => RESHAPE.has(n.opType) && n.inputs[0] === t);
        if (!c) return t;
        t = c.outputs[0];
      }
    }
    return t;
  };
  const stateInputs = new Set<string>();
  const states: StateIO[] = [];
  const cellState = new Map<NodeProto, { hIn: string; hOut: string; cIn: string | null; cOut: string | null }>();
  for (const n of m.nodes.filter((x) => RNN.has(x.opType))) {
    const lstm = n.opType === "LSTM";
    const slots: [number, number][] = lstm ? [[5, 1], [6, 2]] : [[5, 1]];
    const io = slots.map(([si, so]) => {
      const inp = n.inputs[si] ? passThrough(n.inputs[si], "up") : "";
      const out = n.outputs[so] ? passThrough(n.outputs[so], "down") : "";
      if (!inp || !declaredNames.has(inp) || !out || !m.outputs.some((o) => o.name === out))
        throw new UnsupportedGraph(
          `${n.opType} "${nodeName(n)}" must take its initial ${so === 1 ? "hidden" : "cell"} state as a model input and return the new state as a graph output, so netglow can carry it from tick to tick (export forward(obs, h0${lstm ? ", c0" : ""}) -> (logits, h${lstm ? ", c" : ""})).`,
        );
      const v = declared.find((d) => d.name === inp)!;
      const dims = (v.shape ?? []).map((d) => (typeof d === "number" && d > 0 ? d : 1));
      stateInputs.add(inp);
      states.push({ input: inp, output: out, dims, size: dims.reduce((a, b) => a * b, 1) });
      return { inp, out };
    });
    cellState.set(n, { hIn: io[0].inp, hOut: io[0].out, cIn: io[1]?.inp ?? null, cOut: io[1]?.out ?? null });
  }
  const realInputs = declared.filter((i) => !stateInputs.has(i.name));
  if (!realInputs.length) throw new UnsupportedGraph("The model declares no input tensor.");
  const shown = m.outputs.filter((o) => !states.some((s) => s.output === o.name));
  if (shown.length > 1) warnings.push(`The model has ${shown.length} outputs; netglow shows the first one (${graphOutput}).`);
  const inputIdx = new Map(realInputs.map((v, i) => [v.name, i]));
  for (const s of stateInputs) consts.add(s);

  const dataInputs = (n: NodeProto): string[] => {
    const ins = n.inputs.filter(Boolean);
    if (UNARY.has(n.opType)) return ins[0] && !consts.has(ins[0]) ? [ins[0]] : [];
    if (BINARY.has(n.opType)) return ins.filter((i) => !consts.has(i));
    throw new UnsupportedGraph(`netglow does not support the "${n.opType}" op on the signal path (node "${nodeName(n)}"). Supported: MLPs, CNNs, residual blocks, LSTM / GRU cells, and several inputs joined by one Concat.`);
  };
  const isWeighted = (n: NodeProto) =>
    n.opType === "Conv" || n.opType === "Gemm" || RNN.has(n.opType) || (n.opType === "MatMul" && n.inputs.some((i) => m.initializers.has(i)));

  // tensors walking back from t along single data inputs: [t, its source, ...] up to a model input or a join
  const backChain = (t: string): string[] => {
    const out = [t];
    for (;;) {
      if (inputIdx.has(t)) return out;
      const n = producer.get(t);
      if (!n) return out;
      const d = dataInputs(n);
      if (d.length !== 1) return out;
      t = d[0];
      if (out.includes(t)) return out;
      out.push(t);
    }
  };
  // nodes producing chain[0 .. upto), in forward order
  const nodesOn = (chain: string[], upto: number) => chain.slice(0, upto).map((t) => producer.get(t)!).reverse();

  type Residual = { add: NodeProto; t0: string; mainIn: string; mainWeighted: number; skipNodes: NodeProto[]; between: NodeProto[] };
  /** an Add of a block and its skip; null when the two sides do not meet (two separate signals added) */
  const residual = (n: NodeProto, d: string[]): Residual | null => {
    if (d.length !== 2) return null;
    const [ca, cb] = [backChain(d[0]), backChain(d[1])];
    const sa = new Set(ca);
    const ib = cb.findIndex((x) => sa.has(x));
    if (ib < 0) return null;
    const t0 = cb[ib];
    const ia = ca.indexOf(t0);
    const na = nodesOn(ca, ia);
    const nb = nodesOn(cb, ib);
    const wa = na.filter(isWeighted).length;
    const wb = nb.filter(isWeighted).length;
    const mainA = wa > wb || (wa === wb && na.length >= nb.length);
    const [main, skip, mainIn, mw, sw] = mainA ? [na, nb, d[0], wa, wb] : [nb, na, d[1], wb, wa];
    if (mw === 0) throw new UnsupportedGraph(`Add "${nodeName(n)}" sums two paths without weighted layers; netglow supports residual blocks Add(block(x), x).`);
    if (sw > 1) throw new UnsupportedGraph(`Both inputs of Add "${nodeName(n)}" run through several weighted layers (two parallel branches). netglow supports skips that are the identity or one projection layer.`);
    for (const k of skip) {
      if (isWeighted(k) && RNN.has(k.opType)) throw new UnsupportedGraph(`A recurrent cell on the skip path of Add "${nodeName(n)}" is not supported.`);
      if (!isWeighted(k) && !RESHAPE.has(k.opType) && k.opType !== "BatchNormalization")
        throw new UnsupportedGraph(`The skip into Add "${nodeName(n)}" runs through a ${k.opType} (node "${nodeName(k)}"); netglow supports identity skips and one projection layer (Gemm, MatMul or Conv).`);
      if (k.opType === "BatchNormalization") warnings.push(`A BatchNormalization sits on the skip into "${nodeName(n)}"; skip edges are the projection's contribution before it.`);
    }
    // ops between the block's last weighted layer and the Add: anything but a reshape makes the block edges local
    const lastW = main.map((x, i) => (isWeighted(x) ? i : -1)).filter((i) => i >= 0).at(-1)!;
    const between = main.slice(lastW + 1).filter((x) => !RESHAPE.has(x.opType));
    return { add: n, t0, mainIn, mainWeighted: mw, skipNodes: skip, between };
  };

  // walk back from tensor t along the data path until a model input or a node that joins several signals;
  // residual Adds are stepped over along the block (their skip is recorded, the walk continues on the main path)
  const seen = new Set<string>();
  const residuals: Residual[] = [];
  const walk = (t: string): { chain: NodeProto[]; input: string | null; join: NodeProto | null } => {
    const chain: NodeProto[] = [];
    while (!inputIdx.has(t)) {
      if (seen.has(t)) throw new UnsupportedGraph("Cycle in graph, or one tensor feeds two branches.");
      seen.add(t);
      const n = producer.get(t);
      if (!n) throw new UnsupportedGraph(`Tensor "${t}" has no producer and is not a model input.`);
      const d = dataInputs(n);
      if (!d.length) throw new UnsupportedGraph(`Node "${nodeName(n)}" (${n.opType}) does not depend on any input.`);
      if (d.length > 1) {
        const r = n.opType === "Add" || n.opType === "Sum" ? residual(n, d) : null;
        if (!r) return { chain: chain.reverse(), input: null, join: n };
        residuals.push(r);
        chain.push(n);
        t = r.mainIn;
        continue;
      }
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
      throw new UnsupportedGraph(`netglow supports chains, residual blocks, and several inputs joined by one Concat. Node "${nodeName(j)}" (${j.opType}) joins two signals in another way (gating, attention, or two separate inputs added).`);
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

  const dataIn = (n: NodeProto) => dataInputs(n)[0];
  const nTotal = [...chains.map((c) => c.nodes), trunkNodes].reduce((a, ns) => a + ns.filter(isWeighted).length, 0);
  let k = 0;
  let softmaxInGraph = false;
  const skips: Skip[] = [];
  const tickDims = new Map<string, number[]>();
  // weighted layers of one chain; shown tensor = the last non-reshape node before the next weighted one
  // (for the very last layer: before a softmax)
  const layersOf = (chain: NodeProto[], isTrunk: boolean, branch: number): Layer[] => {
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
      const l = weightedLayer(n, dataIn(n), chain[a].outputs[0], m, k++, nTotal, cellState.get(n));
      // an LSTM / GRU output Y is [seq, dirs, N, H]: one tick is its last axis
      if (l.kind === "rnn" && l.actTensor === n.outputs[0]) tickDims.set(l.actTensor, [l.out]);
      out.push(l);
    }
    for (let q = 0; q + 1 < out.length; q++)
      if (out[q].kind !== "conv" && out[q + 1].kind === "conv") throw new UnsupportedGraph("A dense layer feeds a convolution (decoder-style); not supported.");
    // residual blocks of this chain: the skip joins the column of the block's last layer
    for (const r of residuals) {
      const pos = chain.indexOf(r.add);
      if (pos < 0) continue;
      const q = wIdx.filter((i) => i < pos).length - 1;
      const qf = q - r.mainWeighted;
      if (q < 0) throw new UnsupportedGraph(`Add "${nodeName(r.add)}" comes before any weighted layer.`);
      if (qf < 0 && isTrunk && merge) throw new UnsupportedGraph(`A skip from the Concat output (Add "${nodeName(r.add)}") is not supported.`);
      const pn = r.skipNodes.find(isWeighted);
      const proj = pn ? weightedLayer(pn, dataIn(pn), pn.outputs[0], m, -1, nTotal) : null;
      if (proj) proj.label = "projection";
      if (proj && proj.kind !== out[q].kind && !(proj.kind === "dense" && out[q].kind === "dense"))
        throw new UnsupportedGraph(`The projection on the skip into "${nodeName(r.add)}" is a ${proj.kind} layer but the block ends in a ${out[q].kind} layer.`);
      if (r.between.length)
        warnings.push(`A ${r.between.map((x) => x.opType).join(" + ")} sits between ${out[q].label} and the skip's Add; edges from the block are contributions to ${out[q].label}'s pre-activation, skip edges to the sum.`);
      skips.push({ add: r.add, to: out[q], from: qf >= 0 ? out[qf] : null, branch: isTrunk ? -1 : branch, src: proj ? proj.inTensor : r.t0, proj });
    }
    return out;
  };
  const branches: Branch[] = chains.map((c, bi) => ({ input: c.input, layers: layersOf(c.nodes, false, bi), concatIn: c.concatIn }));
  const trunk = layersOf(trunkNodes, true, -1);
  if (!trunk.length) throw new UnsupportedGraph(merge ? "The Concat must feed a dense layer (Gemm or MatMul)." : "No weighted layers (Gemm, MatMul or Conv) found on the signal path.");
  const last = trunk[trunk.length - 1];
  if (last.kind !== "dense") throw new UnsupportedGraph(`The last weighted layer is ${last.kind === "conv" ? "a convolution" : "a recurrent cell"}; netglow expects a dense output head.`);
  if (merge && trunk[0].kind !== "dense") throw new UnsupportedGraph("Only a dense layer can come right after the Concat; join flat features into a dense layer.");
  if (merge && trunkNodes.slice(0, trunkNodes.indexOf(trunk[0].node)).some((n) => !RESHAPE.has(n.opType)))
    warnings.push("An elementwise op sits between the Concat and the first dense layer; edges into it use that op's output.");
  for (const b of branches) {
    const l = b.layers.at(-1);
    if (l && l.kind === "dense" && trunk[0].kind === "conv") throw new UnsupportedGraph("A dense layer feeds a convolution (decoder-style); not supported.");
  }
  const cells = [...branches.flatMap((b) => b.layers), ...trunk].filter((l) => l.kind === "rnn");
  if (cells.length !== cellState.size) throw new UnsupportedGraph("A recurrent cell sits off the signal path.");

  const inputs: PlanInput[] = realInputs.map((v) => {
    if (v.shape && v.shape.length > 4) throw new UnsupportedGraph(`Input "${v.name}" has rank ${v.shape.length}; use [N, features] or [N, C, H, W].`);
    if (v.shape?.length === 3) throw new UnsupportedGraph(`Input "${v.name}": sequence inputs [N, T, F] are not supported. Export a step function forward(obs, state) -> (outputs, state) instead.`);
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
  for (const sk of skips) add(sk.src);
  // a recurrent cell needs its input every tick and the state it hands on (for the recurrent edges)
  for (const l of cells) {
    add(l.inTensor);
    add(l.rnn!.hOut);
  }
  for (const st of states) {
    tickDims.set(st.output, [st.size]);
    add(st.output);
  }
  add(graphOutput);
  // saliency needs every tensor on the image path (activations for the masks, shapes for the convs)
  const img = inputs.find((i) => i.isImage);
  let saliency: SaliencyPath | null = null;
  if (img) {
    saliency = states.length
      ? { input: img.name, nodes: [], fetch: [], error: "Saliency is not available for recurrent models (it would need gradients through time)." }
      : saliencyPath(m, img.name, last.actTensor);
    if (!saliency.error) saliency.fetch.forEach(add);
  }
  return { inputs, branches, merge, trunk, layers, outputTensor: last.actTensor, graphOutput, softmaxInGraph, warnings, fetch, skips, states, tickDims, saliency };
}

function weightedLayer(
  n: NodeProto, inTensor: string, actTensor: string, m: ModelInfo, k: number, nLayers: number,
  state?: { hIn: string; hOut: string; cIn: string | null; cOut: string | null },
): Layer {
  const lastLayer = k + 1 === nLayers;
  const need = (name: string): TensorProto => {
    const t = m.initializers.get(name);
    if (!t || !t.data.length) throw new UnsupportedGraph(`Weights "${name}" of node "${n.name}" are not a readable constant.`);
    return t;
  };
  if (RNN.has(n.opType)) {
    const lstm = n.opType === "LSTM";
    const gates = lstm ? 4 : 3;
    const dir = n.attrs.get("direction")?.s ?? "forward";
    if (dir !== "forward") throw new UnsupportedGraph(`${n.opType} "${n.name}" is ${dir}; netglow steps forward cells only.`);
    if ((n.attrs.get("layout")?.i ?? 0) !== 0) throw new UnsupportedGraph(`${n.opType} "${n.name}" uses layout=1; export with the default layout.`);
    if (n.inputs[4]) throw new UnsupportedGraph(`${n.opType} "${n.name}" takes sequence_lens; step it one tick at a time instead.`);
    if (lstm && n.inputs[7]) throw new UnsupportedGraph(`LSTM "${n.name}" has peephole weights, which are not supported.`);
    if (n.attrs.has("activations") || n.attrs.has("clip")) throw new UnsupportedGraph(`${n.opType} "${n.name}" sets custom activations or clip; only the defaults are supported.`);
    const w = need(n.inputs[1]);
    const r = need(n.inputs[2]);
    const [dirs, gH, I] = w.dims;
    const H = gH / gates;
    if (dirs !== 1) throw new UnsupportedGraph(`${n.opType} "${n.name}" is bidirectional; not supported.`);
    const B = n.inputs[3] ? need(n.inputs[3]).data : new Float32Array(2 * gates * H);
    const cand = lstm ? 3 : 2;
    const Wx = w.data;
    const R = r.data;
    const cell: RnnCell = {
      type: lstm ? "lstm" : "gru", H, I, gates, Wx, R, B, cand,
      linearBeforeReset: (n.attrs.get("linear_before_reset")?.i ?? 0) === 1,
      Rc: R.slice(cand * H * H, (cand + 1) * H * H),
      hIn: state?.hIn ?? "", hOut: state?.hOut ?? "", cIn: state?.cIn ?? null, cOut: state?.cOut ?? null,
    };
    // edges into the cell: contributions to the candidate gate's pre-activation, W_c[j, i] * x[i]
    return { kind: "rnn", node: n, inTensor, actTensor, W: Wx.slice(cand * H * I, (cand + 1) * H * I), out: H, in: I, rnn: cell, label: `${lstm ? "lstm" : "gru"} ${k + 1}` };
  }
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
