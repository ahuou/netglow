// From a trace (real activations at every tick) to what the picture needs: which units to show per
// column, their brightness per tick, and the strongest edges per tick.
//
// Vocabulary:
//   contribution c[t, j, i] - how much source unit i adds to target unit j's pre-activation at tick t
//                             dense:  W[j,i] * x_t[i]                                   (exact)
//                             dense fed by a conv map: sum over pixels of W[j,c,p] * x_t[c,p]   (exact, grouped by channel)
//                             conv:   contribution of input channel c to the spatial MEAN of output channel d's
//                                     pre-activation = sum_k W[d,c,k] * mean of x_t[c] over the positions kernel tap k reads
//                                     (exact for the channel mean; bias excluded)
//                             dense after a Concat: the same, on the slice of the weight columns that belongs to each branch
//                             (W[j, off_b + i] * x_t[off_b + i], so every branch keeps its exact share)
//   column      - one vertical row of nodes: an input, each hidden layer, the output
//   lane        - with several model inputs, each branch (input + its own layers) gets a horizontal band of the
//                 canvas; columns after the Concat span the full height (lane -1)
//   stage       - a column's horizontal position, 0 = inputs; branch columns line up stage by stage
//   selection   - the units shown: output-driving units first, then backwards layer by layer
//   robust norm - map values to 0..1 using the 5th and 95th percentile over the whole recording, so one spike
//                 does not wash out the rest and flicker reflects real change

import type { Layer, Plan } from "./analyze";
import type { Trace } from "./run";
import { toFloat, type NdArray } from "./io/npy";

export type OutputMode = "categorical" | "continuous";

export interface VizColumn {
  kind: "input" | "hidden" | "output";
  label: string;
  /** total units in the layer (before selection) */
  n: number;
  /** selected unit indices (index into the layer) */
  sel: number[];
  /** display order: order[slot] = index into sel */
  order: number[];
  /** brightness 0..1, T x sel.length */
  v: Float32Array;
  /** input column: raw values per tick (T x sel.length) for the live readout; output: probability or value */
  raw?: Float32Array;
  /** conv columns: real activation maps of the selected channels, per tick, each channel scaled 0..255
   *  over the whole recording (1st..99th percentile). Layout: T x sel.length x H x W */
  maps?: { H: number; W: number; data: Uint8Array };
  /** vertical band: branch index, or -1 = full height (columns after the merge, and every column of a chain model) */
  lane: number;
  /** horizontal position, 0 = inputs */
  stage: number;
  /** input columns: which model input (index into plan.inputs), where its names start in labels.inputs, image or not */
  input?: number;
  base?: number;
  image?: boolean;
}

export interface VizEdges {
  from: number;
  to: number;
  /** T x to.sel.length x from.sel.length */
  c: Float32Array;
  scale: number;
}

export interface MatchCheck {
  argmax: { agree: number; total: number } | null;
  maxDiff: number | null;
  /** per tick: 1 when the model's choice disagrees with the logged action */
  mismatch: Uint8Array | null;
  against: string;
}

export interface Viz {
  T: number;
  cols: VizColumn[];
  edges: VizEdges[];
  chosen: Int32Array | null;
  mode: OutputMode;
  match: MatchCheck;
  imageInput: { C: number; H: number; W: number } | null;
  /** number of lanes (1 for chain models) and stages */
  lanes: number;
  stages: number;
}

export interface VizOptions {
  nodesPerColumn: number;
  maxInputs: number;
  mode: OutputMode;
}

export function percentile(a: ArrayLike<number>, q: number): number {
  const n = a.length;
  if (!n) return 0;
  const stride = Math.max(1, Math.floor(n / 200000));
  const s: number[] = [];
  for (let i = 0; i < n; i += stride) s.push(a[i]);
  s.sort((x, y) => x - y);
  const p = (q / 100) * (s.length - 1);
  const lo = Math.floor(p);
  const hi = Math.min(s.length - 1, lo + 1);
  return s[lo] + (s[hi] - s[lo]) * (p - lo);
}

/** a per-tick tensor, or a slice of one: element i of tick t is data[t * stride + off + i] (stride defaults to size).
 *  A slice is how one branch's share of a Concat output is read; dims give its shape for grouping by channel. */
interface TView {
  dims: number[];
  size: number;
  data: Float32Array;
  stride?: number;
  off?: number;
}

/** per-tick unit values of a tensor: dense = |a|, conv map = |spatial mean| per channel */
function unitValues(d: { dims: number[]; size: number; data: Float32Array }, T: number): { n: number; vals: Float32Array; signed: Float32Array } {
  if (d.dims.length === 3) {
    const [C, H, W] = d.dims;
    const hw = H * W;
    const out = new Float32Array(T * C);
    for (let t = 0; t < T; t++)
      for (let c = 0; c < C; c++) {
        let s = 0;
        const o = t * d.size + c * hw;
        for (let p = 0; p < hw; p++) s += d.data[o + p];
        out[t * C + c] = s / hw;
      }
    return { n: C, vals: out.map(Math.abs), signed: out };
  }
  return { n: d.size, vals: d.data.map(Math.abs), signed: d.data };
}

/** conv tap means: S[t, c, k] = mean over output positions of x_t[c] at the pixel kernel tap k reads (0 if padding) */
function tapMeans(layer: Layer, src: { dims: number[]; size: number; data: Float32Array }, T: number): Float32Array {
  const g = layer.conv!;
  const [C, H, W] = src.dims;
  const OH = Math.floor((H + 2 * g.ph - g.dh * (g.kh - 1) - 1) / g.sh) + 1;
  const OW = Math.floor((W + 2 * g.pw - g.dw * (g.kw - 1) - 1) / g.sw) + 1;
  const K = g.kh * g.kw;
  const S = new Float32Array(T * C * K);
  const inv = 1 / (OH * OW);
  for (let t = 0; t < T; t++)
    for (let c = 0; c < C; c++) {
      const base = t * src.size + c * H * W;
      for (let ky = 0; ky < g.kh; ky++)
        for (let kx = 0; kx < g.kw; kx++) {
          let s = 0;
          for (let oy = 0; oy < OH; oy++) {
            const y = oy * g.sh - g.ph + ky * g.dh;
            if (y < 0 || y >= H) continue;
            const row = base + y * W;
            for (let ox = 0; ox < OW; ox++) {
              const x = ox * g.sw - g.pw + kx * g.dw;
              if (x >= 0 && x < W) s += src.data[row + x];
            }
          }
          S[(t * C + c) * K + ky * g.kw + kx] = s * inv;
        }
    }
  return S;
}

/** contributions of every source group to the given targets: T x targets.length x G */
function contributions(layer: Layer, src: TView, T: number, targets: number[], cache: Map<Layer, Float32Array>): { c: Float32Array; G: number } {
  const nt = targets.length;
  if (layer.kind === "conv") {
    const C = src.dims[0];
    if (src.dims.length !== 3 || C !== layer.in) throw new Error(`Conv "${layer.node.name}" expects ${layer.in} input channels, got [${src.dims}].`);
    let S = cache.get(layer);
    if (!S) cache.set(layer, (S = tapMeans(layer, src, T)));
    const K = layer.conv!.kh * layer.conv!.kw;
    const out = new Float32Array(T * nt * C);
    for (let t = 0; t < T; t++)
      for (let jj = 0; jj < nt; jj++) {
        const d = targets[jj];
        for (let c = 0; c < C; c++) {
          let s = 0;
          const wo = (d * C + c) * K;
          const so = (t * C + c) * K;
          for (let k = 0; k < K; k++) s += layer.W[wo + k] * S[so + k];
          out[(t * nt + jj) * C + c] = s;
        }
      }
    return { c: out, G: C };
  }
  const Sz = src.size;
  const off = src.off ?? 0;
  const stride = src.stride ?? Sz;
  if (off + Sz > layer.in || (stride === Sz && Sz !== layer.in))
    throw new Error(`Dense layer "${layer.node.name}" expects ${layer.in} inputs, its source has ${Sz}.`);
  const G = src.dims.length === 3 ? src.dims[0] : Sz;
  const gs = Sz / G;
  const out = new Float32Array(T * nt * G);
  for (let t = 0; t < T; t++) {
    const xo = t * stride + off;
    for (let jj = 0; jj < nt; jj++) {
      // the weight columns of this source: a branch's slice of the Concat, or all of them
      const wo = targets[jj] * layer.in + off;
      const oo = (t * nt + jj) * G;
      if (gs === 1) for (let i = 0; i < Sz; i++) out[oo + i] = layer.W[wo + i] * src.data[xo + i];
      else
        for (let g = 0; g < G; g++) {
          let s = 0;
          for (let p = g * gs; p < (g + 1) * gs; p++) s += layer.W[wo + p] * src.data[xo + p];
          out[oo + g] = s;
        }
    }
  }
  return { c: out, G };
}

function topIdx(score: Float64Array, k: number): number[] {
  return [...score.keys()].sort((a, b) => score[b] - score[a]).slice(0, Math.min(k, score.length));
}

function softmaxRows(x: Float32Array, T: number, n: number): Float32Array {
  const p = new Float32Array(T * n);
  for (let t = 0; t < T; t++) {
    let m = -Infinity;
    for (let i = 0; i < n; i++) m = Math.max(m, x[t * n + i]);
    let s = 0;
    for (let i = 0; i < n; i++) s += p[t * n + i] = Math.exp(x[t * n + i] - m);
    for (let i = 0; i < n; i++) p[t * n + i] /= s;
  }
  return p;
}

/** obs: one recording array per model input, in plan.inputs order */
export function buildViz(plan: Plan, trace: Trace, rec: Map<string, NdArray>, obs: NdArray[], opt: VizOptions, prevOrder?: { inputs?: number[][]; output?: number[] }): Viz {
  const T = trace.T;
  const get = (name: string) => {
    const e = trace.tensors.get(name);
    if (!e) throw new Error(`missing tensor ${name}`);
    return e;
  };
  const obsF = obs.map(toFloat);
  const inView = (i: number): TView => {
    const dims = obs[i].shape.slice(1);
    return { dims, size: dims.reduce((a, b) => a * b, 1), data: obsF[i] };
  };
  const cache = new Map<Layer, Float32Array>();
  const multi = !!plan.merge;

  // ---- columns and what feeds them: branches (input + own layers), then the trunk after the merge
  type Src = { ci: number; view: TView };
  const cols: VizColumn[] = [];
  const feeds: { ci: number; layer: Layer; srcs: Src[] }[] = [];
  const blank = (kind: VizColumn["kind"], label: string, lane: number, stage: number): VizColumn => ({ kind, label, lane, stage, n: 0, sel: [], order: [], v: new Float32Array(0) });
  const nbMax = Math.max(...plan.branches.map((b) => b.layers.length));
  const branchEnd: { ci: number; view: TView }[] = [];
  // where each input's names start in labels.inputs: model input order, an image input takes one name
  const baseOf: number[] = [];
  plan.inputs.reduce((acc, inp, i) => ((baseOf[i] = acc), acc + (inp.isImage ? 1 : inView(i).size)), 0);
  plan.branches.forEach((b, bi) => {
    const lane = multi ? bi : -1;
    const inp = plan.inputs[b.input];
    const col = blank("input", "input", lane, 0);
    col.input = b.input;
    col.base = baseOf[b.input];
    col.image = inp.isImage;
    cols.push(col);
    let prev = cols.length - 1;
    b.layers.forEach((l, li) => {
      const view = li === 0 ? (trace.tensors.get(l.inTensor) ?? inView(b.input)) : get(b.layers[li - 1].actTensor);
      cols.push(blank("hidden", l.label, lane, li + 1));
      feeds.push({ ci: cols.length - 1, layer: l, srcs: [{ ci: prev, view }] });
      prev = cols.length - 1;
    });
    branchEnd.push({ ci: prev, view: b.layers.length ? get(b.layers[b.layers.length - 1].actTensor) : inView(b.input) });
  });
  plan.trunk.forEach((l, li) => {
    const isOut = li === plan.trunk.length - 1;
    cols.push(blank(isOut ? "output" : "hidden", isOut ? "output" : l.label, -1, nbMax + 1 + li));
    const ci = cols.length - 1;
    if (li > 0) feeds.push({ ci, layer: l, srcs: [{ ci: ci - 1, view: get(plan.trunk[li - 1].actTensor) }] });
    else if (!multi) feeds.push({ ci, layer: l, srcs: [{ ci: branchEnd[0].ci, view: trace.tensors.get(l.inTensor) ?? branchEnd[0].view }] });
    else {
      // the dense layer after the Concat: read each branch's slice of its real input, in Concat order
      const x = get(l.inTensor);
      if (x.size !== l.in) throw new Error(`Dense layer "${l.node.name}" expects ${l.in} inputs, the Concat gives ${x.size}.`);
      let off = 0;
      const srcs: Src[] = plan.branches.map((b, bi) => {
        const e = branchEnd[bi];
        const sz = b.concatIn && trace.tensors.has(b.concatIn) ? get(b.concatIn).size : e.view.size;
        if (sz !== e.view.size) throw new Error(`Branch of input "${plan.inputs[b.input].name}" hands ${sz} values to the Concat, but its last layer has ${e.view.size}.`);
        const src = { ci: e.ci, view: { dims: e.view.dims, size: sz, data: x.data, stride: x.size, off } };
        off += sz;
        return src;
      });
      if (off !== l.in) throw new Error(`The Concat branches add up to ${off} values; "${l.node.name}" expects ${l.in}.`);
      feeds.push({ ci, layer: l, srcs });
    }
  });
  const outCi = cols.length - 1;

  // ---- output column values
  const logits = get(plan.outputTensor);
  const nOut = logits.size;
  const mode = opt.mode;
  let prob: Float32Array;
  let chosen: Int32Array | null = null;
  if (mode === "categorical") {
    prob = softmaxRows(logits.data, T, nOut);
    chosen = new Int32Array(T);
    for (let t = 0; t < T; t++) {
      let b = 0;
      for (let i = 1; i < nOut; i++) if (logits.data[t * nOut + i] > logits.data[t * nOut + b]) b = i;
      chosen[t] = b;
    }
  } else {
    const go = get(plan.graphOutput);
    prob = new Float32Array(T * nOut);
    for (let i = 0; i < nOut; i++) {
      const col = new Float32Array(T);
      for (let t = 0; t < T; t++) col[t] = Math.abs(go.data[t * nOut + i]);
      const sc = Math.max(percentile(col, 95), 1e-9);
      for (let t = 0; t < T; t++) prob[t * nOut + i] = Math.min(1, col[t] / sc);
    }
  }
  const outMean = new Float64Array(nOut);
  for (let t = 0; t < T; t++) for (let i = 0; i < nOut; i++) outMean[i] += prob[t * nOut + i];
  const outSel = nOut <= 16 ? [...Array(nOut).keys()] : topIdx(outMean, 16).sort((a, b) => a - b);

  // ---- backwards selection: consumers come after their sources in `cols`, so walk the feeds in reverse
  const sels: number[][] = cols.map(() => []);
  sels[outCi] = outSel;
  const edgeRaw: { from: number; to: number; c: Float32Array; meanAbs: Float64Array }[] = [];
  for (let f = feeds.length - 1; f >= 0; f--) {
    const { ci, layer, srcs } = feeds[f];
    const targets = sels[ci];
    const nt = targets.length;
    for (const src of srcs) {
      const sc = cols[src.ci];
      const { c, G } = contributions(layer, src.view, T, targets, cache);
      const score = new Float64Array(G);
      for (let t = 0; t < T; t++) for (let jj = 0; jj < nt; jj++) for (let g = 0; g < G; g++) score[g] += Math.abs(c[(t * nt + jj) * G + g]);
      const isImageSrc = sc.kind === "input" && !!sc.image;
      // an image input is one node: every channel of the stack is kept, then summed into one edge per target
      const cap = isImageSrc ? G : sc.kind === "input" ? (G <= opt.maxInputs ? G : opt.nodesPerColumn) : opt.nodesPerColumn;
      let sel = topIdx(score, cap);
      if (sc.kind === "input") sel = sel.sort((a, b) => a - b);
      const ns = sel.length;
      const r = new Float32Array(T * nt * ns);
      const ma = new Float64Array(nt * ns);
      for (let t = 0; t < T; t++)
        for (let jj = 0; jj < nt; jj++)
          for (let ii = 0; ii < ns; ii++) {
            const v = c[(t * nt + jj) * G + sel[ii]];
            r[(t * nt + jj) * ns + ii] = v;
            ma[jj * ns + ii] += Math.abs(v) / T;
          }
      if (isImageSrc) {
        const r1 = new Float32Array(T * nt);
        const m1 = new Float64Array(nt);
        for (let q = 0; q < T * nt; q++) {
          let sum = 0;
          for (let ii = 0; ii < ns; ii++) sum += r[q * ns + ii];
          r1[q] = sum;
          m1[q % nt] += Math.abs(sum) / T;
        }
        sels[src.ci] = [0];
        edgeRaw.push({ from: src.ci, to: ci, c: r1, meanAbs: m1 });
        continue;
      }
      sels[src.ci] = sel;
      edgeRaw.push({ from: src.ci, to: ci, c: r, meanAbs: ma });
    }
  }

  // ---- column values
  let inputNo = 0;
  let imageInput: Viz["imageInput"] = null;
  for (let ci = 0; ci < cols.length; ci++) {
    const col = cols[ci];
    const sel = sels[ci];
    const ns = sel.length;
    col.sel = sel;
    if (col.kind === "input") {
      const x = obsF[col.input!];
      const dims = obs[col.input!].shape.slice(1);
      const size = dims.reduce((a, b) => a * b, 1);
      if (col.image) {
        // image node: brightness = how much the whole input image changed since the last tick
        imageInput = { C: dims[0], H: dims[1], W: dims[2] };
        const chg = new Float32Array(T);
        for (let t = 1; t < T; t++) {
          let d = 0;
          for (let p = 0; p < size; p++) d += Math.abs(x[t * size + p] - x[(t - 1) * size + p]);
          chg[t] = d / size;
        }
        const sc = Math.max(percentile(chg, 95), 1e-9);
        Object.assign(col, { n: 1, sel: [0], order: [0], v: chg.map((q) => Math.min(1, q / sc)) });
      } else {
        // brightness = how fast each input changes, scaled per input over the recording
        const raw = new Float32Array(T * ns);
        const chg = new Float32Array(T * ns);
        for (let t = 0; t < T; t++)
          for (let ii = 0; ii < ns; ii++) {
            const g = sel[ii];
            raw[t * ns + ii] = x[t * size + g];
            chg[t * ns + ii] = t > 0 ? Math.abs(x[t * size + g] - x[(t - 1) * size + g]) : 0;
          }
        const v = new Float32Array(T * ns);
        for (let ii = 0; ii < ns; ii++) {
          const cc = new Float32Array(T);
          for (let t = 0; t < T; t++) cc[t] = chg[t * ns + ii];
          const sc = Math.max(percentile(cc, 95), 1e-9);
          for (let t = 0; t < T; t++) v[t * ns + ii] = Math.min(1, chg[t * ns + ii] / sc);
        }
        const prev = prevOrder?.inputs?.[inputNo];
        Object.assign(col, { n: size, order: prev && prev.length === ns ? prev : [...Array(ns).keys()], v, raw });
      }
      inputNo++;
    } else if (col.kind === "hidden") {
      const l = feeds.find((f) => f.ci === ci)!.layer;
      const act = get(l.actTensor);
      const u = unitValues(act, T);
      const vals = new Float32Array(T * ns);
      for (let t = 0; t < T; t++) for (let ii = 0; ii < ns; ii++) vals[t * ns + ii] = u.vals[t * u.n + sel[ii]];
      const lo = percentile(vals, 5);
      const hi = percentile(vals, 95);
      const v = vals.map((x) => Math.min(1, Math.max(0, (x - lo) / Math.max(hi - lo, 1e-9))));
      Object.assign(col, { label: `${l.label} · ${u.n}`, n: u.n, order: [...Array(ns).keys()], v, maps: act.dims.length === 3 ? featureMaps(act, sel, T) : undefined });
    } else {
      const v = new Float32Array(T * ns);
      for (let t = 0; t < T; t++) for (let ii = 0; ii < ns; ii++) v[t * ns + ii] = prob[t * nOut + outSel[ii]];
      const order = prevOrder?.output && prevOrder.output.length === ns ? prevOrder.output : [...Array(ns).keys()];
      Object.assign(col, { n: nOut, order, v, raw: v });
    }
  }

  // edges into the same column share one scale, so a Concat's branches are drawn on a common footing
  const scaleTo = new Map<number, number>();
  for (const to of new Set(edgeRaw.map((e) => e.to))) {
    const all = edgeRaw.filter((e) => e.to === to).flatMap((e) => Array.from(e.c, Math.abs));
    scaleTo.set(to, Math.max(percentile(all, 95), 1e-9));
  }
  edgeRaw.sort((a, b) => a.to - b.to || a.from - b.from);
  const edges: VizEdges[] = edgeRaw.map((e) => ({ from: e.from, to: e.to, c: e.c, scale: scaleTo.get(e.to)! }));

  const viz: Viz = {
    T, cols, edges, chosen, mode,
    match: matchCheck(plan, trace, rec, chosen, mode),
    imageInput,
    lanes: multi ? plan.branches.length : 1,
    stages: nbMax + 1 + plan.trunk.length,
  };
  (viz as VizInternal)._meanAbs = edgeRaw.map((e) => e.meanAbs);
  orderHidden(viz);
  return viz;
}

/** selected channels' activation maps as bytes, each channel scaled by its own 1st..99th percentile */
function featureMaps(act: { dims: number[]; size: number; data: Float32Array }, sel: number[], T: number) {
  const [, H, W] = act.dims;
  const hw = H * W;
  const ns = sel.length;
  const out = new Uint8Array(T * ns * hw);
  for (let ii = 0; ii < ns; ii++) {
    const c = sel[ii];
    const vals = new Float32Array(T * hw);
    for (let t = 0; t < T; t++) vals.set(act.data.subarray(t * act.size + c * hw, t * act.size + (c + 1) * hw), t * hw);
    const lo = percentile(vals, 1);
    const hi = Math.max(percentile(vals, 99), lo + 1e-9);
    for (let t = 0; t < T; t++)
      for (let p = 0; p < hw; p++) out[(t * ns + ii) * hw + p] = Math.max(0, Math.min(255, ((vals[t * hw + p] - lo) / (hi - lo)) * 255));
  }
  return { H, W, data: out };
}

interface VizInternal extends Viz {
  _meanAbs: Float64Array[];
}

/** Barycentric ordering of hidden columns, right to left: each unit sits near the targets it drives most. */
export function orderHidden(viz: Viz) {
  const meanAbs = (viz as VizInternal)._meanAbs; // aligned with viz.edges
  for (let ci = viz.cols.length - 2; ci >= 0; ci--) {
    const col = viz.cols[ci];
    if (col.kind !== "hidden") continue;
    const ei = viz.edges.findIndex((e) => e.from === ci);
    if (ei < 0) continue;
    const next = viz.cols[viz.edges[ei].to];
    const ma = meanAbs[ei]; // edges ci -> next: (next.sel x col.sel)
    const slotOf = new Float64Array(next.sel.length);
    next.order.forEach((ii, slot) => (slotOf[ii] = slot / Math.max(1, next.order.length - 1)));
    const ns = col.sel.length;
    const bc = new Float64Array(ns);
    for (let ii = 0; ii < ns; ii++) {
      let w = 0;
      let s = 0;
      for (let jj = 0; jj < next.sel.length; jj++) {
        const m = ma[jj * ns + ii];
        w += m;
        s += m * slotOf[jj];
      }
      bc[ii] = w ? s / w : 0.5;
    }
    col.order = [...Array(ns).keys()].sort((a, b) => bc[a] - bc[b]);
  }
}

function matchCheck(plan: Plan, trace: Trace, rec: Map<string, NdArray>, chosen: Int32Array | null, mode: OutputMode): MatchCheck {
  const T = trace.T;
  const res: MatchCheck = { argmax: null, maxDiff: null, mismatch: null, against: "" };
  const go = trace.tensors.get(plan.graphOutput)!;
  const logged = rec.get("outputs") ?? rec.get("logits");
  if (logged && logged.shape[0] === T) {
    const l = toFloat(logged);
    if (l.length === go.data.length) {
      let m = 0;
      for (let i = 0; i < l.length; i++) m = Math.max(m, Math.abs(l[i] - go.data[i]));
      res.maxDiff = m;
      res.against = "logged outputs";
    }
  }
  const acts = rec.get("actions");
  if (acts && acts.shape[0] === T) {
    const a = toFloat(acts);
    if (mode === "categorical" && chosen && a.length === T) {
      const mis = new Uint8Array(T);
      let agree = 0;
      for (let t = 0; t < T; t++) {
        if (Math.round(a[t]) === chosen[t]) agree++;
        else mis[t] = 1;
      }
      res.argmax = { agree, total: T };
      res.mismatch = mis;
      res.against = res.against ? `${res.against} and actions` : "logged actions";
    } else if (mode === "continuous" && a.length === go.data.length && res.maxDiff === null) {
      let m = 0;
      for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - go.data[i]));
      res.maxDiff = m;
      res.against = "logged actions";
    }
  }
  return res;
}
