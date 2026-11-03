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
//   column      - one vertical row of nodes: the input, each hidden layer, the output
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
function contributions(layer: Layer, src: { dims: number[]; size: number; data: Float32Array }, T: number, targets: number[], cache: Map<Layer, Float32Array>): { c: Float32Array; G: number } {
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
  if (Sz !== layer.in) throw new Error(`Dense layer "${layer.node.name}" expects ${layer.in} inputs, its source has ${Sz}.`);
  const G = src.dims.length === 3 ? src.dims[0] : Sz;
  const gs = Sz / G;
  const out = new Float32Array(T * nt * G);
  for (let t = 0; t < T; t++) {
    const xo = t * Sz;
    for (let jj = 0; jj < nt; jj++) {
      const wo = targets[jj] * Sz;
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

export function buildViz(plan: Plan, trace: Trace, rec: Map<string, NdArray>, obs: NdArray, opt: VizOptions, prevOrder?: { input?: number[]; output?: number[] }): Viz {
  const T = trace.T;
  const L = plan.layers;
  const get = (name: string) => {
    const e = trace.tensors.get(name);
    if (!e) throw new Error(`missing tensor ${name}`);
    return e;
  };
  // sources per layer k: column k values. column 0 = the tensor that feeds layer 0.
  const obsF = toFloat(obs);
  const obsDims = obs.shape.slice(1);
  const in0 = trace.tensors.get(L[0].inTensor) ?? { dims: obsDims, size: obsDims.reduce((a, b) => a * b, 1), data: obsF };
  const srcOf = (k: number) => (k === 0 ? in0 : get(L[k - 1].actTensor));
  const cache = new Map<Layer, Float32Array>();

  // output column
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

  // backwards selection
  const sels: number[][] = new Array(L.length + 1);
  sels[L.length] = outSel;
  const edgeRaw: { c: Float32Array; G: number; from: number }[] = new Array(L.length);
  const meanAbs: Float64Array[] = new Array(L.length);
  for (let k = L.length - 1; k >= 0; k--) {
    const targets = sels[k + 1];
    const { c, G } = contributions(L[k], srcOf(k), T, targets, cache);
    const score = new Float64Array(G);
    for (let t = 0; t < T; t++) for (let jj = 0; jj < targets.length; jj++) for (let g = 0; g < G; g++) score[g] += Math.abs(c[(t * targets.length + jj) * G + g]);
    const cap = k === 0 ? (G <= opt.maxInputs ? G : opt.nodesPerColumn) : opt.nodesPerColumn;
    let sel = topIdx(score, cap);
    if (k === 0) sel = sel.sort((a, b) => a - b);
    sels[k] = sel;
    // restrict to selected sources
    const nt = targets.length;
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
    edgeRaw[k] = { c: r, G, from: k };
    meanAbs[k] = ma;
  }

  // columns
  const cols: VizColumn[] = [];
  // input column: brightness = how fast each input changes, scaled per input over the recording
  {
    const sel = sels[0];
    const ns = sel.length;
    const isImg = obsDims.length === 3;
    const C = isImg ? obsDims[0] : obsDims.reduce((a, b) => a * b, 1);
    const per = isImg ? obsDims[1] * obsDims[2] : 1;
    const size = C * per;
    const raw = new Float32Array(T * ns);
    const chg = new Float32Array(T * ns);
    for (let t = 0; t < T; t++)
      for (let ii = 0; ii < ns; ii++) {
        const g = sel[ii];
        let m = 0;
        let d = 0;
        for (let p = 0; p < per; p++) {
          const x = obsF[t * size + g * per + p];
          m += x;
          if (t > 0) d += Math.abs(x - obsF[(t - 1) * size + g * per + p]);
        }
        raw[t * ns + ii] = m / per;
        chg[t * ns + ii] = d / per;
      }
    const v = new Float32Array(T * ns);
    for (let ii = 0; ii < ns; ii++) {
      const col = new Float32Array(T);
      for (let t = 0; t < T; t++) col[t] = chg[t * ns + ii];
      const sc = Math.max(percentile(col, 95), 1e-9);
      for (let t = 0; t < T; t++) v[t * ns + ii] = Math.min(1, chg[t * ns + ii] / sc);
    }
    const order = prevOrder?.input && prevOrder.input.length === ns ? prevOrder.input : [...Array(ns).keys()];
    cols.push({ kind: "input", label: "input", n: isImg ? C : C, sel, order, v, raw });
  }
  for (let k = 0; k + 1 < L.length; k++) {
    const sel = sels[k + 1];
    const u = unitValues(get(L[k].actTensor), T);
    const ns = sel.length;
    const vals = new Float32Array(T * ns);
    for (let t = 0; t < T; t++) for (let ii = 0; ii < ns; ii++) vals[t * ns + ii] = u.vals[t * u.n + sel[ii]];
    const lo = percentile(vals, 5);
    const hi = percentile(vals, 95);
    const v = vals.map((x) => Math.min(1, Math.max(0, (x - lo) / Math.max(hi - lo, 1e-9))));
    cols.push({ kind: "hidden", label: `${L[k].label} · ${u.n}`, n: u.n, sel, order: [...Array(ns).keys()], v });
  }
  {
    const ns = outSel.length;
    const v = new Float32Array(T * ns);
    for (let t = 0; t < T; t++) for (let ii = 0; ii < ns; ii++) v[t * ns + ii] = prob[t * nOut + outSel[ii]];
    const order = prevOrder?.output && prevOrder.output.length === ns ? prevOrder.output : [...Array(ns).keys()];
    cols.push({ kind: "output", label: "output", n: nOut, sel: outSel, order, v, raw: v });
  }

  const edges: VizEdges[] = edgeRaw.map((e, k) => {
    const abs = e.c.map(Math.abs);
    return { from: k, to: k + 1, c: e.c, scale: Math.max(percentile(abs, 95), 1e-9) };
  });

  const viz: Viz = {
    T, cols, edges, chosen, mode,
    match: matchCheck(plan, trace, rec, chosen, mode),
    imageInput: obsDims.length === 3 ? { C: obsDims[0], H: obsDims[1], W: obsDims[2] } : null,
  };
  (viz as VizInternal)._meanAbs = meanAbs;
  orderHidden(viz);
  return viz;
}

interface VizInternal extends Viz {
  _meanAbs: Float64Array[];
}

/** Barycentric ordering of hidden columns, right to left: each unit sits near the targets it drives most. */
export function orderHidden(viz: Viz) {
  const meanAbs = (viz as VizInternal)._meanAbs;
  for (let ci = viz.cols.length - 2; ci >= 1; ci--) {
    const col = viz.cols[ci];
    const next = viz.cols[ci + 1];
    const ma = meanAbs[ci]; // edges ci -> ci+1: (next.sel x col.sel)
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
