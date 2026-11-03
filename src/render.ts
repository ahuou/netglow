// Draws one frame of the glowing network onto a 2D canvas. Pure function of (viz, labels, options, time),
// plus a small easing state so 50 Hz decisions do not strobe at 30 fps.
//
// Reading left to right:
//   input   - one row per input: label, live value, node colour = how fast that input is changing
//   hidden  - one column per layer, brightness = activation (robust-normalised per column over the recording)
//   output  - one node per output with its icon; glow = probability, the chosen (argmax) one is ringed and lit
//   edges   - per target node, the top-k sources by |contribution| this tick; warm = excites, cool = inhibits
//   pulse   - a faint band sweeping left to right once per decision tick
//   skip    - a residual connection: an arc from the block's input column over the block into the column after it
//   loop    - a recurrent cell's h(t-1) -> h(t) edges, dashed loops on the right of its column
// Image models: the input column is one image node (the current frame, pixels kept visible, border lit by how
// much the frame changed), optionally a large "What the AI sees" view above or beside the network, and conv
// channels can be drawn as their live feature maps instead of dots.

import type { Viz, VizColumn } from "./compute";
import type { NdArray } from "./io/npy";
import { frameRGBA, type ImageView } from "./image";
import type { SaliencyMap } from "./saliency";

/** where a picture goes: a band above the network, a column beside it, a corner inset, or nowhere.
 *  "auto" (recording only) = above when the AI view is above, else beside left */
export type PicPos = "above" | "left" | "right" | "tl" | "tr" | "bl" | "br" | "off" | "auto";

/** every size and look knob; all lengths are in pixels at 720 p and scale with the export height */
export interface Style {
  nodeScale: number;
  edgeMin: number;
  edgeMax: number;
  labelSize: number;
  valueSize: number;
  mapScale: number;
  imageScale: number;
  aiPos: PicPos;
  aiScale: number;
  recPos: PicPos;
  recScale: number;
  colSpacing: number;
  rowSpacing: number;
  padding: number;
  titles: boolean;
  glow: number;
  pulse: boolean;
  pulseIntensity: number;
  /** colour overrides; "" = use the theme */
  pos: string;
  neg: string;
  bg: string;
  shape: "circle" | "square";
  showValues: boolean;
  showProbs: boolean;
  /** gradient × input heatmap over the AI view and the image node */
  saliency: boolean;
  saliencyOpacity: number;
}

export const DEFAULT_STYLE: Style = {
  nodeScale: 1, edgeMin: 1, edgeMax: 2.6, labelSize: 17, valueSize: 15, mapScale: 1, imageScale: 1,
  aiPos: "above", aiScale: 1, recPos: "auto", recScale: 1, colSpacing: 1, rowSpacing: 1, padding: 28, titles: true,
  glow: 1, pulse: true, pulseIntensity: 1, pos: "", neg: "", bg: "", shape: "circle", showValues: true, showProbs: true,
  saliency: true, saliencyOpacity: 0.8,
};

export interface Theme {
  id: string;
  name: string;
  bg: string;
  off: string;
  on: string;
  pos: string;
  neg: string;
  text: string;
  dim: string;
  pulse: string;
}

export const THEMES: Theme[] = [
  { id: "ember", name: "Ember", bg: "#0e0f12", off: "#2a2c33", on: "#ffc400", pos: "#ffc400", neg: "#5aa0ff", text: "#f0f0f0", dim: "#9698a0", pulse: "#ffffff" },
  { id: "ice", name: "Ice", bg: "#0b1015", off: "#25303a", on: "#8ce8ff", pos: "#8ce8ff", neg: "#ff6f8e", text: "#eef6fa", dim: "#8fa1ad", pulse: "#ffffff" },
  { id: "phosphor", name: "Phosphor", bg: "#0a0f0b", off: "#223026", on: "#6dff9a", pos: "#6dff9a", neg: "#ff9a52", text: "#e8f5ea", dim: "#8ea595", pulse: "#ffffff" },
  { id: "paper", name: "Paper", bg: "#f4f2ec", off: "#d7d3c8", on: "#d9480f", pos: "#d9480f", neg: "#1c6dd0", text: "#16161a", dim: "#5e5c57", pulse: "#000000" },
];

export interface Labels {
  title: string;
  inputs: string[]; // by unit index
  outputs: string[];
  icons: (HTMLImageElement | null)[];
  units: (string | null)[];
}

export interface RenderOpts {
  width: number;
  height: number;
  theme: Theme;
  transparent: boolean;
  topK: number;
  showFrame: boolean;
  /** largest fit shrink allowed (compare mode passes the other panel's, so both draw at one scale) */
  fitCap?: number;
  frames: NdArray | null;
  obs: NdArray;
  /** decision rate of the recording in Hz */
  hz: number;
  /** playback speed multiplier (video seconds -> recording seconds) */
  speed: number;
  /** output fps, so the pulse never strobes faster than the frames can show */
  fps: number;
  /** image models: how to display the input tensor (null for vector inputs) */
  image: ImageView | null;
  showMaps: boolean;
  style: Style;
  /** image models: the saliency heatmap, when it is switched on and available */
  saliency?: SaliencyMap | null;
}

interface Pic {
  kind: "rec" | "ai";
  x: number;
  y: number;
  w: number;
  h: number;
  caption: string;
  /** corner positions only choose the vertical alignment of a picture beside the network; nothing is drawn on top */
  corner: boolean;
  /** caption font size */
  capPx: number;
}

type RGB = [number, number, number];
const hex = (h: string): RGB => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
const mix = (a: RGB, b: RGB, t: number) => `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)},${Math.round(a[1] + (b[1] - a[1]) * t)},${Math.round(a[2] + (b[2] - a[2]) * t)})`;
const rgba = (c: RGB, a: number) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

/** a placed column title: drawn left-aligned at (x0, y), and its bounding box */
export interface TitleBox {
  text: string;
  x0: number;
  x1: number;
  y: number;
  y0: number;
  y1: number;
}

/**
 * Place labels in one row without overlap, each as close as possible to its wanted centre.
 * Least-squares with ordering constraints: substitute y_i = x_i - (widths and gaps before i), the constraint
 * "label i+1 starts after label i ends" becomes y nondecreasing, solved exactly by pool-adjacent-violators;
 * then the whole row is clipped into [lo, hi]. Labels keep their input order; returns left edges.
 */
function spread(cx: number[], w: number[], gap: number, lo: number, hi: number): number[] {
  const n = cx.length;
  const pre: number[] = [];
  let acc = 0;
  for (let i = 0; i < n; i++) {
    pre.push(acc);
    acc += w[i] + gap;
  }
  const total = acc - gap;
  // blocks of pooled values: [mean target, count]
  const blocks: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    blocks.push([cx[i] - w[i] / 2 - pre[i], 1]);
    while (blocks.length > 1 && blocks[blocks.length - 2][0] > blocks[blocks.length - 1][0]) {
      const [v2, c2] = blocks.pop()!;
      const [v1, c1] = blocks.pop()!;
      blocks.push([(v1 * c1 + v2 * c2) / (c1 + c2), c1 + c2]);
    }
  }
  const ys: number[] = [];
  for (const [v, c] of blocks) for (let j = 0; j < c; j++) ys.push(Math.min(Math.max(v, lo), Math.max(lo, hi - total)));
  return ys.map((y, i) => y + pre[i]);
}

/** cut text with "…" until it fits maxW */
function ellipsize(ctx: CanvasRenderingContext2D, t: string, maxW: number): string {
  if (ctx.measureText(t).width <= maxW) return t;
  let a = 0, b = t.length;
  while (a < b) {
    const mid = (a + b + 1) >> 1;
    if (ctx.measureText(t.slice(0, mid).trimEnd() + "…").width <= maxW) a = mid;
    else b = mid - 1;
  }
  return a ? t.slice(0, a).trimEnd() + "…" : "…";
}

export interface Hit {
  col: number;
  slot: number;
}

/** an axis-aligned box the layout reserved for one thing; tests check that boxes of different things never overlap */
export interface Box {
  kind: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

const SANS = `"Public Sans", system-ui, sans-serif`;
const MONO = `"JetBrains Mono", ui-monospace, monospace`;

export class Renderer {
  private ease: Float32Array[] | null = null;
  private lastTick = -1;
  private thumb = document.createElement("canvas");
  private frameCanvas = document.createElement("canvas");
  private tinted = new Map<string, HTMLCanvasElement>();
  private mapCanvas = document.createElement("canvas");
  // layout, recomputed in draw
  X: number[] = [];
  Y: number[][] = [];
  r = 10;
  /** per column: half-width of a node, where edges attach */
  half: number[] = [];
  mapSize = 0;
  pics: Pic[] = [];
  /** image node thumbnail [x, y, w, h] */
  imgRect: [number, number, number, number] | null = null;
  private headerY = 40;
  /** column titles as last drawn (for tests and hit checks); never overlapping */
  titleBoxes: TitleBox[] = [];
  /** everything the last layout placed: pictures, titles, label boxes, columns (for overlap tests) */
  boxes: Box[] = [];
  /** uniform shrink applied because the content did not fit the width at full size (1 = none) */
  fit = 1;
  /** the shrink this panel would pick on its own, before any fitCap */
  natFit = 1;
  /** per edge set (index into viz.edges): the y where its skip arcs peak */
  skipPeak = new Map<number, number>();
  /** per recurrent column: how far its loops may bulge to the right */
  loopRoom = new Map<number, number>();
  private bandTop = 0;
  private textX = 28;
  private titleLo = 0;
  private titleHi = 0;
  /** length unit: pixels per "720 p pixel", times the fit shrink */
  private u = 1;
  /** text sizes after fitting rows: input label / value, output label, output pill scale */
  private fonts = { inLabel: 17, inValue: 15, out: 16, ls: 1 };
  /** widest value readout per input column (monospace pattern), measured once */
  private valuePattern = new Map<number, string>();

  constructor(public viz: Viz, public labels: Labels, public opt: RenderOpts) {}

  resetEase() {
    this.ease = null;
    this.lastTick = -1;
  }

  /**
   * Lay everything out without overlap: pictures and text take their measured widths first, the network columns
   * get what is left (with a minimum spacing per pair of columns), and if even that does not fit, everything is
   * scaled down uniformly until it does. Returns the length unit to draw with.
   */
  private layout(ctx: CanvasRenderingContext2D): number {
    let f = 1;
    for (let it = 0; it < 10; it++) {
      const need = this.place(ctx, f);
      if (need <= this.opt.width + 0.5 || f <= 0.3) break;
      f = Math.max(0.3, f * Math.min(0.97, this.opt.width / need));
    }
    this.natFit = f;
    const cap = this.opt.fitCap ?? 1;
    if (cap < f) this.place(ctx, cap);
    return this.u;
  }

  /** one layout attempt at shrink factor f; returns the width it needs */
  private place(ctx: CanvasRenderingContext2D, f: number): number {
    const { width: W, height: H } = this.opt;
    const s = H / 720;
    const u = s * f;
    this.u = u;
    this.fit = f;
    const o = this.opt;
    const st = o.style;
    const m = st.padding * s;
    const cap = 30 * u;
    const boxes: Box[] = [];
    const font = (w: number, px: number, mono = false) => `${w} ${px}px ${mono ? MONO : SANS}`;
    const measure = (t: string, f0: string) => {
      ctx.font = f0;
      return ctx.measureText(t).width;
    };

    // ---- pictures: the recording's render and / or what the AI sees, each placed by its own position
    const want: { kind: Pic["kind"]; w0: number; h0: number; caption: string; pos: PicPos; scale: number }[] = [];
    const aiOn = !!o.image && st.aiPos !== "off";
    if (o.showFrame && o.frames && st.recPos !== "off") {
      const pos = st.recPos === "auto" ? (aiOn && st.aiPos === "above" ? "above" : "left") : st.recPos;
      want.push({ kind: "rec", w0: o.frames.shape[2], h0: o.frames.shape[1], caption: "Recording", pos, scale: st.recScale });
    }
    const heatOn = st.saliency && !!o.saliency;
    if (aiOn) want.push({ kind: "ai", w0: o.image!.W, h0: o.image!.H, caption: `What the AI sees (${o.image!.W}×${o.image!.H})${heatOn ? " · saliency" : ""}`, pos: st.aiPos, scale: st.aiScale });
    this.pics = [];
    let topBand = 0;
    const cols = this.viz.cols;
    // the least height each lane needs: a row per node with room for its label, an image node at a readable size
    const lanes = this.viz.lanes;
    const laneMin = Array.from({ length: lanes }, (_, l) => {
      const lc = cols.filter((c) => (lanes <= 1 ? true : c.lane === l));
      return Math.max(...lc.map((c) => (c.image ? 96 * s : c.sel.length * 22 * s)), 22 * s);
    });
    const fullMin = Math.max(0, ...cols.filter((c) => c.lane < 0).map((c) => c.sel.length * 22 * s));
    const laneGap = 30 * u;
    const netMin = Math.max(fullMin, laneMin.reduce((a, b) => a + b, 0) + laneGap * (lanes - 1)) + 2 * m + 48 * s;
    const above = want.filter((p) => p.pos === "above");
    if (above.length) {
      // a band across the top: it never competes for width, and it leaves the network the height it needs
      const capS = 30 * s;
      const gap = 40 * s;
      const base = Math.max(H * 0.15, Math.min(H * 0.42, H - netMin) - capS - 14 * s);
      let hs = above.map((p) => base * p.scale);
      let ws = above.map((p, i) => (hs[i] * p.w0) / p.h0);
      const total = ws.reduce((a, b) => a + b, 0) + gap * (above.length - 1);
      const k = Math.min(1, (W - 2 * m) / total);
      hs = hs.map((h) => h * k);
      ws = ws.map((w) => w * k);
      let x = (W - (ws.reduce((a, b) => a + b, 0) + gap * (above.length - 1))) / 2;
      const hMax = Math.max(...hs);
      above.forEach((p, i) => {
        this.pics.push({ kind: p.kind, x, y: 14 * s + capS + (hMax - hs[i]), w: ws[i], h: hs[i], caption: p.caption, corner: false, capPx: 15 * s });
        x += ws[i] + gap;
      });
      topBand = 14 * s + capS + hMax;
    }
    // beside and corner pictures take a column on their side; corners only pick the vertical alignment
    let leftEdge = m;
    let rightEdge = W - m;
    for (const side of ["left", "right"] as const) {
      const group = want
        .filter((p) => p.pos === side || (p.pos.length === 2 && p.pos[1] === side[0]))
        .sort((a, b) => (a.pos === side ? 1 : a.pos[0] === "t" ? 0 : 2) - (b.pos === side ? 1 : b.pos[0] === "t" ? 0 : 2));
      if (!group.length) continue;
      const gap = 20 * u;
      const yTop = topBand ? topBand + 16 * u : m;
      const yBot = H - m;
      const avail = yBot - yTop;
      const n = group.length;
      const each = Math.min((avail - n * cap - gap * (n - 1)) / n, H * 0.62);
      let sizes = group.map((p) => {
        const corner = p.pos.length === 2;
        let h = (corner ? H * 0.28 : each) * p.scale;
        let w = (h * p.w0) / p.h0;
        const maxW = W * (corner ? 0.35 : 0.3) * p.scale;
        if (w > maxW) {
          w = maxW;
          h = (w * p.h0) / p.w0;
        }
        return [w * f, h * f];
      });
      const fixedH = n * cap + gap * (n - 1);
      const picH = sizes.reduce((a, [, h]) => a + h, 0);
      if (picH + fixedH > avail) {
        const k = Math.max(0.05, (avail - fixedH) / picH);
        sizes = sizes.map(([w, h]) => [w * k, h * k]);
      }
      const totalH = sizes.reduce((a, [, h]) => a + h, 0) + fixedH;
      const corner = (p: (typeof group)[number], c: string) => p.pos.length === 2 && p.pos[0] === c;
      let y = group.every((p) => corner(p, "t")) ? yTop : group.every((p) => corner(p, "b")) ? yBot - totalH : yTop + (avail - totalH) / 2;
      const wMax = Math.max(...sizes.map(([w]) => w));
      group.forEach((p, i) => {
        const [w, h] = sizes[i];
        this.pics.push({ kind: p.kind, x: side === "left" ? m : W - m - w, y: y + cap, w, h, caption: p.caption, corner: false, capPx: 15 * u });
        y += cap + h + gap;
      });
      if (side === "left") leftEdge = m + wMax + 24 * u;
      else rightEdge = W - m - wMax - 24 * u;
    }
    // captions never run past their picture: shrink a little, then cut with "…"
    for (const p of this.pics) {
      let px = p.capPx;
      while (px > p.capPx * 0.75 && measure(p.caption, font(500, px)) > p.w) px *= 0.94;
      p.capPx = px;
      p.caption = ellipsize(ctx, p.caption, Math.max(p.w, 10 * u));
      boxes.push({ kind: `pic ${p.kind}`, x0: p.x, y0: p.y - 10 * u - px, x1: p.x + p.w, y1: p.y + p.h });
    }

    // ---- vertical: title row, then the band the columns live in, split into lanes for multi-input models
    this.headerY = topBand + m + 12 * u;
    this.bandTop = topBand;
    const nc = cols.length;
    const stages = this.viz.stages;
    const imgCol = cols.findIndex((c) => c.kind === "input" && c.image);
    const vecInputs = cols.map((c, ci) => (c.kind === "input" && !c.image ? ci : -1)).filter((ci) => ci >= 0);
    const outCi = nc - 1;
    const isMap = (c: VizColumn) => o.showMaps && !!c.maps;
    const bot = H - m - 2 * u;
    const P: number[] = new Array(nc);
    const p: number[] = new Array(nc);
    const band: [number, number][] = new Array(nc);
    const vertical = (top: number) => {
      const laneBand: [number, number][] = [];
      if (lanes <= 1) laneBand.push([top, bot]);
      else {
        const gap = laneGap;
        const w = laneMin;
        const span = bot - top - gap * (lanes - 1);
        const sum = w.reduce((a, b) => a + b, 0);
        let y = top;
        for (let l = 0; l < lanes; l++) {
          laneBand.push([y, y + (span * w[l]) / sum]);
          y += (span * w[l]) / sum + gap;
        }
      }
      cols.forEach((c, ci) => {
        band[ci] = c.lane < 0 ? [top, bot] : laneBand[c.lane];
        P[ci] = (band[ci][1] - band[ci][0]) / Math.max(1, c.sel.length);
      });
      // row pitch caps use the unshrunk unit: when the width forced a shrink, the rows still spread over the
      // height that is free, instead of the whole picture shrinking into the middle
      const pitch = Math.min(46 * s, ...P);
      cols.forEach((c, ci) => {
        // each column spreads over its own band; lanes of a multi-input model each have their own pitch
        const own = Math.min(46 * s, P[ci]);
        const want = c.kind === "hidden" ? own * st.rowSpacing : Math.min(own * 1.25, 56 * s) * st.rowSpacing;
        p[ci] = Math.min(want, P[ci]);
      });
      // node and tile sizes never exceed the row pitch they sit in
      const minP = (pred: (c: VizColumn) => boolean) => Math.min(Infinity, ...cols.filter((c) => pred(c) && c.sel.length > 1).map((c) => p[cols.indexOf(c)]));
      this.r = Math.min(Math.max(3, Math.min(11 * s * Math.sqrt(f), pitch * 0.3)) * st.nodeScale, 0.46 * minP((c) => !c.image && !isMap(c)));
      this.mapSize = Math.min(Math.max(6, Math.min(pitch * 0.78, 38 * u)) * st.mapScale, 0.9 * minP(isMap) - 4 * u);
      this.mapSize = Math.max(4, this.mapSize);
    };
    const titleRow = 15 * u;
    // skip arcs run between the titles and the columns: give them a rail's worth of room
    const arcRoom = this.viz.edges.some((e) => e.kind === "skip") ? 24 * u : 0;
    let top = this.headerY + titleRow / 2 + 26 * u + arcRoom;
    vertical(top);

    // ---- text widths, measured: input labels and values on the left, output pills and values on the right
    const fsIn = Math.min(st.labelSize * u, 0.9 * Math.min(Infinity, ...vecInputs.filter((ci) => cols[ci].sel.length > 1).map((ci) => p[ci])));
    const fsVal = Math.min(st.valueSize * u, fsIn * 1.1, 0.9 * Math.min(Infinity, ...vecInputs.filter((ci) => cols[ci].sel.length > 1).map((ci) => p[ci])));
    const outCol = cols[outCi];
    const ls0 = (st.labelSize - 1) / 16;
    const q = outCol.sel.length > 1 ? Math.min(1, (0.92 * p[outCi]) / (30 * u * ls0)) : 1;
    this.fonts = { inLabel: fsIn, inValue: fsVal, out: (st.labelSize - 1) * u * q, ls: ls0 * q };
    const r = this.r;
    let labelW = 0;
    for (const ci of vecInputs) {
      const c = cols[ci];
      const vw = st.showValues ? measure(this.valueText(ci), font(400, fsVal, true)) : 0;
      for (const unit of c.sel) {
        const lw = measure(this.inputLabel(c, unit), font(600, fsIn));
        labelW = Math.max(labelW, lw + (vw ? 14 * u + vw : 0) + 12 * u + r);
      }
    }
    // image node size: the frame at a size where its pixels are visible, inside its band (with the stack below it)
    let tw = 0;
    let th = 0;
    let stackH = 0;
    if (imgCol >= 0 && o.image) {
      const iv = o.image;
      const [b0, b1] = band[imgCol];
      th = Math.min(124 * u, (b1 - b0) * 0.5) * st.imageScale;
      tw = Math.min(160 * u * st.imageScale, (th * iv.W) / iv.H);
      th = (tw * iv.H) / iv.W;
      const stack = (w: number) => (iv.frames > 1 ? (Math.min(w / 3 - 4 * u, 44 * u) * iv.H) / iv.W + 3 * u + 24 * u : 0);
      const room = b1 - b0 + 2 * 11 * u;
      if (th + stack(tw) > room) {
        const k = room / (th + stack(tw));
        tw *= k;
        th *= k;
      }
      stackH = stack(tw);
      const lw = measure(this.labels.inputs[cols[imgCol].base ?? 0] ?? "image", font(600, st.labelSize * u));
      labelW = Math.max(labelW, lw + 14 * u + 3 * u + tw);
    }
    const isz = 22 * u * this.fonts.ls;
    let outW = 0;
    const probs = st.showProbs && (this.viz.mode === "continuous" || outCol.sel.length <= 8);
    const probW = probs ? measure(this.viz.mode === "categorical" ? "100%" : "−0.00", font(400, (st.valueSize - 2) * u, true)) : 0;
    for (const unit of outCol.sel) {
      const label = this.labels.outputs[unit] ?? `out ${unit}`;
      const pill = (this.labels.icons[unit] ? isz + 8 * u : 0) + measure(label, font(700, this.fonts.out)) + 16 * u;
      outW = Math.max(outW, r + 4.5 * u + 11.5 * u + pill + (probs ? 4 * u + probW : 0));
    }

    // ---- horizontal: columns share what is left, never closer than their nodes need
    const halfR = new Array(stages).fill(0);
    const halfL = new Array(stages).fill(0);
    cols.forEach((c, ci) => {
      const hw = c.image ? 2 * u : isMap(c) ? this.mapSize / 2 + 2 * u : ci === outCi ? r + 4.5 * u : r;
      // a recurrent column's loops bulge out to its right
      halfR[c.stage] = Math.max(halfR[c.stage], hw + (c.recurrent ? 42 * u : 0));
      halfL[c.stage] = Math.max(halfL[c.stage], c.image ? 0 : hw);
    });
    const minSp = Array.from({ length: Math.max(0, stages - 1) }, (_, i) => halfR[i] + halfL[i + 1] + 30 * u);
    const sumMin = minSp.reduce((a, b) => a + b, 0);
    this.textX = leftEdge;
    const leftX = leftEdge + labelW;
    const rightX = rightEdge - outW;
    const avail = rightX - leftX;
    const need = leftEdge + labelW + sumMin + outW + (W - rightEdge);
    const span = Math.max(sumMin, Math.min(avail, avail * st.colSpacing));
    // even spacing where possible; a pair that needs more than the even share keeps its minimum
    let lam = minSp.length ? span / minSp.length : 0;
    for (let it = 0; it < minSp.length; it++) {
      const fixed = minSp.filter((x) => x > lam);
      const nl = minSp.length - fixed.length;
      const next = nl ? (span - fixed.reduce((a, b) => a + b, 0)) / nl : lam;
      if (Math.abs(next - lam) < 1e-6) break;
      lam = next;
    }
    const xs = [leftX + Math.max(0, avail - span) / 2];
    for (let i = 0; i < minSp.length; i++) xs.push(xs[i] + Math.max(minSp[i], lam));
    this.X = cols.map((c) => xs[c.stage]);
    this.half = cols.map((c, ci) => (c.image ? 2 * u : isMap(c) ? this.mapSize / 2 : ci === outCi ? r : r));
    this.titleLo = leftEdge > m ? leftEdge : 6 * s;
    this.titleHi = rightEdge < W - m ? rightEdge : W - 6 * s;
    this.imgRect = imgCol >= 0 && o.image ? [this.X[imgCol] - tw, 0, tw, th] : null;
    this.titleBoxes = st.titles ? this.placeTitles(ctx, u) : [];
    // a second title row below the first pushes the columns down
    const titleBottom = Math.max(this.headerY + titleRow / 2, ...this.titleBoxes.map((t) => t.y1));
    if (titleBottom + 26 * u + arcRoom > top) {
      top = titleBottom + 26 * u + arcRoom;
      vertical(top);
    }

    // ---- rows
    const halfV = (c: VizColumn, ci: number) => (isMap(c) ? this.mapSize / 2 + 2 * u : ci === outCi ? Math.max(r + 4.5 * u, 15 * u * this.fonts.ls) : vecInputs.includes(ci) ? Math.max(r, fsIn * 0.6) : r);
    this.Y = cols.map((c, ci) => {
      const n = c.sel.length;
      const [b0, b1] = band[ci];
      const hv = halfV(c, ci);
      const pp = Math.min(p[ci], n > 1 ? (b1 - b0 - 2 * hv) / (n - 1) : Infinity);
      const span = pp * (n - 1);
      const y0 = (b0 + b1) / 2 - span / 2;
      const ys = new Array(n);
      c.order.forEach((ii, slot) => (ys[ii] = y0 + slot * pp));
      return ys;
    });
    if (this.imgRect) {
      const [b0, b1] = band[imgCol];
      let y = this.Y[imgCol][0] - th / 2;
      y = Math.max(b0 - 11 * u + 3 * u, Math.min(y, b1 + 11 * u - th - stackH));
      this.imgRect[1] = y;
      this.Y[imgCol][0] = y + th / 2;
    }

    // ---- recurrent loops: room to the right of the column, half the way to the next one
    this.loopRoom.clear();
    cols.forEach((c, ci) => {
      if (!c.recurrent) return;
      const next = Math.min(...cols.filter((d) => d.stage === c.stage + 1).map((d) => this.X[cols.indexOf(d)] - this.half[cols.indexOf(d)]));
      const room = Math.max(14 * u, Math.min(64 * u, (next - this.X[ci] - this.half[ci]) * 0.55));
      this.loopRoom.set(ci, room);
      const yb = Math.max(...this.Y[ci]) + r + 16 * u;
      const lw = measure("h(t−1)", font(500, 11 * u));
      const x0 = this.X[ci] + this.half[ci] + 4 * u;
      boxes.push({ kind: `loop label ${ci}`, x0, x1: x0 + lw, y0: yb - 11 * u * 0.6, y1: yb + 11 * u * 0.4 });
    });

    // ---- skip arcs peak above the columns they bypass, below the titles (and inside their lane)
    this.skipPeak.clear();
    const titleFloor = Math.max(this.headerY + titleRow / 2, ...this.titleBoxes.map((t) => t.y1));
    this.viz.edges.forEach((e, ei) => {
      if (e.kind !== "skip") return;
      const a = cols[e.from], b = cols[e.to];
      const lane = b.lane >= 0 ? b.lane : a.lane;
      let yTop = Math.min(...this.Y[e.from], ...this.Y[e.to]) - r;
      cols.forEach((c, ci) => {
        if (c.stage > a.stage && c.stage < b.stage && (c.lane === lane || c.lane < 0 || lane < 0)) yTop = Math.min(yTop, Math.min(...this.Y[ci]) - (isMap(c) ? this.mapSize / 2 : r));
      });
      const floor = lanes > 1 && lane > 0 ? band[e.to][0] - laneGap * 0.3 : titleFloor + 24 * u;
      const y = Math.max(floor, yTop - 16 * u);
      this.skipPeak.set(ei, y);
      const xm = (this.X[e.from] + this.X[e.to]) / 2;
      const lw = measure(e.proj ? "projection" : "skip", font(500, 11 * u));
      boxes.push({ kind: `skip label ${ei}`, x0: xm - lw / 2, x1: xm + lw / 2, y0: y - 9 * u - 11 * u * 0.6, y1: y - 9 * u + 11 * u * 0.4 });
    });

    // ---- boxes, for overlap checks
    for (const t of this.titleBoxes) boxes.push({ kind: `title ${t.text}`, x0: t.x0, y0: t.y0, x1: t.x1, y1: t.y1 });
    cols.forEach((c, ci) => {
      if (c.image) return;
      const ys = this.Y[ci];
      const hv = isMap(c) ? this.mapSize / 2 + 2 * u : ci === outCi ? r + 4.5 * u : r;
      const hw = isMap(c) ? this.mapSize / 2 + 2 * u : ci === outCi ? r + 4.5 * u : r;
      boxes.push({ kind: `column ${ci}`, x0: this.X[ci] - hw, x1: this.X[ci] + hw, y0: Math.min(...ys) - hv, y1: Math.max(...ys) + hv });
    });
    for (const ci of vecInputs) {
      const c = cols[ci];
      const vw = st.showValues ? measure(this.valueText(ci), font(400, fsVal, true)) : 0;
      c.sel.forEach((unit, ii) => {
        const y = this.Y[ci][ii];
        const lw = measure(this.inputLabel(c, unit), font(600, fsIn));
        boxes.push({ kind: `input label ${ci}:${unit}`, x0: this.textX, x1: this.textX + lw, y0: y - fsIn * 0.45, y1: y + fsIn * 0.45 });
        const xv = this.X[ci] - r - 12 * u;
        if (vw) boxes.push({ kind: `input value ${ci}:${unit}`, x0: xv - vw, x1: xv, y0: y - fsVal * 0.45, y1: y + fsVal * 0.45 });
      });
    }
    if (this.imgRect) {
      const [x, y, w, h] = this.imgRect;
      const b = 3 * u;
      boxes.push({ kind: "image node", x0: x - b, y0: y - b, x1: x + w + b, y1: y + h + b + stackH });
      const lw = measure(this.labels.inputs[cols[imgCol].base ?? 0] ?? "image", font(600, st.labelSize * u));
      boxes.push({ kind: "image label", x0: x - b - 14 * u - lw, x1: x - b - 14 * u, y0: y + h / 2 - st.labelSize * u * 0.45, y1: y + h / 2 + st.labelSize * u * 0.45 });
    }
    outCol.sel.forEach((unit, ii) => {
      const y = this.Y[outCi][ii];
      const label = this.labels.outputs[unit] ?? `out ${unit}`;
      const pill = (this.labels.icons[unit] ? isz + 8 * u : 0) + measure(label, font(700, this.fonts.out)) + 16 * u;
      const ix = this.X[outCi] + r + 16 * u;
      boxes.push({ kind: `output label ${unit}`, x0: ix - 6 * u, x1: ix - 6 * u + pill + (probs ? 10 * u + probW : 0), y0: y - 15 * u * this.fonts.ls, y1: y + 15 * u * this.fonts.ls });
    });
    this.boxes = boxes;
    return need;
  }

  /** name of unit `unit` of input column c */
  private inputLabel(c: VizColumn, unit: number) {
    return this.labels.inputs[(c.base ?? 0) + unit] ?? `in ${unit}`;
  }

  /** the widest value readout of an input column, as a monospace pattern (so it is measured once) */
  private valueText(ci: number) {
    let t = this.valuePattern.get(ci);
    if (t === undefined) {
      const c = this.viz.cols[ci];
      let big = 0;
      for (const x of c.raw ?? []) big = Math.max(big, Math.abs(x));
      const unitW = Math.max(0, ...c.sel.map((unit) => (this.labels.units[(c.base ?? 0) + unit] ?? "").length));
      t = (big >= 1000 ? "−0.0e+0" : big >= 100 ? "−000" : "−00.0".slice(big >= 10 ? 0 : 1) + "0") + (unitW ? " " + "0".repeat(unitW) : "");
      this.valuePattern.set(ci, t);
    }
    return t;
  }

  /**
   * Column titles centred on their columns when they fit; when columns are squeezed or the canvas is crowded,
   * first nudge them apart (as little as possible, least squares), then alternate them over two rows if a nudge
   * exceeds 20 px, then shrink the font,
   * and finally cut the longest ones with "…". Sets ctx.font to the chosen size.
   * Columns that share a stage (one per branch) share one title.
   */
  private placeTitles(ctx: CanvasRenderingContext2D, s: number): TitleBox[] {
    const cols = this.viz.cols;
    const byStage = new Map<number, { texts: string[]; cx: number }>();
    cols.forEach((c, ci) => {
      const cx = this.X[ci] + (c.kind === "output" ? 40 * s : c.image && this.imgRect ? -this.imgRect[2] / 2 : 0);
      const e = byStage.get(c.stage);
      if (!e) byStage.set(c.stage, { texts: [c.label], cx });
      else {
        if (!e.texts.includes(c.label)) e.texts.push(c.label);
        if (c.image) e.cx = cx;
      }
    });
    // kept in column order, even when squeezed columns cross: the titles then still read input -> output
    const items = [...byStage.entries()].sort((a, b) => a[0] - b[0]).map(([, e]) => ({ text: e.texts.join(" + "), cx: e.cx }));
    if (!items.length) return [];
    const gap = 14 * s;
    const lo = this.titleLo, hi = this.titleHi;
    const font = (px: number) => `500 ${px}px ${SANS}`;
    const f0 = 15 * s;
    // how far a layout moves titles off their columns (0 = all centred, no overlap)
    const solve = (px: number, rows: number) => {
      ctx.font = font(px);
      const w = items.map((t) => ctx.measureText(t.text).width);
      const x = new Array<number>(items.length);
      let worst = 0;
      let fits = true;
      for (let r = 0; r < rows; r++) {
        const idx = items.map((_, i) => i).filter((i) => i % rows === r);
        const ws = idx.map((i) => w[i]);
        if (ws.reduce((a, b) => a + b, 0) + gap * (idx.length - 1) > hi - lo) fits = false;
        const xs = spread(idx.map((i) => items[i].cx), ws, gap, lo, hi);
        idx.forEach((i, k) => {
          x[i] = xs[k];
          worst = Math.max(worst, Math.abs(xs[k] + w[i] / 2 - items[i].cx));
        });
      }
      return { px, rows, w, x, worst, fits };
    };
    // one row unless that pushes a title more than 20 px off its column
    let best = solve(f0, 1);
    if (best.worst > 20 * s) {
      const two = solve(f0, 2);
      if (two.worst < best.worst) best = two;
    }
    // too wide even when nudged apart: shrink, down to 70 %
    for (let px = f0 * 0.9; !best.fits && px >= f0 * 0.7 - 1e-6; px -= f0 * 0.1) {
      const one = solve(px, 1);
      const two = solve(px, 2);
      best = one.fits && one.worst <= two.worst ? one : two;
    }
    ctx.font = font(best.px);
    let { w, x } = best;
    const texts = items.map((t) => t.text);
    if (!best.fits) {
      // still too wide: cap every title at one shared width (longest ones get cut), then place again
      for (let r = 0; r < best.rows; r++) {
        const idx = items.map((_, i) => i).filter((i) => i % best.rows === r);
        const room = hi - lo - gap * (idx.length - 1);
        const sorted = idx.map((i) => w[i]).sort((a, b) => a - b);
        let cap = room / idx.length;
        let used = 0;
        for (let k = 0; k < sorted.length; k++) {
          const c = (room - used) / (sorted.length - k);
          if (sorted[k] > c) {
            cap = c;
            break;
          }
          used += sorted[k];
          cap = Infinity;
        }
        for (const i of idx) {
          texts[i] = ellipsize(ctx, texts[i], cap);
          w[i] = ctx.measureText(texts[i]).width;
        }
        const xs = spread(idx.map((i) => items[i].cx), idx.map((i) => w[i]), gap, lo, hi);
        idx.forEach((i, k) => (x[i] = xs[k]));
      }
    }
    // second row goes above the first when there is room, else below it
    const line = best.px * 1.3;
    const up = this.headerY - line - best.px / 2 >= this.bandTop + 4 * s;
    return items.map((_, i) => {
      const y = this.headerY + (i % best.rows === 1 ? (up ? -line : line) : 0);
      return { text: texts[i], x0: x[i], x1: x[i] + w[i], y, y0: y - best.px / 2, y1: y + best.px / 2 };
    });
  }

  /** draw recording time tRec (seconds); dt = seconds of wall/video time since the previous frame (0 = snap) */
  draw(ctx: CanvasRenderingContext2D, tRec: number, dt: number) {
    const o = this.opt;
    const th = o.theme;
    const viz = this.viz;
    const T = viz.T;
    const tickF = Math.max(0, tRec * o.hz);
    const k = Math.min(T - 1, Math.floor(tickF));
    ctx.save();
    const s = this.layout(ctx);
    const W = o.width, H = o.height;
    const st = o.style;
    const C = { off: hex(th.off), on: hex(th.on), pos: hex(st.pos || th.pos), neg: hex(st.neg || th.neg), bg: hex(st.bg || th.bg), text: hex(th.text), dim: hex(th.dim), pulse: hex(th.pulse) };

    // easing towards this tick's values (30 ms time constant in recording time)
    const target = viz.cols.map((c) => c.v.subarray(k * c.sel.length, (k + 1) * c.sel.length));
    if (!this.ease || dt <= 0 || Math.abs(k - this.lastTick) > 5) this.ease = target.map((t) => Float32Array.from(t));
    else {
      const a = 1 - Math.exp(-(dt * o.speed) / 0.03);
      this.ease.forEach((e, ci) => {
        for (let i = 0; i < e.length; i++) e[i] += (target[ci][i] - e[i]) * a;
      });
    }
    this.lastTick = k;
    const E = this.ease;

    ctx.clearRect(0, 0, W, H);
    if (!o.transparent) {
      ctx.fillStyle = st.bg || th.bg;
      ctx.fillRect(0, 0, W, H);
    }

    // pulse: sweep once per decision tick, or once per m ticks when ticks come faster than ~4 frames
    const ticksPerSec = o.hz * o.speed;
    const m = Math.max(1, Math.ceil((ticksPerSec * 4) / o.fps));
    const ph = (tickF / m) % 1;
    const nc = viz.cols.length;
    const ns = viz.stages;
    if (st.pulse)
      for (let ci = 0; ci < nc; ci++) {
        const col = viz.cols[ci];
        if (col.image) continue;
        const g = Math.exp(-((ph * (ns + 1) - col.stage - 0.5) ** 2) / 0.35);
        if (g < 0.05) continue;
        const x = this.X[ci];
        const ys = this.Y[ci];
        const yTop = Math.min(...ys) - 22 * s;
        const yBot = Math.max(...ys) + 22 * s;
        ctx.fillStyle = rgba(C.pulse, 0.06 * g * st.pulseIntensity);
        roundRect(ctx, x - 20 * s, yTop, 40 * s, yBot - yTop, 14 * s);
        ctx.fill();
      }

    // edges
    ctx.lineCap = "round";
    viz.edges.forEach((e, ei) => {
      const from = viz.cols[e.from];
      const to = viz.cols[e.to];
      const nf = from.sel.length;
      const nt = to.sel.length;
      const xs = this.X[e.from] + this.half[e.from];
      const xd = this.X[e.to] - this.half[e.to];
      const kk = Math.min(o.topK, nf);
      const base = k * nt * nf;
      const peak = this.skipPeak.get(ei);
      if (e.kind === "skip" && peak !== undefined) {
        ctx.font = `500 ${11 * s}px ${SANS}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "alphabetic";
        ctx.fillStyle = th.dim;
        ctx.fillText(e.proj ? "projection" : "skip", (this.X[e.from] + this.X[e.to]) / 2, peak - 9 * s);
      }
      if (e.kind === "recurrent") {
        ctx.font = `500 ${11 * s}px ${SANS}`;
        ctx.textAlign = "left";
        ctx.textBaseline = "alphabetic";
        ctx.fillStyle = th.dim;
        ctx.fillText("h(t−1)", this.X[e.to] + this.half[e.to] + 4 * s, Math.max(...this.Y[e.to]) + this.r + 16 * s + 4 * s);
        ctx.setLineDash([4 * s, 3.5 * s]);
      }
      for (let j = 0; j < nt; j++) {
        const row = e.c.subarray(base + j * nf, base + (j + 1) * nf);
        const idx = topK(row, kk);
        for (const i of idx) {
          const v = row[i];
          const str = Math.min(Math.abs(v) / e.scale, 1);
          if (str < 0.06) continue;
          ctx.strokeStyle = rgba(v > 0 ? C.pos : C.neg, 0.12 + 0.68 * str);
          ctx.lineWidth = (st.edgeMin + (st.edgeMax - st.edgeMin) * str) * s;
          ctx.beginPath();
          const y0 = this.Y[e.from][i];
          const y1 = this.Y[e.to][j];
          if (peak !== undefined) {
            // over the block: rise to a rail above it, run along the rail, come down into the target
            const w = Math.min((xd - xs) * 0.3, 90 * s);
            ctx.moveTo(xs, y0);
            ctx.bezierCurveTo(xs + w * 0.55, y0, xs + w * 0.45, peak, xs + w, peak);
            ctx.lineTo(xd - w, peak);
            ctx.bezierCurveTo(xd - w * 0.45, peak, xd - w * 0.55, y1, xd, y1);
          } else if (e.kind === "recurrent") {
            // h(t-1) of unit i into unit j: a loop out to the right of the column and back
            const x = this.X[e.to] + this.half[e.to] + 2 * s;
            const same = i === j;
            const ya = same ? y0 - 3.5 * s : y0;
            const yb = same ? y1 + 3.5 * s : y1;
            const bulge = Math.min(this.loopRoom.get(e.to) ?? 40 * s, 12 * s + Math.abs(y1 - y0) * 0.45);
            ctx.moveTo(x, ya);
            ctx.bezierCurveTo(x + bulge, ya, x + bulge, yb, x, yb);
          } else {
            const mx = (xs + xd) / 2;
            ctx.moveTo(xs, y0);
            ctx.bezierCurveTo(mx, y0, mx, y1, xd, y1);
          }
          ctx.stroke();
        }
      }
      if (e.kind === "recurrent") ctx.setLineDash([]);
    });

    // pictures: recording render and what the AI sees, each with a caption
    for (const p of this.pics) {
      ctx.font = `500 ${p.capPx}px ${SANS}`;
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      ctx.fillStyle = th.dim;
      ctx.fillText(p.caption, p.x, p.y - 10 * s);
      if (p.kind === "rec") this.blitRecording(ctx, k, p.x, p.y, p.w, p.h);
      else {
        this.blitInput(ctx, k, 0, p.x, p.y, p.w, p.h);
        this.drawHeat(ctx, k, p.x, p.y, p.w, p.h, C);
      }
    }

    // column headings, after the pictures so a title is never hidden under one
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillStyle = th.dim;
    if (this.titleBoxes.length) ctx.font = `500 ${this.titleBoxes[0].y1 - this.titleBoxes[0].y0}px ${SANS}`;
    for (const t of this.titleBoxes) ctx.fillText(t.text, t.x0, t.y);

    // input columns: an image node, or one labelled row per number
    for (let ci = 0; ci < nc; ci++) {
      const inCol = viz.cols[ci];
      if (inCol.kind !== "input") continue;
      if (inCol.image) {
        if (this.imgRect) this.drawImageNode(ctx, k, E[ci][0], C, s, this.labels.inputs[inCol.base ?? 0] ?? "image");
        continue;
      }
      const xIn = this.X[ci];
      inCol.sel.forEach((unit, ii) => {
        const y = this.Y[ci][ii];
        ctx.textAlign = "left";
        ctx.fillStyle = th.text;
        ctx.font = `600 ${this.fonts.inLabel}px ${SANS}`;
        ctx.fillText(this.inputLabel(inCol, unit), this.textX, y);
        if (st.showValues) {
          const raw = inCol.raw![k * inCol.sel.length + ii];
          ctx.textAlign = "right";
          ctx.fillStyle = th.dim;
          ctx.font = `400 ${this.fonts.inValue}px ${MONO}`;
          const un = this.labels.units[(inCol.base ?? 0) + unit];
          ctx.fillText(fmt(raw) + (un ? ` ${un}` : ""), xIn - this.r - 12 * s, y);
        }
        this.node(ctx, xIn, y, E[ci][ii], C, s, false, 0);
      });
    }

    // hidden columns
    for (let ci = 0; ci < nc; ci++) {
      const col = viz.cols[ci];
      if (col.kind !== "hidden") continue;
      const maps = o.showMaps ? col.maps : undefined;
      col.sel.forEach((_, ii) => (maps ? this.mapNode(ctx, k, col, ii, this.X[ci], this.Y[ci][ii], E[ci][ii], C, s) : this.node(ctx, this.X[ci], this.Y[ci][ii], E[ci][ii], C, s, false, 0)));
    }

    // outputs
    const outCol = viz.cols[nc - 1];
    const xo = this.X[nc - 1];
    const ch = viz.chosen ? viz.chosen[k] : -1;
    const ls = this.fonts.ls;
    outCol.sel.forEach((unit, ii) => {
      const y = this.Y[nc - 1][ii];
      const p = E[nc - 1][ii];
      const isChosen = unit === ch;
      this.node(ctx, xo, y, p, C, s, isChosen, p);
      const ix = xo + this.r + 16 * s;
      const isz = 22 * s * ls;
      const icon = this.labels.icons[unit];
      const label = this.labels.outputs[unit] ?? `out ${unit}`;
      ctx.font = `${isChosen ? 700 : 500} ${this.fonts.out}px ${SANS}`;
      const tw = ctx.measureText(label).width;
      const pillW = (icon ? isz + 8 * s : 0) + tw + 16 * s;
      if (isChosen) {
        ctx.fillStyle = th.on;
        roundRect(ctx, ix - 6 * s, y - 15 * s * ls, pillW, 30 * s * ls, 8 * s * ls);
        ctx.fill();
      }
      const fg = isChosen ? th.bg : p > 0.5 && viz.mode === "continuous" ? th.text : th.dim;
      let tx = ix + 2 * s;
      if (icon) {
        ctx.drawImage(this.tint(icon, fg, Math.round(isz)), tx, y - isz / 2, isz, isz);
        tx += isz + 8 * s;
      }
      ctx.textAlign = "left";
      ctx.fillStyle = isChosen ? th.bg : th.text;
      ctx.fillText(label, tx, y + 0.5 * s);
      if (st.showProbs && (viz.mode === "continuous" || outCol.sel.length <= 8)) {
        ctx.font = `400 ${(st.valueSize - 2) * s}px ${MONO}`;
        ctx.fillStyle = th.dim;
        const val = viz.mode === "categorical" ? `${Math.round(outCol.raw![k * outCol.sel.length + ii] * 100)}%` : fmt(outCol.raw![k * outCol.sel.length + ii]);
        ctx.fillText(val, ix + pillW + 4 * s, y + 0.5 * s);
      }
    });
    ctx.restore();
  }

  private node(ctx: CanvasRenderingContext2D, x: number, y: number, v: number, C: Record<string, RGB>, s: number, ring: boolean, glow: number) {
    const r = this.r;
    if (glow > 0.03 || v > 0.55) {
      const g = Math.max(glow, (v - 0.55) / 0.45);
      const R = r + 9 * s;
      const grad = ctx.createRadialGradient(x, y, r * 0.6, x, y, R);
      grad.addColorStop(0, rgba(C.on, Math.min(1, 0.42 * g * this.opt.style.glow)));
      grad.addColorStop(1, rgba(C.on, 0));
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, R, 0, Math.PI * 2);
      ctx.fill();
    }
    // gamma 0.7: perceptual lift so mid activations read as lit; still monotonic in the activation
    ctx.fillStyle = mix(C.off, C.on, Math.max(0, Math.min(1, v)) ** 0.7);
    const square = this.opt.style.shape === "square";
    const shape = (rr: number) => {
      ctx.beginPath();
      if (square) ctx.roundRect(x - rr, y - rr, 2 * rr, 2 * rr, Math.min(3 * s, rr * 0.3));
      else ctx.arc(x, y, rr, 0, Math.PI * 2);
    };
    shape(r);
    ctx.fill();
    if (ring) {
      ctx.strokeStyle = rgba(C.text, 1);
      ctx.lineWidth = 2.5 * s;
      shape(r + 4.5 * s);
      ctx.stroke();
    }
  }

  /** nearest-neighbour blit of input frame `age` (0 = newest) */
  private blitInput(ctx: CanvasRenderingContext2D, k: number, age: number, x: number, y: number, w: number, h: number) {
    const iv = this.opt.image!;
    if (this.thumb.width !== iv.W || this.thumb.height !== iv.H) {
      this.thumb.width = iv.W;
      this.thumb.height = iv.H;
    }
    const tc = this.thumb.getContext("2d")!;
    const img = tc.createImageData(iv.W, iv.H);
    frameRGBA(this.opt.obs, iv, k, age, img.data);
    tc.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.thumb, x, y, w, h);
    ctx.imageSmoothingEnabled = true;
  }

  /** saliency overlay: warm = pushes the chosen output up, cool = down; alpha grows with |grad × input| */
  private heatCanvas = document.createElement("canvas");
  private drawHeat(ctx: CanvasRenderingContext2D, k: number, x: number, y: number, w: number, h: number, C: Record<string, RGB>) {
    const sm = this.opt.saliency;
    if (!sm || !this.opt.style.saliency) return;
    const { H, W } = sm;
    if (this.heatCanvas.width !== W || this.heatCanvas.height !== H) {
      this.heatCanvas.width = W;
      this.heatCanvas.height = H;
    }
    const hc = this.heatCanvas.getContext("2d")!;
    const img = hc.createImageData(W, H);
    const off = k * H * W;
    const op = this.opt.style.saliencyOpacity;
    for (let p = 0; p < H * W; p++) {
      const v = sm.heat[off + p] / sm.scale;
      const c = v > 0 ? C.pos : C.neg;
      const a = Math.min(1, Math.abs(v)) ** 0.7;
      img.data[p * 4] = c[0];
      img.data[p * 4 + 1] = c[1];
      img.data[p * 4 + 2] = c[2];
      img.data[p * 4 + 3] = Math.round(255 * a * op);
    }
    hc.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.heatCanvas, x, y, w, h);
    ctx.imageSmoothingEnabled = true;
  }

  private blitRecording(ctx: CanvasRenderingContext2D, k: number, x: number, y: number, w: number, h: number) {
    const f = this.opt.frames!;
    const [, fh, fw] = f.shape;
    this.frameCanvas.width = fw;
    this.frameCanvas.height = fh;
    const fc = this.frameCanvas.getContext("2d")!;
    const img = fc.createImageData(fw, fh);
    const src = f.data as Uint8Array;
    const off = k * fh * fw * 3;
    for (let p = 0, q = 0; p < fh * fw; p++, q += 4) {
      img.data[q] = src[off + p * 3];
      img.data[q + 1] = src[off + p * 3 + 1];
      img.data[q + 2] = src[off + p * 3 + 2];
      img.data[q + 3] = 255;
    }
    fc.putImageData(img, 0, 0);
    ctx.drawImage(this.frameCanvas, x, y, w, h);
  }

  /** the input column of an image model: current frame, border lit by how much it changed, older stack frames below */
  private drawImageNode(ctx: CanvasRenderingContext2D, k: number, v: number, C: Record<string, RGB>, s: number, label: string) {
    // (the heatmap goes over the thumbnail too, so a viewer without the big AI view still sees it)
    const [x, y, w, h] = this.imgRect!;
    const iv = this.opt.image!;
    const b = 3 * s;
    if (v > 0.3) {
      ctx.fillStyle = rgba(C.on, Math.min(1, ((0.25 * (v - 0.3)) / 0.7) * this.opt.style.glow));
      roundRect(ctx, x - b - 6 * s, y - b - 6 * s, w + 2 * b + 12 * s, h + 2 * b + 12 * s, 8 * s);
      ctx.fill();
    }
    ctx.fillStyle = mix(C.off, C.on, Math.min(1, v) ** 0.7);
    ctx.fillRect(x - b, y - b, w + 2 * b, h + 2 * b);
    this.blitInput(ctx, k, 0, x, y, w, h);
    this.drawHeat(ctx, k, x, y, w, h, C);
    ctx.textAlign = "right";
    ctx.fillStyle = this.opt.theme.text;
    ctx.font = `600 ${this.opt.style.labelSize * s}px "Public Sans", system-ui, sans-serif`;
    ctx.fillText(label, x - b - 14 * s, y + h / 2);
    // older frames of the stack, newest first, small and dimmer
    const older = Math.min(iv.frames - 1, 3);
    if (older > 0) {
      const sw = Math.min(w / 3 - 4 * s, 44 * s);
      const sh = (sw * iv.H) / iv.W;
      ctx.font = `400 ${12 * s}px "Public Sans", system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.fillStyle = this.opt.theme.dim;
      for (let a = 1; a <= older; a++) {
        const sx = x + w - a * (sw + 6 * s) + 6 * s;
        const sy = y + h + b + 10 * s;
        ctx.globalAlpha = 0.75;
        this.blitInput(ctx, k, a, sx, sy, sw, sh);
        ctx.globalAlpha = 1;
        ctx.fillText(`t−${a}`, sx + sw / 2, sy + sh + 10 * s);
      }
    }
  }

  /** a conv channel drawn as its live activation map; outline lit by the channel's mean activation */
  private mapNode(ctx: CanvasRenderingContext2D, k: number, col: Viz["cols"][number], ii: number, x: number, y: number, v: number, C: Record<string, RGB>, s: number) {
    const { H, W, data } = col.maps!;
    const q = this.mapSize;
    const ns = col.sel.length;
    if (this.mapCanvas.width !== W || this.mapCanvas.height !== H) {
      this.mapCanvas.width = W;
      this.mapCanvas.height = H;
    }
    const mc = this.mapCanvas.getContext("2d")!;
    const img = mc.createImageData(W, H);
    const off = (k * ns + ii) * H * W;
    for (let p = 0; p < H * W; p++) {
      const t = (data[off + p] / 255) ** 0.8;
      img.data[p * 4] = C.off[0] + (C.on[0] - C.off[0]) * t;
      img.data[p * 4 + 1] = C.off[1] + (C.on[1] - C.off[1]) * t;
      img.data[p * 4 + 2] = C.off[2] + (C.on[2] - C.off[2]) * t;
      img.data[p * 4 + 3] = 255;
    }
    mc.putImageData(img, 0, 0);
    const b = Math.max(1.5, 2 * s);
    ctx.fillStyle = mix(C.off, C.on, Math.min(1, v) ** 0.7);
    roundRect(ctx, x - q / 2 - b, y - q / 2 - b, q + 2 * b, q + 2 * b, 3 * s);
    ctx.fill();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.mapCanvas, x - q / 2, y - q / 2, q, q);
    ctx.imageSmoothingEnabled = true;
  }

  private tint(img: HTMLImageElement, color: string, size: number): HTMLCanvasElement {
    const key = `${img.src.slice(-80)}|${color}|${size}`;
    let c = this.tinted.get(key);
    if (c) return c;
    c = document.createElement("canvas");
    c.width = c.height = size * 2;
    const g = c.getContext("2d")!;
    g.drawImage(img, 0, 0, size * 2, size * 2);
    g.globalCompositeOperation = "source-in";
    g.fillStyle = color;
    g.fillRect(0, 0, size * 2, size * 2);
    this.tinted.set(key, c);
    return c;
  }

  /** which input/output row is under (x, y) in canvas pixels */
  hit(x: number, y: number): Hit | null {
    const s = this.u;
    for (let col = 0; col < this.viz.cols.length; col++) {
      const c = this.viz.cols[col];
      if (!((c.kind === "input" && !c.image) || c.kind === "output")) continue;
      const xc = this.X[col];
      const inBand = c.kind === "input" ? x < xc + 30 * s && x > this.textX - 10 * s : x > xc - 30 * s;
      if (!inBand) continue;
      const ys = this.Y[col];
      for (let slot = 0; slot < c.order.length; slot++) {
        const yy = ys[c.order[slot]];
        const half = ys.length > 1 ? Math.abs(ys[c.order[1]] - ys[c.order[0]]) / 2 : 20 * s;
        if (Math.abs(y - yy) <= Math.max(half, 10 * s)) return { col, slot };
      }
    }
    return null;
  }

  slotAt(col: number, y: number): number {
    const c = this.viz.cols[col];
    const ys = this.Y[col];
    let best = 0;
    let bd = Infinity;
    c.order.forEach((ii, slot) => {
      const d = Math.abs(ys[ii] - y);
      if (d < bd) {
        bd = d;
        best = slot;
      }
    });
    return best;
  }
}

function topK(row: Float32Array, k: number): number[] {
  const idx: number[] = [];
  for (let i = 0; i < row.length; i++) {
    const a = Math.abs(row[i]);
    if (idx.length < k) {
      idx.push(i);
      idx.sort((p, q) => Math.abs(row[q]) - Math.abs(row[p]));
    } else if (a > Math.abs(row[idx[k - 1]])) {
      idx[k - 1] = i;
      idx.sort((p, q) => Math.abs(row[q]) - Math.abs(row[p]));
    }
  }
  return idx;
}

function fmt(v: number): string {
  const a = Math.abs(v);
  const sgn = v < 0 ? "−" : " ";
  if (a >= 1000) return sgn + a.toExponential(1);
  if (a >= 100) return sgn + a.toFixed(0);
  if (a >= 10) return sgn + a.toFixed(1);
  return sgn + a.toFixed(2);
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}
