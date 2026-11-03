// Draws one frame of the glowing network onto a 2D canvas. Pure function of (viz, labels, options, time),
// plus a small easing state so 50 Hz decisions do not strobe at 30 fps.
//
// Reading left to right:
//   input   - one row per input: label, live value, node colour = how fast that input is changing
//   hidden  - one column per layer, brightness = activation (robust-normalised per column over the recording)
//   output  - one node per output with its icon; glow = probability, the chosen (argmax) one is ringed and lit
//   edges   - per target node, the top-k sources by |contribution| this tick; warm = excites, cool = inhibits
//   pulse   - a faint band sweeping left to right once per decision tick
// Image models: the input column is one image node (the current frame, pixels kept visible, border lit by how
// much the frame changed), optionally a large "What the AI sees" view above or beside the network, and conv
// channels can be drawn as their live feature maps instead of dots.

import type { Viz } from "./compute";
import type { NdArray } from "./io/npy";
import { frameRGBA, type ImageView } from "./image";

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
}

export const DEFAULT_STYLE: Style = {
  nodeScale: 1, edgeMin: 1, edgeMax: 2.6, labelSize: 17, valueSize: 15, mapScale: 1, imageScale: 1,
  aiPos: "above", aiScale: 1, recPos: "auto", recScale: 1, colSpacing: 1, rowSpacing: 1, padding: 28, titles: true,
  glow: 1, pulse: true, pulseIntensity: 1, pos: "", neg: "", bg: "", shape: "circle", showValues: true, showProbs: true,
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
}

interface Pic {
  kind: "rec" | "ai";
  x: number;
  y: number;
  w: number;
  h: number;
  caption: string;
  /** corner insets are drawn on top of the network */
  corner: boolean;
}

type RGB = [number, number, number];
const hex = (h: string): RGB => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
const mix = (a: RGB, b: RGB, t: number) => `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)},${Math.round(a[1] + (b[1] - a[1]) * t)},${Math.round(a[2] + (b[2] - a[2]) * t)})`;
const rgba = (c: RGB, a: number) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

export interface Hit {
  col: number;
  slot: number;
}

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
  private textX = 28;

  constructor(public viz: Viz, public labels: Labels, public opt: RenderOpts) {}

  resetEase() {
    this.ease = null;
    this.lastTick = -1;
  }

  private layout() {
    const { width: W, height: H } = this.opt;
    const s = H / 720;
    const cols = this.viz.cols;
    const o = this.opt;
    const st = o.style;
    const cap = 30 * s;
    const m = st.padding * s;
    // pictures: the recording's render and / or what the AI sees, each placed by its own position
    const want: { kind: Pic["kind"]; w0: number; h0: number; caption: string; pos: PicPos; scale: number }[] = [];
    const aiOn = !!o.image && st.aiPos !== "off";
    if (o.showFrame && o.frames && st.recPos !== "off") {
      const pos = st.recPos === "auto" ? (aiOn && st.aiPos === "above" ? "above" : "left") : st.recPos;
      want.push({ kind: "rec", w0: o.frames.shape[2], h0: o.frames.shape[1], caption: "Recording", pos, scale: st.recScale });
    }
    if (aiOn) want.push({ kind: "ai", w0: o.image!.W, h0: o.image!.H, caption: `What the AI sees (${o.image!.W}×${o.image!.H})`, pos: st.aiPos, scale: st.aiScale });
    let left = 0;
    let right = 0;
    let topBand = 0;
    this.pics = [];
    const above = want.filter((p) => p.pos === "above");
    if (above.length) {
      const gap = 40 * s;
      const base = H * 0.42 - cap - 14 * s;
      let hs = above.map((p) => base * p.scale);
      let ws = above.map((p, i) => (hs[i] * p.w0) / p.h0);
      const total = ws.reduce((a, b) => a + b, 0) + gap * (above.length - 1);
      const k = Math.min(1, (W - 2 * m) / total);
      hs = hs.map((h) => h * k);
      ws = ws.map((w) => w * k);
      let x = (W - (ws.reduce((a, b) => a + b, 0) + gap * (above.length - 1))) / 2;
      const hMax = Math.max(...hs);
      above.forEach((p, i) => {
        this.pics.push({ kind: p.kind, x, y: 14 * s + cap + (hMax - hs[i]), w: ws[i], h: hs[i], caption: p.caption, corner: false });
        x += ws[i] + gap;
      });
      topBand = 14 * s + cap + hMax;
    }
    for (const side of ["left", "right"] as const) {
      const group = want.filter((p) => p.pos === side);
      if (!group.length) continue;
      const gap = 20 * s;
      const each = Math.min((H - 2 * m - group.length * cap - gap * (group.length - 1)) / group.length, H * 0.62);
      const sizes = group.map((p) => {
        const maxW = W * 0.3 * p.scale;
        let h = each * p.scale;
        let w = (h * p.w0) / p.h0;
        if (w > maxW) {
          w = maxW;
          h = (w * p.h0) / p.w0;
        }
        return [w, h];
      });
      let y = (H - sizes.reduce((a, [, h]) => a + h + cap, 0) - gap * (group.length - 1)) / 2;
      const wMax = Math.max(...sizes.map(([w]) => w));
      group.forEach((p, i) => {
        const [w, h] = sizes[i];
        this.pics.push({ kind: p.kind, x: side === "left" ? m : W - m - w, y: y + cap, w, h, caption: p.caption, corner: false });
        y += cap + h + gap;
      });
      if (side === "left") left = m + wMax + 12 * s;
      else right = wMax + 12 * s;
    }
    for (const p of want.filter((q) => q.pos.length === 2)) {
      let h = H * 0.28 * p.scale;
      let w = (h * p.w0) / p.h0;
      if (w > W * 0.35 * p.scale) {
        w = W * 0.35 * p.scale;
        h = (w * p.h0) / p.w0;
      }
      const x = p.pos[1] === "l" ? m : W - m - w;
      const y = p.pos[0] === "t" ? m + cap : H - m - h;
      this.pics.push({ kind: p.kind, x, y, w, h, caption: p.caption, corner: true });
    }
    this.headerY = topBand + m + 12 * s;
    const top = topBand + m + 58 * s;
    const bot = H - m - 12 * s;
    // image node size (image models): the frame at a size where its pixels are visible
    let labelW = 200 * s;
    this.imgRect = null;
    if (this.viz.imageInput && o.image) {
      const th = Math.min(124 * s, (bot - top) * 0.5) * st.imageScale;
      const tw = Math.min(160 * s * st.imageScale, (th * o.image.W) / o.image.H);
      labelW = tw + 96 * s;
      this.imgRect = [0, 0, tw, (tw * o.image.H) / o.image.W];
    }
    const outW = 196 * s + m + right;
    this.textX = left + (left ? 0 : m);
    let x0 = this.textX + labelW;
    let x1 = W - outW;
    if (st.colSpacing !== 1) {
      const c = (x0 + x1) / 2;
      const hw = ((x1 - x0) / 2) * st.colSpacing;
      x0 = c - hw;
      x1 = c + hw;
    }
    const nc = cols.length;
    this.X = cols.map((_, i) => x0 + ((x1 - x0) * i) / Math.max(1, nc - 1));
    const maxN = Math.max(...cols.map((c) => c.sel.length));
    const pitch = Math.min(46 * s, (bot - top) / Math.max(1, maxN - 1));
    this.r = Math.max(3, Math.min(11 * s, pitch * 0.3)) * st.nodeScale;
    this.mapSize = Math.max(6, Math.min(pitch * 0.78, 38 * s)) * st.mapScale;
    this.half = cols.map((c, ci) => (ci === 0 && this.imgRect ? 2 * s : o.showMaps && c.maps ? this.mapSize / 2 : this.r));
    this.Y = cols.map((c) => {
      const n = c.sel.length;
      const p = (c.kind === "hidden" ? pitch : Math.min(pitch * 1.25, (bot - top) / Math.max(1, n - 1), 56 * s)) * st.rowSpacing;
      const span = p * (n - 1);
      const y0 = (top + bot) / 2 - span / 2;
      const ys = new Array(n);
      c.order.forEach((ii, slot) => (ys[ii] = y0 + slot * p));
      return ys;
    });
    if (this.imgRect) {
      const [, , tw, th] = this.imgRect;
      this.imgRect = [this.X[0] - tw, this.Y[0][0] - th / 2, tw, th];
    }
    return s;
  }

  /** draw recording time tRec (seconds); dt = seconds of wall/video time since the previous frame (0 = snap) */
  draw(ctx: CanvasRenderingContext2D, tRec: number, dt: number) {
    const o = this.opt;
    const th = o.theme;
    const viz = this.viz;
    const T = viz.T;
    const tickF = Math.max(0, tRec * o.hz);
    const k = Math.min(T - 1, Math.floor(tickF));
    const s = this.layout();
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

    ctx.save();
    ctx.clearRect(0, 0, W, H);
    if (!o.transparent) {
      ctx.fillStyle = st.bg || th.bg;
      ctx.fillRect(0, 0, W, H);
    }

    // column headings
    ctx.font = `500 ${15 * s}px "Public Sans", system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = th.dim;
    if (st.titles) viz.cols.forEach((c, ci) => ctx.fillText(c.label, this.X[ci] + (c.kind === "output" ? 40 * s : ci === 0 && this.imgRect ? -this.imgRect[2] / 2 : 0), this.headerY));

    // pulse: sweep once per decision tick, or once per m ticks when ticks come faster than ~4 frames
    const ticksPerSec = o.hz * o.speed;
    const m = Math.max(1, Math.ceil((ticksPerSec * 4) / o.fps));
    const ph = (tickF / m) % 1;
    const nc = viz.cols.length;
    for (let ci = this.imgRect ? 1 : 0; ci < nc; ci++) {
      const g = Math.exp(-((ph * (nc + 1) - ci - 0.5) ** 2) / 0.35);
      if (g < 0.05) continue;
      const x = this.X[ci];
      const ys = this.Y[ci];
      const yTop = Math.min(...ys) - 22 * s;
      const yBot = Math.max(...ys) + 22 * s;
      if (!st.pulse) break;
      ctx.fillStyle = rgba(C.pulse, 0.06 * g * st.pulseIntensity);
      roundRect(ctx, x - 20 * s, yTop, 40 * s, yBot - yTop, 14 * s);
      ctx.fill();
    }

    // edges
    ctx.lineCap = "round";
    for (const e of viz.edges) {
      const from = viz.cols[e.from];
      const to = viz.cols[e.to];
      const nf = from.sel.length;
      const nt = to.sel.length;
      const xs = this.X[e.from] + this.half[e.from];
      const xd = this.X[e.to] - this.half[e.to];
      const kk = Math.min(o.topK, nf);
      const base = k * nt * nf;
      for (let j = 0; j < nt; j++) {
        const row = e.c.subarray(base + j * nf, base + (j + 1) * nf);
        const idx = topK(row, kk);
        for (const i of idx) {
          const v = row[i];
          const st = Math.min(Math.abs(v) / e.scale, 1);
          if (st < 0.06) continue;
          ctx.strokeStyle = rgba(v > 0 ? C.pos : C.neg, 0.12 + 0.68 * st);
          ctx.lineWidth = (o.style.edgeMin + (o.style.edgeMax - o.style.edgeMin) * st) * s;
          ctx.beginPath();
          const y0 = this.Y[e.from][i];
          const y1 = this.Y[e.to][j];
          const mx = (xs + xd) / 2;
          ctx.moveTo(xs, y0);
          ctx.bezierCurveTo(mx, y0, mx, y1, xd, y1);
          ctx.stroke();
        }
      }
    }

    // pictures: recording render and what the AI sees, each with a caption
    const drawPics = (corner: boolean) => {
     for (const p of this.pics) {
      if (p.corner !== corner) continue;
      ctx.font = `500 ${15 * s}px "Public Sans", system-ui, sans-serif`;
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      ctx.fillStyle = th.dim;
      ctx.fillText(p.caption, p.x, p.y - 10 * s);
      if (p.kind === "rec") this.blitRecording(ctx, k, p.x, p.y, p.w, p.h);
      else this.blitInput(ctx, k, 0, p.x, p.y, p.w, p.h);
     }
    };
    drawPics(false);
    ctx.textBaseline = "middle";

    // input column
    const inCol = viz.cols[0];
    const xIn = this.X[0];
    const labX = this.textX;
    ctx.textBaseline = "middle";
    if (this.imgRect) this.drawImageNode(ctx, k, E[0][0], C, s);
    else inCol.sel.forEach((unit, ii) => {
      const y = this.Y[0][ii];
      ctx.textAlign = "left";
      ctx.fillStyle = th.text;
      ctx.font = `600 ${st.labelSize * s}px "Public Sans", system-ui, sans-serif`;
      ctx.fillText(this.labels.inputs[unit] ?? `in ${unit}`, labX, y);
      if (st.showValues) {
        const raw = inCol.raw![k * inCol.sel.length + ii];
        ctx.textAlign = "right";
        ctx.fillStyle = th.dim;
        ctx.font = `400 ${st.valueSize * s}px "JetBrains Mono", ui-monospace, monospace`;
        const u = this.labels.units[unit];
        ctx.fillText(fmt(raw) + (u ? ` ${u}` : ""), xIn - this.r - 12 * s, y);
      }
      this.node(ctx, xIn, y, E[0][ii], C, s, false, 0);
    });

    // hidden columns
    for (let ci = 1; ci < nc - 1; ci++) {
      const col = viz.cols[ci];
      const maps = o.showMaps ? col.maps : undefined;
      col.sel.forEach((_, ii) => (maps ? this.mapNode(ctx, k, col, ii, this.X[ci], this.Y[ci][ii], E[ci][ii], C, s) : this.node(ctx, this.X[ci], this.Y[ci][ii], E[ci][ii], C, s, false, 0)));
    }

    // outputs
    const outCol = viz.cols[nc - 1];
    const xo = this.X[nc - 1];
    const ch = viz.chosen ? viz.chosen[k] : -1;
    outCol.sel.forEach((unit, ii) => {
      const y = this.Y[nc - 1][ii];
      const p = E[nc - 1][ii];
      const isChosen = unit === ch;
      this.node(ctx, xo, y, p, C, s, isChosen, p);
      const ix = xo + this.r + 16 * s;
      const isz = 22 * s * ((st.labelSize - 1) / 16);
      const icon = this.labels.icons[unit];
      const label = this.labels.outputs[unit] ?? `out ${unit}`;
      const ls = (st.labelSize - 1) / 16;
      ctx.font = `${isChosen ? 700 : 500} ${(st.labelSize - 1) * s}px "Public Sans", system-ui, sans-serif`;
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
        ctx.font = `400 ${(st.valueSize - 2) * s}px "JetBrains Mono", ui-monospace, monospace`;
        ctx.fillStyle = th.dim;
        const val = viz.mode === "categorical" ? `${Math.round(outCol.raw![k * outCol.sel.length + ii] * 100)}%` : fmt(outCol.raw![k * outCol.sel.length + ii]);
        ctx.fillText(val, ix + pillW + 4 * s, y + 0.5 * s);
      }
    });
    drawPics(true);
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
  private drawImageNode(ctx: CanvasRenderingContext2D, k: number, v: number, C: Record<string, RGB>, s: number) {
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
    ctx.textAlign = "right";
    ctx.fillStyle = this.opt.theme.text;
    ctx.font = `600 ${this.opt.style.labelSize * s}px "Public Sans", system-ui, sans-serif`;
    ctx.fillText(this.labels.inputs[0] ?? "image", x - b - 14 * s, y + h / 2);
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
    const nc = this.viz.cols.length;
    const s = this.opt.height / 720;
    for (const col of [0, nc - 1]) {
      const xc = this.X[col];
      const inBand = col === 0 ? x < xc + 30 * s : x > xc - 30 * s;
      if (!inBand) continue;
      const ys = this.Y[col];
      const c = this.viz.cols[col];
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
