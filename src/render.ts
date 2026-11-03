// Draws one frame of the glowing network onto a 2D canvas. Pure function of (viz, labels, options, time),
// plus a small easing state so 50 Hz decisions do not strobe at 30 fps.
//
// Reading left to right:
//   input   - one row per input: label, live value, node colour = how fast that input is changing
//   hidden  - one column per layer, brightness = activation (robust-normalised per column over the recording)
//   output  - one node per output with its icon; glow = probability, the chosen (argmax) one is ringed and lit
//   edges   - per target node, the top-k sources by |contribution| this tick; warm = excites, cool = inhibits
//   pulse   - a faint band sweeping left to right once per decision tick

import type { Viz } from "./compute";
import type { NdArray } from "./io/npy";

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
  private obsRange: [number, number] = [0, 1];
  // layout, recomputed in draw
  X: number[] = [];
  Y: number[][] = [];
  r = 10;
  frameRect: [number, number, number, number] | null = null;

  constructor(public viz: Viz, public labels: Labels, public opt: RenderOpts) {
    if (viz.imageInput) {
      const d = opt.obs.data;
      if (opt.obs.dtype === "u1") this.obsRange = [0, 255];
      else {
        let lo = Infinity, hi = -Infinity;
        for (let i = 0; i < d.length; i += 7) {
          const v = Number(d[i]);
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
        this.obsRange = [lo, hi > lo ? hi : lo + 1];
      }
    }
  }

  resetEase() {
    this.ease = null;
    this.lastTick = -1;
  }

  private layout() {
    const { width: W, height: H } = this.opt;
    const s = H / 720;
    const cols = this.viz.cols;
    let left = 0;
    this.frameRect = null;
    const fr = this.opt.frames;
    if (this.opt.showFrame && fr) {
      const [fh, fw] = [fr.shape[1], fr.shape[2]];
      const maxW = W * 0.36;
      let h = H - 80 * s;
      let w = (h * fw) / fh;
      if (w > maxW) {
        w = maxW;
        h = (w * fh) / fw;
      }
      this.frameRect = [28 * s, (H - h) / 2, w, h];
      left = w + 48 * s;
    }
    const labelW = (this.viz.imageInput ? 190 : 230) * s;
    const outW = 210 * s;
    const x0 = left + labelW;
    const x1 = W - outW;
    const nc = cols.length;
    this.X = cols.map((_, i) => x0 + ((x1 - x0) * i) / Math.max(1, nc - 1));
    const top = 86 * s;
    const bot = H - 40 * s;
    const maxN = Math.max(...cols.map((c) => c.sel.length));
    const pitch = Math.min(46 * s, (bot - top) / Math.max(1, maxN - 1));
    this.r = Math.max(3, Math.min(11 * s, pitch * 0.3));
    this.Y = cols.map((c) => {
      const n = c.sel.length;
      const p = c.kind === "hidden" ? pitch : Math.min(pitch * 1.25, (bot - top) / Math.max(1, n - 1), 56 * s);
      const span = p * (n - 1);
      const y0 = (top + bot) / 2 - span / 2;
      const ys = new Array(n);
      c.order.forEach((ii, slot) => (ys[ii] = y0 + slot * p));
      return ys;
    });
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
    const C = { off: hex(th.off), on: hex(th.on), pos: hex(th.pos), neg: hex(th.neg), bg: hex(th.bg), text: hex(th.text), dim: hex(th.dim), pulse: hex(th.pulse) };

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
      ctx.fillStyle = th.bg;
      ctx.fillRect(0, 0, W, H);
    }

    // column headings
    ctx.font = `500 ${15 * s}px "Public Sans", system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = th.dim;
    viz.cols.forEach((c, ci) => ctx.fillText(c.label, this.X[ci] + (c.kind === "output" ? 40 * s : 0), 40 * s));

    // pulse: sweep once per decision tick, or once per m ticks when ticks come faster than ~4 frames
    const ticksPerSec = o.hz * o.speed;
    const m = Math.max(1, Math.ceil((ticksPerSec * 4) / o.fps));
    const ph = (tickF / m) % 1;
    const nc = viz.cols.length;
    for (let ci = 0; ci < nc; ci++) {
      const g = Math.exp(-((ph * (nc + 1) - ci - 0.5) ** 2) / 0.35);
      if (g < 0.05) continue;
      const x = this.X[ci];
      const ys = this.Y[ci];
      const yTop = Math.min(...ys) - 22 * s;
      const yBot = Math.max(...ys) + 22 * s;
      ctx.fillStyle = rgba(C.pulse, 0.06 * g);
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
      const xs = this.X[e.from] + this.r;
      const xd = this.X[e.to] - this.r;
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
          ctx.lineWidth = (1 + 1.6 * st) * s;
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

    // frame (context render from the recording)
    if (this.frameRect && o.frames) {
      const [fx, fy, fw, fh] = this.frameRect;
      const f = o.frames;
      const [, h, w] = f.shape;
      this.frameCanvas.width = w;
      this.frameCanvas.height = h;
      const fc = this.frameCanvas.getContext("2d")!;
      const img = fc.createImageData(w, h);
      const src = f.data as Uint8Array;
      const off = k * h * w * 3;
      for (let p = 0, q = 0; p < h * w; p++, q += 4) {
        img.data[q] = src[off + p * 3];
        img.data[q + 1] = src[off + p * 3 + 1];
        img.data[q + 2] = src[off + p * 3 + 2];
        img.data[q + 3] = 255;
      }
      fc.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(this.frameCanvas, fx, fy, fw, fh);
    }

    // input column
    const inCol = viz.cols[0];
    const xIn = this.X[0];
    const labX = (this.frameRect ? this.frameRect[0] + this.frameRect[2] + 20 * s : 28 * s);
    ctx.textBaseline = "middle";
    inCol.sel.forEach((unit, ii) => {
      const y = this.Y[0][ii];
      ctx.textAlign = "left";
      ctx.fillStyle = th.text;
      ctx.font = `600 ${17 * s}px "Public Sans", system-ui, sans-serif`;
      ctx.fillText(this.labels.inputs[unit] ?? `in ${unit}`, labX, y);
      if (viz.imageInput) this.drawChannel(ctx, k, unit, xIn - this.r - 12 * s, y, s);
      else {
        const raw = inCol.raw![k * inCol.sel.length + ii];
        ctx.textAlign = "right";
        ctx.fillStyle = th.dim;
        ctx.font = `400 ${15 * s}px "JetBrains Mono", ui-monospace, monospace`;
        const u = this.labels.units[unit];
        ctx.fillText(fmt(raw) + (u ? ` ${u}` : ""), xIn - this.r - 12 * s, y);
      }
      this.node(ctx, xIn, y, E[0][ii], C, s, false, 0);
    });

    // hidden columns
    for (let ci = 1; ci < nc - 1; ci++) viz.cols[ci].sel.forEach((_, ii) => this.node(ctx, this.X[ci], this.Y[ci][ii], E[ci][ii], C, s, false, 0));

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
      const isz = 22 * s;
      const icon = this.labels.icons[unit];
      const label = this.labels.outputs[unit] ?? `out ${unit}`;
      ctx.font = `${isChosen ? 700 : 500} ${16 * s}px "Public Sans", system-ui, sans-serif`;
      const tw = ctx.measureText(label).width;
      const pillW = (icon ? isz + 8 * s : 0) + tw + 16 * s;
      if (isChosen) {
        ctx.fillStyle = th.on;
        roundRect(ctx, ix - 6 * s, y - 15 * s, pillW, 30 * s, 8 * s);
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
      if (viz.mode === "continuous" || outCol.sel.length <= 8) {
        ctx.font = `400 ${13 * s}px "JetBrains Mono", ui-monospace, monospace`;
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
      grad.addColorStop(0, rgba(C.on, 0.42 * g));
      grad.addColorStop(1, rgba(C.on, 0));
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, R, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = mix(C.off, C.on, Math.max(0, Math.min(1, v)));
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    if (ring) {
      ctx.strokeStyle = rgba(C.text, 1);
      ctx.lineWidth = 2.5 * s;
      ctx.beginPath();
      ctx.arc(x, y, r + 4.5 * s, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  private drawChannel(ctx: CanvasRenderingContext2D, k: number, c: number, xr: number, y: number, s: number) {
    const im = this.viz.imageInput!;
    const { H, W, C } = im;
    this.thumb.width = W;
    this.thumb.height = H;
    const tc = this.thumb.getContext("2d")!;
    const img = tc.createImageData(W, H);
    const d = this.opt.obs.data;
    const [lo, hi] = this.obsRange;
    const off = (k * C + c) * H * W;
    for (let p = 0; p < H * W; p++) {
      const v = Math.max(0, Math.min(255, ((Number(d[off + p]) - lo) / (hi - lo)) * 255));
      img.data[p * 4] = img.data[p * 4 + 1] = img.data[p * 4 + 2] = v;
      img.data[p * 4 + 3] = 255;
    }
    tc.putImageData(img, 0, 0);
    const h = Math.min(64 * s, (this.Y[0].length > 1 ? Math.abs(this.Y[0][1] - this.Y[0][0]) : 80 * s) - 8 * s);
    const w = (h * W) / H;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.thumb, xr - w, y - h / 2, w, h);
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
