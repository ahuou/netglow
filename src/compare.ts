// Compare mode: two models run on the same recording, drawn side by side (or stacked) on one canvas, synced on
// one timeline, with a strip along the bottom that marks, tick by tick, whether the two pick the same output.
//
// Vocabulary:
//   panel  - the part of the canvas one model is drawn in; each panel is laid out by its own Renderer, as if it
//            were a canvas of that size, so everything (sizes, look, pictures) works the same as for one model
//   agree  - per tick: both models' argmax is the same output (Choice outputs), or, for Values outputs, how close
//            the two output vectors are (1 - |a - b| / scale)
//   strip  - the agreement row under the panels, with a playhead; it is drawn on the canvas, so exports have it

import { Renderer, type Labels, type RenderOpts } from "./render";
import type { Viz } from "./compute";

export type CompareLayout = "auto" | "side" | "stacked";

export interface Side {
  viz: Viz;
  labels: Labels;
  /** shown above the panel, e.g. "A · trained, 1M steps" */
  name: string;
  /** what differs per model in the render options (its own image view, saliency) */
  opt?: Partial<RenderOpts>;
}

/** per tick agreement in 0..1, and the share of ticks where the choices are the same (null for Values outputs) */
export function agreement(a: Viz, b: Viz): { v: Float32Array; same: number | null } {
  const T = Math.min(a.T, b.T);
  const v = new Float32Array(T);
  if (a.chosen && b.chosen) {
    let n = 0;
    for (let t = 0; t < T; t++) {
      v[t] = a.chosen[t] === b.chosen[t] ? 1 : 0;
      n += v[t];
    }
    return { v, same: n / T };
  }
  const oa = a.cols[a.cols.length - 1], ob = b.cols[b.cols.length - 1];
  const n = Math.min(oa.sel.length, ob.sel.length);
  for (let t = 0; t < T; t++) {
    let d = 0;
    for (let i = 0; i < n; i++) d += Math.abs(oa.raw![t * oa.sel.length + i] - ob.raw![t * ob.sel.length + i]);
    v[t] = Math.max(0, 1 - d / Math.max(1, n));
  }
  return { v, same: null };
}

const SANS = `"Public Sans", system-ui, sans-serif`;

export class CompareRenderer {
  a: Renderer;
  b: Renderer;
  stacked: boolean;
  /** [x, y, w, h] of the two panels and the strip, in canvas pixels */
  rects: [number, number, number, number][] = [];
  strip: [number, number, number, number];
  agree: { v: Float32Array; same: number | null };
  /** writes go to both panels (theme, topK, style, ...), reads come from panel A */
  opt: RenderOpts;

  /** size of the whole composite: side by side = the chosen size; stacked = two full-size panels plus the strip */
  width: number;
  height: number;
  private W: number;
  private H: number;

  constructor(public A: Side, public B: Side, base: RenderOpts, layout: CompareLayout, narrowViewport: boolean) {
    const { width: W, height: H } = base;
    this.W = W;
    this.H = H;
    const s = H / 720;
    this.stacked = layout === "stacked" || (layout === "auto" && (H >= W * 0.8 || narrowViewport));
    const head = Math.round(30 * s);
    const stripH = Math.round(46 * s);
    if (this.stacked) {
      // each model keeps a full panel of the chosen size, so a phone shows each one as large as a single model
      this.rects = [[0, head, W, H], [0, H + 2 * head, W, H]];
      this.strip = [0, 2 * H + 2 * head, W, stripH];
      this.H = 2 * H + 2 * head + stripH + ((2 * head + stripH) % 2); // even, for video encoders
    } else {
      const avail = H - stripH;
      const pw = Math.floor(W / 2);
      this.rects = [[0, head, pw, avail - head], [pw, head, W - pw, avail - head]];
      this.strip = [0, avail, W, stripH];
    }
    this.width = this.W;
    this.height = this.H;
    const mk = (side: Side, r: [number, number, number, number]) => new Renderer(side.viz, side.labels, { ...base, ...side.opt, width: r[2], height: r[3] });
    // the recording's render shows the same episode for both: draw it once, in panel A
    this.a = mk(A, this.rects[0]);
    this.b = mk({ ...B, opt: { ...B.opt, showFrame: false } }, this.rects[1]);
    this.agree = agreement(A.viz, B.viz);
    const a = this.a, b = this.b;
    this.opt = new Proxy(a.opt, {
      set: (_t, k, v) => {
        (a.opt as unknown as Record<string | symbol, unknown>)[k] = v;
        // per-model options stay per model
        if (k !== "saliency" && k !== "image" && k !== "obs" && k !== "showFrame") (b.opt as unknown as Record<string | symbol, unknown>)[k] = v;
        return true;
      },
    });
  }

  get viz() {
    return this.a.viz;
  }

  resetEase() {
    this.a.resetEase();
    this.b.resetEase();
  }

  draw(ctx: CanvasRenderingContext2D, tRec: number, dt: number) {
    const o = this.a.opt;
    const th = o.theme;
    const st = o.style;
    const W = this.W;
    const Hc = this.H;
    const s = Hc / 720;
    ctx.save();
    ctx.clearRect(0, 0, W, Hc);
    if (!o.transparent) {
      ctx.fillStyle = st.bg || th.bg;
      ctx.fillRect(0, 0, W, Hc);
    }
    const sides: [Renderer, Side][] = [[this.a, this.A], [this.b, this.B]];
    sides.forEach(([r, side], i) => {
      const [x, y, w, h] = this.rects[i];
      ctx.save();
      ctx.translate(x, y);
      ctx.beginPath();
      ctx.rect(0, 0, w, h);
      ctx.clip();
      // one common scale: each panel may shrink no less than the other (A also reserves the recording's column)
      const other = i === 0 ? this.b : this.a;
      r.opt.fitCap = other.natFit;
      r.draw(ctx, tRec, dt);
      ctx.restore();
      // panel name, above the panel
      ctx.font = `600 ${15 * s}px ${SANS}`;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillStyle = th.text;
      const pad = Math.max(12 * s, st.padding * s * 0.6);
      ctx.fillText(side.name, x + pad, y - 15 * s);
      // a hairline between the panels
      if (i === 1) {
        ctx.fillStyle = th.off;
        if (this.stacked) ctx.fillRect(pad, y - 31 * s, W - 2 * pad, Math.max(1, s));
        else ctx.fillRect(x, y - 20 * s, Math.max(1, s), h + 20 * s);
      }
    });
    this.drawStrip(ctx, tRec, s, W);
    ctx.restore();
  }

  /** agreement strip: one cell per tick, lit where the two choose the same output; a playhead at the current tick */
  private drawStrip(ctx: CanvasRenderingContext2D, tRec: number, s: number, W: number) {
    const o = this.a.opt;
    const th = o.theme;
    const [, y0, , h] = this.strip;
    const pad = Math.max(12 * s, o.style.padding * s * 0.6);
    const label = this.agree.same !== null ? `Choices agree ${Math.round(this.agree.same * 100)}%` : "Outputs agree";
    ctx.font = `500 ${13 * s}px ${SANS}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillStyle = th.dim;
    const lw = ctx.measureText(label).width;
    const yc = y0 + h / 2;
    ctx.fillText(label, pad, yc);
    const x0 = pad + lw + 14 * s;
    const x1 = W - pad;
    const bh = 12 * s;
    const v = this.agree.v;
    const T = v.length;
    const cw = (x1 - x0) / T;
    // disagreeing ticks in the same red as the timeline's mismatch marks (the edge colours mean excite / inhibit)
    const on = hexRGB(th.on), neg = hexRGB("#e5566f"), off = hexRGB(th.off);
    for (let t = 0; t < T; t++) {
      const a = v[t];
      const c = this.agree.same !== null ? (a ? off : neg) : mixRGB(neg, off, a);
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      ctx.fillRect(x0 + t * cw, yc - bh / 2, Math.max(cw, 1) + 0.5, bh);
    }
    const k = Math.min(T - 1, Math.floor(tRec * o.hz));
    ctx.fillStyle = `rgb(${on[0]},${on[1]},${on[2]})`;
    ctx.fillRect(x0 + (k + 0.5) * cw - Math.max(1, 1.5 * s), yc - bh / 2 - 5 * s, Math.max(2, 3 * s), bh + 10 * s);
  }

  /** canvas point -> panel A's row under it (drag works in panel A) */
  hit(x: number, y: number) {
    const [rx, ry, rw, rh] = this.rects[0];
    if (x < rx || y < ry || x > rx + rw || y > ry + rh) return null;
    return this.a.hit(x - rx, y - ry);
  }

  slotAt(col: number, y: number) {
    return this.a.slotAt(col, y - this.rects[0][1]);
  }
}

type RGB = [number, number, number];
const hexRGB = (h: string): RGB => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
const mixRGB = (a: RGB, b: RGB, t: number): RGB => [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * t)) as RGB;
