// How to turn an image-shaped model input [C, H, W] back into pictures a person can look at.
//
// Vocabulary:
//   frame stack      - several frames concatenated on the channel axis (e.g. 2 grayscale frames = 2 channels,
//                      4 RGB frames = 12 channels) so the model can see motion
//   channels/frame   - 1 (grayscale) or 3 (RGB)
//   pixel range      - what the stored numbers mean: 0..255 bytes, 0..1 floats, -1..1, or mean/std normalised
//                      ("stretch": map each channel's 1st..99th percentile to black..white)

import type { NdArray } from "./io/npy";

export type PixelRange = "auto" | "0-255" | "0-1" | "-1-1" | "stretch";
export type ChannelLayout = "auto" | "gray" | "rgb";
export type NewestFrame = "last" | "first";

export interface ImageSettings {
  range: PixelRange;
  layout: ChannelLayout;
  newest: NewestFrame;
}

export interface ImageView {
  C: number;
  H: number;
  W: number;
  /** channels per frame: 1 or 3 */
  cpf: number;
  /** frames in the stack */
  frames: number;
  newest: NewestFrame;
  /** per channel display window */
  lo: Float32Array;
  hi: Float32Array;
  /** what auto detection decided, for the UI */
  describe: string;
}

export function imageView(obs: NdArray, s: ImageSettings): ImageView {
  const [, C, H, W] = obs.shape;
  const cpf = s.layout === "gray" ? 1 : s.layout === "rgb" ? (C % 3 === 0 ? 3 : 1) : C % 3 === 0 ? 3 : 1;
  const frames = C / cpf;
  const d = obs.data;
  const n = d.length;
  let range = s.range;
  let note = "";
  if (range === "auto") {
    if (obs.dtype === "u1") {
      range = "0-255";
      note = "bytes";
    } else {
      let mn = Infinity;
      let mx = -Infinity;
      const step = Math.max(1, Math.floor(n / 400000));
      for (let i = 0; i < n; i += step) {
        const v = Number(d[i]);
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      if (mn >= -1e-3 && mx <= 1.001) range = "0-1";
      else if (mn >= -1.001 && mx <= 1.001) range = "-1-1";
      else if (mn >= 0 && mx <= 255.5) range = "0-255";
      else range = "stretch";
      note = `values ${fmt(mn)} to ${fmt(mx)}`;
    }
  }
  const lo = new Float32Array(C);
  const hi = new Float32Array(C);
  if (range === "stretch") {
    const plane = H * W;
    const T = obs.shape[0];
    for (let c = 0; c < C; c++) {
      const sample: number[] = [];
      const stride = Math.max(1, Math.floor((T * plane) / 50000));
      for (let q = 0; q < T * plane; q += stride) {
        const t = Math.floor(q / plane);
        sample.push(Number(d[(t * C + c) * plane + (q % plane)]));
      }
      sample.sort((a, b) => a - b);
      lo[c] = sample[Math.floor(sample.length * 0.01)];
      hi[c] = sample[Math.floor(sample.length * 0.99)];
      if (hi[c] <= lo[c]) hi[c] = lo[c] + 1;
    }
  } else {
    const [a, b] = range === "0-255" ? [0, 255] : range === "0-1" ? [0, 1] : [-1, 1];
    lo.fill(a);
    hi.fill(b);
  }
  const label = { "0-255": "0 to 255", "0-1": "0 to 1", "-1-1": "−1 to 1", stretch: "stretched per channel", auto: "" }[range];
  const describe = `${frames > 1 ? `${frames} stacked ${cpf === 3 ? "RGB" : "grayscale"} frames` : cpf === 3 ? "RGB" : "grayscale"}, ${label}${s.range === "auto" && note ? ` (auto: ${note})` : ""}`;
  return { C, H, W, cpf, frames, newest: s.newest, lo, hi, describe };
}

/** RGBA pixels of frame `age` (0 = newest) at tick t */
export function frameRGBA(obs: NdArray, v: ImageView, t: number, age: number, out?: Uint8ClampedArray): Uint8ClampedArray {
  const { C, H, W, cpf, frames } = v;
  const plane = H * W;
  const px = out ?? new Uint8ClampedArray(plane * 4);
  const f = v.newest === "last" ? frames - 1 - age : age;
  const c0 = Math.max(0, Math.min(frames - 1, f)) * cpf;
  const d = obs.data;
  const base = t * C * plane;
  for (let p = 0; p < plane; p++) {
    for (let k = 0; k < 3; k++) {
      const c = c0 + (cpf === 3 ? k : 0);
      const x = (Number(d[base + c * plane + p]) - v.lo[c]) / (v.hi[c] - v.lo[c]);
      px[p * 4 + k] = x * 255;
    }
    px[p * 4 + 3] = 255;
  }
  return px;
}

function fmt(x: number) {
  return Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(2);
}
