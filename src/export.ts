// Offline export: render every output frame deterministically, then encode.
//   GIF          - gifenc, one shared palette sampled from a few frames (stable colours, no shimmer)
//   WebM / MP4   - WebCodecs through mediabunny; WebM VP9 can keep the alpha channel for compositing
// Nothing leaves the browser: the result is a Blob handed to a download link.

import { GIFEncoder, applyPalette, quantize } from "gifenc";
import { BufferTarget, CanvasSource, Mp4OutputFormat, Output, QUALITY_HIGH, WebMOutputFormat, getFirstEncodableVideoCodec, type VideoCodec } from "mediabunny";
import { Renderer, type Labels, type RenderOpts } from "./render";
import type { Viz } from "./compute";

export type ExportFormat = "gif" | "webm" | "webm-alpha" | "mp4";

export interface ExportJob {
  /** builds what draws a frame (compare mode: both panels); default: one Renderer */
  make?: (opt: RenderOpts) => { draw(ctx: CanvasRenderingContext2D, tRec: number, dt: number): void; width?: number; height?: number };
  viz: Viz;
  labels: Labels;
  opt: RenderOpts; // width/height/fps/speed/transparent are the export's
  format: ExportFormat;
  tStart: number; // recording seconds
  tEnd: number;
  onProgress: (f: number, note: string) => void;
  signal: AbortSignal;
}

export interface ExportResult {
  blob: Blob;
  ext: string;
  frames: number;
  note: string;
}

export async function runExport(job: ExportJob): Promise<ExportResult> {
  const { opt } = job;
  const canvas = document.createElement("canvas");
  const transparent = job.format === "webm-alpha" || (job.format === "gif" && opt.transparent);
  const make = job.make ?? ((o: RenderOpts) => new Renderer(job.viz, job.labels, o));
  const r: NonNullable<ExportJob["make"]> extends (o: RenderOpts) => infer D ? D : never = make({ ...opt, transparent });
  // a stacked comparison is taller than the chosen size
  const W = r.width ?? opt.width;
  const H = r.height ?? opt.height;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d", { alpha: true, willReadFrequently: job.format === "gif" })!;
  const duration = (job.tEnd - job.tStart) / opt.speed;
  const n = Math.max(1, Math.round(duration * opt.fps));
  const frameAt = (i: number) => {
    r.draw(ctx, job.tStart + (i / opt.fps) * opt.speed, i === 0 ? 0 : 1 / opt.fps);
  };
  const yieldUI = () => new Promise((res) => setTimeout(res, 0));
  if (job.format === "gif") {
    // palette from ~12 frames spread over the clip
    const samples: Uint8ClampedArray[] = [];
    const probe = make({ ...opt, transparent });
    for (let q = 0; q < 12; q++) {
      const i = Math.floor((q / 12) * n);
      probe.draw(ctx, job.tStart + (i / opt.fps) * opt.speed, 0);
      samples.push(ctx.getImageData(0, 0, W, H).data);
    }
    const all = new Uint8ClampedArray(samples.reduce((a, s) => a + s.length, 0));
    let off = 0;
    for (const sm of samples) {
      all.set(sm, off);
      off += sm.length;
    }
    const format = transparent ? "rgba4444" : "rgb565";
    const palette = quantize(all, 256, transparent ? { format, oneBitAlpha: true, clearAlpha: true } : { format });
    const tIndex = transparent ? palette.findIndex((c) => c[3] === 0) : -1;
    const gif = GIFEncoder();
    const delay = Math.round(1000 / opt.fps);
    for (let i = 0; i < n; i++) {
      if (job.signal.aborted) throw new DOMException("Export cancelled", "AbortError");
      frameAt(i);
      const data = ctx.getImageData(0, 0, W, H).data;
      const index = applyPalette(data, palette, format);
      gif.writeFrame(index, W, H, { palette: i === 0 ? palette : undefined, delay, transparent: tIndex >= 0, transparentIndex: Math.max(0, tIndex), repeat: 0 });
      if (i % 4 === 0) {
        job.onProgress(i / n, `frame ${i + 1} of ${n}`);
        await yieldUI();
      }
    }
    gif.finish();
    return { blob: new Blob([gif.bytes() as BlobPart], { type: "image/gif" }), ext: "gif", frames: n, note: transparent ? "GIF, 1-bit transparency" : "GIF" };
  }

  const mp4 = job.format === "mp4";
  const candidates: VideoCodec[] = mp4 ? ["avc", "vp9", "av1"] : ["vp9", "vp8", "av1"];
  const codec = await getFirstEncodableVideoCodec(candidates, { width: W, height: H });
  if (!codec) throw new Error(`This browser cannot encode ${mp4 ? "MP4" : "WebM"} video with WebCodecs. Try GIF, or Chrome / Edge.`);
  const output = new Output({ format: mp4 ? new Mp4OutputFormat({ fastStart: "in-memory" }) : new WebMOutputFormat(), target: new BufferTarget() });
  const src = new CanvasSource(canvas, { codec, bitrate: QUALITY_HIGH, alpha: transparent ? "keep" : "discard" });
  output.addVideoTrack(src, { frameRate: opt.fps });
  await output.start();
  for (let i = 0; i < n; i++) {
    if (job.signal.aborted) {
      await output.cancel();
      throw new DOMException("Export cancelled", "AbortError");
    }
    frameAt(i);
    await src.add(i / opt.fps, 1 / opt.fps);
    if (i % 4 === 0) {
      job.onProgress(i / n, `frame ${i + 1} of ${n}`);
      await yieldUI();
    }
  }
  await output.finalize();
  const buf = (output.target as BufferTarget).buffer!;
  return {
    blob: new Blob([buf], { type: mp4 ? "video/mp4" : "video/webm" }),
    ext: mp4 ? "mp4" : "webm",
    frames: n,
    note: `${codec.toUpperCase()}${transparent ? " with alpha" : ""}`,
  };
}
