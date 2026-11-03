// App wiring: files in -> plan -> onnxruntime trace -> viz -> canvas; settings, timeline, drag, export.

import "@fontsource/public-sans/400.css";
import "@fontsource/public-sans/500.css";
import "@fontsource/public-sans/600.css";
import "@fontsource/public-sans/700.css";
import "@fontsource/jetbrains-mono/400.css";
import "./styles.css";

import { parseModel, type ModelInfo } from "./onnx/proto";
import { buildPlan, type Plan } from "./analyze";
import { runTrace, type Trace } from "./run";
import { buildViz, orderHidden, type OutputMode, type Viz } from "./compute";
import { parseJsonRecording, parseNpy, parseNpz, toFloat, type NdArray } from "./io/npy";
import { Renderer, THEMES, type Labels, type Theme } from "./render";
import { runExport, type ExportFormat } from "./export";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

interface LabelsFile {
  title?: string;
  description?: string;
  inputs?: string[];
  outputs?: string[];
  icons?: (string | null)[];
  output_mode?: OutputMode;
  input_units?: (string | null)[] | null;
}

const DEMOS = [
  { id: "lunar-mlp", title: "LunarLander, MLP", sub: "PPO policy reading 8 numbers" },
  { id: "lunar-cnn", title: "LunarLander, CNN", sub: "the same pilot, flying from pixels" },
];

const S = {
  model: null as ModelInfo | null,
  modelName: "",
  plan: null as Plan | null,
  rec: null as Map<string, NdArray> | null,
  recName: "",
  obs: null as NdArray | null,
  labelsFile: null as LabelsFile | null,
  iconFiles: new Map<string, HTMLImageElement>(), // by lowercased basename without extension
  iconOverride: new Map<number, HTMLImageElement>(),
  nameOverride: { inputs: new Map<number, string>(), outputs: new Map<number, string>() },
  trace: null as Trace | null,
  viz: null as Viz | null,
  renderer: null as Renderer | null,
  labels: null as Labels | null,
  t: 0, // recording seconds
  playing: false,
  theme: THEMES[0] as Theme,
  mode: "categorical" as OutputMode,
  modeTouched: false,
  hz: 30,
  speed: 1,
  topK: 3,
  nodes: 10,
  showFrame: false,
  transparent: false,
  exportAbort: null as AbortController | null,
  lastUrl: "",
};

const view = $<HTMLCanvasElement>("view");
const ctx = view.getContext("2d")!;

// ---------- status ----------
function status(msg: string, error = false) {
  const el = $("status");
  el.textContent = msg;
  el.classList.toggle("error", error);
}
function busy(on: boolean, text = "", f = 0) {
  $("busy").hidden = !on;
  $("busyText").textContent = text;
  $("busyFill").style.width = `${Math.round(f * 100)}%`;
}

// ---------- loading ----------
async function loadModelBytes(bytes: Uint8Array, name: string) {
  S.model = null;
  S.plan = null;
  S.modelName = name;
  markFile("model", name);
  const m = parseModel(bytes);
  S.model = m;
  S.plan = buildPlan(m); // throws UnsupportedGraph with a clear message
  S.modeTouched = false;
}

async function loadRecording(bytes: Uint8Array, name: string) {
  let rec: Map<string, NdArray>;
  if (name.endsWith(".npz") || (bytes[0] === 0x50 && bytes[1] === 0x4b)) rec = await parseNpz(bytes);
  else if (name.endsWith(".npy")) rec = new Map([["obs", parseNpy(bytes)]]);
  else rec = parseJsonRecording(new TextDecoder().decode(bytes));
  const obs = rec.get("obs") ?? rec.get("inputs") ?? (S.plan ? rec.get(S.plan.inputName) : undefined);
  if (!obs) throw new Error(`The recording needs an "obs" array (one row per tick). Found: ${[...rec.keys()].join(", ") || "nothing"}.`);
  S.rec = rec;
  S.obs = obs;
  S.recName = name;
  markFile("recording", name);
  const fps = rec.get("fps");
  const t = rec.get("t");
  if (fps && fps.data.length) S.hz = Number(fps.data[0]);
  else if (t && t.shape[0] > 2) {
    const tf = toFloat(t);
    const d = (tf[tf.length - 1] - tf[0]) / (tf.length - 1);
    if (d > 0) S.hz = 1 / d;
  }
  ($("hz") as HTMLInputElement).value = String(+S.hz.toFixed(3));
  S.t = 0;
}

function markFile(kind: string, name: string) {
  const row = document.querySelector(`.file-row[data-kind="${kind}"]`)!;
  row.classList.add("loaded");
  const id = kind === "model" ? "nameModel" : kind === "recording" ? "nameRec" : "nameLabels";
  $(id).textContent = name;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error(`Could not load icon ${src}`));
    im.src = src;
  });
}

async function handleFiles(files: File[]) {
  const by = (pred: (f: File) => boolean) => files.filter(pred);
  const ext = (f: File) => f.name.toLowerCase().split(".").pop() ?? "";
  try {
    const models = by((f) => ext(f) === "onnx");
    const icons = by((f) => ["png", "svg", "jpg", "jpeg", "webp"].includes(ext(f)));
    const jsons = by((f) => ext(f) === "json");
    const recs = by((f) => ["npz", "npy"].includes(ext(f)));
    for (const f of icons) {
      const url = URL.createObjectURL(f);
      S.iconFiles.set(f.name.toLowerCase().replace(/\.[^.]+$/, ""), await loadImage(url));
    }
    if (icons.length) markFile("labels", `${icons.length} icon${icons.length > 1 ? "s" : ""}${S.labelsFile ? " + labels.json" : ""}`);
    for (const f of jsons) {
      const text = await f.text();
      const j = JSON.parse(text);
      if (Array.isArray(j.obs) || Array.isArray(j.inputs?.[0])) recs.push(f);
      else {
        S.labelsFile = j;
        markFile("labels", f.name + (icons.length ? ` + ${icons.length} icons` : ""));
      }
    }
    if (models[0]) await loadModelBytes(new Uint8Array(await models[0].arrayBuffer()), models[0].name);
    if (recs[0]) await loadRecording(new Uint8Array(await recs[0].arrayBuffer()), recs[0].name);
    await rebuild(true);
  } catch (e) {
    fail(e);
  }
}

async function loadDemo(id: string) {
  const base = `demos/${id}/`;
  try {
    stop();
    status(`Loading demo ${id}`);
    busy(true, "Downloading the demo", 0.1);
    const get = async (p: string) => {
      const r = await fetch(base + p);
      if (!r.ok) throw new Error(`Demo file ${base + p} is missing (${r.status}). Run python/train_demos.py to build the demos.`);
      return new Uint8Array(await r.arrayBuffer());
    };
    const [m, rec, lab] = await Promise.all([get("model.onnx"), get("recording.npz"), get("labels.json")]);
    S.labelsFile = JSON.parse(new TextDecoder().decode(lab));
    S.iconFiles.clear();
    S.iconOverride.clear();
    S.nameOverride.inputs.clear();
    S.nameOverride.outputs.clear();
    const icons = S.labelsFile?.icons ?? [];
    await Promise.all(
      icons.map(async (p, i) => {
        if (!p) return;
        S.iconOverride.set(i, await loadImage(base + p));
      }),
    );
    markFile("labels", "labels.json + icons");
    await loadModelBytes(m, `${id}/model.onnx`);
    await loadRecording(rec, `${id}/recording.npz`);
    S.showFrame = !!S.rec?.get("frames");
    ($("showFrame") as HTMLInputElement).checked = S.showFrame;
    await rebuild(true);
  } catch (e) {
    fail(e);
  }
}

function fail(e: unknown) {
  busy(false);
  console.error(e);
  status(e instanceof Error ? e.message : String(e), true);
}

// ---------- pipeline ----------
async function rebuild(rerun: boolean) {
  if (!S.model || !S.plan) {
    if (S.rec) status("Recording loaded. Now drop the ONNX model.");
    return;
  }
  if (!S.rec || !S.obs) {
    status(`Model loaded: ${S.plan.layers.length} weighted layers. Now drop a recording.`);
    $("docTitle").textContent = S.labelsFile?.title ?? S.modelName;
    return;
  }
  if (rerun || !S.trace) {
    if (!S.modeTouched) {
      const lf = S.labelsFile?.output_mode;
      const acts = S.rec.get("actions");
      const nOut = S.plan.layers[S.plan.layers.length - 1].out;
      S.mode = lf ?? (S.plan.softmaxInGraph ? "categorical" : acts && acts.dtype.startsWith("f") && acts.shape.length > 1 ? "continuous" : nOut > 1 ? "categorical" : "continuous");
      ($("mode") as HTMLSelectElement).value = S.mode;
    }
    busy(true, "Running the model in this tab", 0);
    S.trace = await runTrace(S.model, S.plan, S.obs, (f) => busy(true, "Running the model in this tab", f));
    busy(false);
  }
  const prev = S.viz ? { input: S.viz.cols[0].order, output: S.viz.cols[S.viz.cols.length - 1].order } : undefined;
  S.viz = buildViz(S.plan, S.trace, S.rec, S.obs, { nodesPerColumn: S.nodes, maxInputs: 24, mode: S.mode }, rerun ? undefined : prev);
  S.labels = resolveLabels();
  makeRenderer();
  const warn = S.plan.warnings.length ? ` Note: ${S.plan.warnings.join(" ")}` : "";
  status(`Ran ${S.trace.T} ticks in ${Math.round(S.trace.ms)} ms. ${S.plan.layers.length} weighted layers.${warn}`);
  if (rerun) {
    const dur = (S.viz.T - 1) / S.hz;
    ($("from") as HTMLInputElement).value = "0";
    ($("to") as HTMLInputElement).value = dur.toFixed(2);
  }
  renderLabelsPanel();
  renderMatch();
  $("empty").hidden = true;
  $("docTitle").textContent = S.labelsFile?.title ?? S.modelName.replace(/\.onnx$/, "");
  for (const id of ["play", "scrub", "exportBtn", "exportTop"]) ($(id) as HTMLButtonElement).disabled = false;
  const sc = $("scrub") as HTMLInputElement;
  sc.max = String(S.viz.T - 1);
  $("frameSection").hidden = !S.rec.get("frames");
  drawRibbon();
  redraw(0);
}

function resolveLabels(): Labels {
  const lf = S.labelsFile ?? {};
  const rec = S.rec!;
  const viz = S.viz!;
  const nIn = viz.cols[0].n;
  const nOut = viz.cols[viz.cols.length - 1].n;
  const recIn = rec.get("input_names")?.strings;
  const recOut = rec.get("output_names")?.strings;
  const inputs = Array.from({ length: nIn }, (_, i) => S.nameOverride.inputs.get(i) ?? lf.inputs?.[i] ?? recIn?.[i] ?? (viz.imageInput ? `channel ${i}` : `in ${i}`));
  const outputs = Array.from({ length: nOut }, (_, i) => S.nameOverride.outputs.get(i) ?? lf.outputs?.[i] ?? recOut?.[i] ?? `out ${i}`);
  const icons = outputs.map((name, i) => {
    const o = S.iconOverride.get(i);
    if (o) return o;
    const keys = [String(i), `out_${i}`, `output_${i}`, name.toLowerCase(), name.toLowerCase().replace(/\s+/g, "_"), name.toLowerCase().replace(/\s+/g, "-")];
    for (const k of keys) if (S.iconFiles.has(k)) return S.iconFiles.get(k)!;
    const lp = lf.icons?.[i];
    if (lp) {
      const base = lp.toLowerCase().split("/").pop()!.replace(/\.[^.]+$/, "");
      if (S.iconFiles.has(base)) return S.iconFiles.get(base)!;
    }
    return null;
  });
  return { title: lf.title ?? S.modelName, inputs, outputs, icons, units: Array.from({ length: nIn }, (_, i) => lf.input_units?.[i] ?? null) };
}

function exportSize(): [number, number] {
  const [w, h] = ($("size") as HTMLSelectElement).value.split("x").map(Number);
  return [w, h];
}

function renderOpts() {
  const [w, h] = exportSize();
  return {
    width: w,
    height: h,
    theme: S.theme,
    transparent: S.transparent,
    topK: S.topK,
    showFrame: S.showFrame,
    frames: S.rec?.get("frames") ?? null,
    obs: S.obs!,
    hz: S.hz,
    speed: S.speed,
    fps: Number(($("fps") as HTMLSelectElement).value),
  };
}

function makeRenderer() {
  if (!S.viz || !S.labels) return;
  const o = renderOpts();
  view.width = o.width;
  view.height = o.height;
  S.renderer = new Renderer(S.viz, S.labels, o);
  $("well").classList.toggle("checker", S.transparent);
}

// ---------- drawing / playback ----------
let lastFrame = 0;
function redraw(dt: number) {
  const r = S.renderer;
  if (!r || !S.viz) return;
  r.draw(ctx, S.t, dt);
  const k = Math.min(S.viz.T - 1, Math.floor(S.t * S.hz));
  ($("scrub") as HTMLInputElement).value = String(k);
  $("clock").textContent = `${S.t.toFixed(2)} s · tick ${k} / ${S.viz.T - 1}`;
  drawFramePreview(k);
}

function loop(ts: number) {
  if (!S.playing) return;
  const dt = lastFrame ? Math.min(0.1, (ts - lastFrame) / 1000) : 0;
  lastFrame = ts;
  S.t += dt * S.speed;
  const dur = (S.viz!.T - 1) / S.hz;
  if (S.t > dur) {
    S.t = 0;
    S.renderer?.resetEase();
  }
  redraw(dt);
  requestAnimationFrame(loop);
}
function play() {
  if (!S.viz) return;
  S.playing = true;
  lastFrame = 0;
  $("play").setAttribute("aria-label", "Pause");
  $("playIcon").setAttribute("d", "M4 2.5h3v11H4zM9 2.5h3v11H9z");
  requestAnimationFrame(loop);
}
function stop() {
  S.playing = false;
  $("play").setAttribute("aria-label", "Play");
  $("playIcon").setAttribute("d", "M4 2.5v11l9-5.5z");
}

const frameCanvas = $<HTMLCanvasElement>("framePreview");
function drawFramePreview(k: number) {
  const f = S.rec?.get("frames");
  if (!f || $("frameSection").hidden) return;
  const [, h, w] = f.shape;
  if (frameCanvas.width !== w) {
    frameCanvas.width = w;
    frameCanvas.height = h;
  }
  const c = frameCanvas.getContext("2d")!;
  const img = c.createImageData(w, h);
  const src = f.data as Uint8Array;
  const off = k * h * w * 3;
  for (let p = 0; p < h * w; p++) {
    img.data[p * 4] = src[off + p * 3];
    img.data[p * 4 + 1] = src[off + p * 3 + 1];
    img.data[p * 4 + 2] = src[off + p * 3 + 2];
    img.data[p * 4 + 3] = 255;
  }
  c.putImageData(img, 0, 0);
}

/** decision ribbon: one row per output, brightness = probability; mismatches against the log marked below */
function drawRibbon() {
  const viz = S.viz;
  const cv = $<HTMLCanvasElement>("ribbon");
  if (!viz) return;
  const w = Math.max(200, cv.clientWidth * devicePixelRatio);
  const h = 40 * devicePixelRatio;
  cv.width = w;
  cv.height = h;
  const c = cv.getContext("2d")!;
  c.fillStyle = "#0b0b0d";
  c.fillRect(0, 0, w, h);
  const out = viz.cols[viz.cols.length - 1];
  const n = out.sel.length;
  const misH = viz.match.mismatch ? 4 * devicePixelRatio : 0;
  const rowH = (h - misH) / n;
  const on = S.theme.on;
  for (let x = 0; x < w; x++) {
    const t = Math.min(viz.T - 1, Math.floor((x / w) * viz.T));
    for (let slot = 0; slot < n; slot++) {
      const ii = out.order[slot];
      const v = out.v[t * n + ii];
      if (v < 0.04) continue;
      c.globalAlpha = Math.min(1, v);
      c.fillStyle = viz.chosen && viz.chosen[t] === out.sel[ii] ? on : "#9698a0";
      c.fillRect(x, slot * rowH, 1, rowH - 1);
    }
    if (viz.match.mismatch && viz.match.mismatch[t]) {
      c.globalAlpha = 1;
      c.fillStyle = "#e5566f";
      c.fillRect(x, h - misH, 1, misH);
    }
  }
  c.globalAlpha = 1;
}

function renderMatch() {
  const m = S.viz!.match;
  const kv = $("matchKv");
  const pill = $("matchPill");
  const rows: [string, string, boolean?][] = [];
  rows.push(["Ticks", String(S.viz!.T)]);
  if (m.argmax) {
    const pct = (100 * m.argmax.agree) / m.argmax.total;
    rows.push(["Same choice as log", `${m.argmax.agree} / ${m.argmax.total}`]);
    pill.hidden = false;
    pill.textContent = `Match ${pct.toFixed(pct === 100 ? 0 : 1)}%`;
    pill.className = `pill ${pct >= 99.5 ? "ok" : "bad"}`;
  }
  if (m.maxDiff !== null) rows.push(["Max output difference", m.maxDiff.toExponential(2)]);
  if (!m.argmax && m.maxDiff === null) {
    rows.push(["Status", "No logged actions or outputs", true]);
    pill.hidden = false;
    pill.textContent = "No log to check";
    pill.className = "pill neutral";
  } else if (!m.argmax) {
    const ok = (m.maxDiff ?? 1) < 1e-3;
    pill.hidden = false;
    pill.textContent = ok ? "Outputs match" : "Outputs differ";
    pill.className = `pill ${ok ? "ok" : "bad"}`;
  }
  kv.innerHTML = rows.map(([k, v, text]) => `<div><dt>${k}</dt><dd${text ? ' class="text"' : ""}>${v}</dd></div>`).join("");
  const note = document.createElement("p");
  note.className = "kv-note";
  note.textContent = m.against ? `Re-ran the model on the recorded inputs and compared with the ${m.against}. Red marks on the timeline are ticks that disagree.` : "Add “actions” or “outputs” to the recording to check that this model is the one that produced it.";
  kv.appendChild(note);
}

function esc(s: string) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
}

function renderLabelsPanel() {
  const viz = S.viz!;
  const L = S.labels!;
  $("labelsSection").hidden = false;
  const inCol = viz.cols[0];
  const inList = $("inputLabels");
  inList.innerHTML = inCol.order
    .map((ii) => {
      const u = inCol.sel[ii];
      return `<li><span class="idx">${u}</span><input type="text" data-in="${u}" value="${esc(L.inputs[u])}" aria-label="Name of input ${u}" spellcheck="false" autocomplete="off" /><span></span></li>`;
    })
    .join("");
  const outCol = viz.cols[viz.cols.length - 1];
  const outList = $("outputLabels");
  outList.innerHTML = outCol.order
    .map((ii) => {
      const u = outCol.sel[ii];
      const icon = L.icons[u];
      return `<li><span class="idx">${u}</span><input type="text" data-out="${u}" value="${esc(L.outputs[u])}" aria-label="Name of output ${u}" spellcheck="false" autocomplete="off" />
        <label class="icon-pick" title="Set icon">${icon ? `<img src="${icon.src}" alt="" />` : `<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M8 3v10M3 8h10" stroke="currentColor" stroke-width="1.5"/></svg>`}<input type="file" accept=".svg,.png,.jpg,.jpeg,.webp" data-icon="${u}" aria-label="Icon for output ${u}" /></label></li>`;
    })
    .join("");
}

// ---------- events ----------
$("inputLabels").addEventListener("input", (e) => {
  const el = e.target as HTMLInputElement;
  const u = Number(el.dataset.in);
  S.nameOverride.inputs.set(u, el.value);
  S.labels!.inputs[u] = el.value;
  redraw(0);
});
$("outputLabels").addEventListener("input", (e) => {
  const el = e.target as HTMLInputElement;
  if (el.dataset.out === undefined) return;
  const u = Number(el.dataset.out);
  S.nameOverride.outputs.set(u, el.value);
  S.labels!.outputs[u] = el.value;
  redraw(0);
});
$("outputLabels").addEventListener("change", async (e) => {
  const el = e.target as HTMLInputElement;
  if (el.dataset.icon === undefined || !el.files?.[0]) return;
  const u = Number(el.dataset.icon);
  const img = await loadImage(URL.createObjectURL(el.files[0]));
  S.iconOverride.set(u, img);
  S.labels!.icons[u] = img;
  renderLabelsPanel();
  redraw(0);
});

for (const [id, fn] of [
  ["fileModel", (f: File[]) => handleFiles(f)],
  ["fileRec", (f: File[]) => handleFiles(f)],
  ["fileLabels", (f: File[]) => handleFiles(f)],
] as const) {
  $(id).addEventListener("change", (e) => {
    const input = e.target as HTMLInputElement;
    if (input.files?.length) fn([...input.files]);
    input.value = "";
  });
}

let dragDepth = 0;
window.addEventListener("dragenter", (e) => {
  e.preventDefault();
  dragDepth++;
  document.body.classList.add("drop-on");
});
window.addEventListener("dragleave", () => {
  if (--dragDepth <= 0) document.body.classList.remove("drop-on");
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove("drop-on");
  const files = [...(e.dataTransfer?.files ?? [])];
  if (files.length) handleFiles(files);
});

for (const host of [$("demoList"), $("emptyDemos")]) {
  host.innerHTML = DEMOS.map((d) => `<button class="demo" type="button" data-demo="${d.id}"><b>${d.title}</b><span>${d.sub}</span></button>`).join("");
  host.addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest("[data-demo]") as HTMLElement | null;
    if (b) loadDemo(b.dataset.demo!);
  });
}

$("play").addEventListener("click", () => (S.playing ? stop() : play()));
$("scrub").addEventListener("input", (e) => {
  stop();
  S.t = Number((e.target as HTMLInputElement).value) / S.hz;
  redraw(0);
});
window.addEventListener("keydown", (e) => {
  if (!S.viz || (e.target as HTMLElement).closest("input, select, textarea")) return;
  if (e.code === "Space") {
    e.preventDefault();
    S.playing ? stop() : play();
  } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    stop();
    const k = Math.floor(S.t * S.hz + 1e-6) + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 10 : 1);
    S.t = Math.max(0, Math.min(S.viz.T - 1, k)) / S.hz;
    redraw(0);
  }
});

const themeSeg = $("themeSeg");
themeSeg.innerHTML = THEMES.map((t, i) => `<button type="button" role="radio" aria-checked="${i === 0}" data-theme="${t.id}"><span class="swatch" style="background:${t.on}"></span>${t.name}</button>`).join("");
themeSeg.addEventListener("click", (e) => {
  const b = (e.target as HTMLElement).closest("[data-theme]") as HTMLElement | null;
  if (!b) return;
  S.theme = THEMES.find((t) => t.id === b.dataset.theme)!;
  themeSeg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
  if (S.renderer) S.renderer.opt.theme = S.theme;
  drawRibbon();
  redraw(0);
});

const bindRange = (id: string, out: string, fn: (v: number) => void) =>
  $(id).addEventListener("input", (e) => {
    const v = Number((e.target as HTMLInputElement).value);
    $(out).textContent = String(v);
    fn(v);
  });
let nodesTimer = 0;
bindRange("nodes", "nodesOut", (v) => {
  S.nodes = v;
  clearTimeout(nodesTimer);
  nodesTimer = window.setTimeout(() => S.trace && rebuild(false).catch(fail), 60);
});
bindRange("topk", "topkOut", (v) => {
  S.topK = v;
  if (S.renderer) S.renderer.opt.topK = v;
  redraw(0);
});
$("mode").addEventListener("change", (e) => {
  S.mode = (e.target as HTMLSelectElement).value as OutputMode;
  S.modeTouched = true;
  if (S.trace) rebuild(false).catch(fail);
});
$("speed").addEventListener("change", (e) => {
  S.speed = Number((e.target as HTMLSelectElement).value);
  if (S.renderer) S.renderer.opt.speed = S.speed;
});
$("hz").addEventListener("change", (e) => {
  const v = Number((e.target as HTMLInputElement).value);
  if (v > 0) {
    S.hz = v;
    makeRenderer();
    redraw(0);
  }
});
$("showFrame").addEventListener("change", (e) => {
  S.showFrame = (e.target as HTMLInputElement).checked;
  if (S.renderer) S.renderer.opt.showFrame = S.showFrame;
  redraw(0);
});
$("transparent").addEventListener("change", (e) => {
  S.transparent = (e.target as HTMLInputElement).checked;
  if (S.renderer) S.renderer.opt.transparent = S.transparent;
  $("well").classList.toggle("checker", S.transparent);
  redraw(0);
});
$("size").addEventListener("change", () => {
  makeRenderer();
  redraw(0);
});
$("fps").addEventListener("change", () => {
  if (S.renderer) S.renderer.opt.fps = Number(($("fps") as HTMLSelectElement).value);
});
const FORMAT_LABEL: Record<ExportFormat, string> = { gif: "GIF", mp4: "MP4", webm: "WebM", "webm-alpha": "WebM with alpha" };
$("format").addEventListener("change", (e) => {
  const f = (e.target as HTMLSelectElement).value as ExportFormat;
  $("exportBtn").textContent = `Export ${FORMAT_LABEL[f]}`;
});

// drag inputs / outputs to reorder
let drag: { col: number; slot: number } | null = null;
const toCanvas = (e: PointerEvent) => {
  const r = view.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * view.width, ((e.clientY - r.top) / r.height) * view.height];
};
view.addEventListener("pointerdown", (e) => {
  if (!S.renderer) return;
  const [x, y] = toCanvas(e);
  const h = S.renderer.hit(x, y);
  if (!h) return;
  drag = h;
  view.setPointerCapture(e.pointerId);
  view.classList.add("dragging");
});
view.addEventListener("pointermove", (e) => {
  if (!S.renderer || !S.viz) return;
  const [x, y] = toCanvas(e);
  if (!drag) {
    view.classList.toggle("can-drag", !!S.renderer.hit(x, y));
    return;
  }
  const to = S.renderer.slotAt(drag.col, y);
  if (to !== drag.slot) {
    const col = S.viz.cols[drag.col];
    const [it] = col.order.splice(drag.slot, 1);
    col.order.splice(to, 0, it);
    drag.slot = to;
    if (drag.col === S.viz.cols.length - 1) orderHidden(S.viz);
    redraw(0);
  }
});
const endDrag = () => {
  if (!drag) return;
  drag = null;
  view.classList.remove("dragging");
  renderLabelsPanel();
  drawRibbon();
};
view.addEventListener("pointerup", endDrag);
view.addEventListener("pointercancel", endDrag);

// export
async function doExport() {
  if (!S.viz || !S.labels) return;
  stop();
  const format = ($("format") as HTMLSelectElement).value as ExportFormat;
  const o = renderOpts();
  const dur = (S.viz.T - 1) / S.hz;
  const tStart = Math.max(0, Math.min(dur, Number(($("from") as HTMLInputElement).value) || 0));
  const tEnd = Math.max(tStart + 1 / S.hz, Math.min(dur, Number(($("to") as HTMLInputElement).value) || dur));
  const ac = new AbortController();
  S.exportAbort = ac;
  $("exportProgress").hidden = false;
  $("exportResult").textContent = "";
  ($("exportBtn") as HTMLButtonElement).disabled = true;
  ($("exportTop") as HTMLButtonElement).disabled = true;
  try {
    const t0 = performance.now();
    const res = await runExport({
      viz: S.viz,
      labels: S.labels,
      opt: o,
      format,
      tStart,
      tEnd,
      signal: ac.signal,
      onProgress: (f, note) => {
        $("exportFill").style.width = `${Math.round(f * 100)}%`;
        $("exportNote").textContent = note;
      },
    });
    if (S.lastUrl) URL.revokeObjectURL(S.lastUrl);
    S.lastUrl = URL.createObjectURL(res.blob);
    const name = `${(S.labels.title || "netglow").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")}.${res.ext}`;
    const a = document.createElement("a");
    a.href = S.lastUrl;
    a.download = name;
    a.textContent = `Download ${name}`;
    const p = $("exportResult");
    p.textContent = "";
    p.append(a, ` · ${res.note}, ${o.width}×${o.height}, ${res.frames} frames, ${(res.blob.size / 1e6).toFixed(1)} MB, ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    (window as unknown as { __lastExport: { url: string; name: string; size: number } }).__lastExport = { url: S.lastUrl, name, size: res.blob.size };
    a.click();
  } catch (e) {
    $("exportResult").textContent = e instanceof DOMException && e.name === "AbortError" ? "Export cancelled." : `Export failed: ${e instanceof Error ? e.message : e}`;
  } finally {
    $("exportProgress").hidden = true;
    ($("exportBtn") as HTMLButtonElement).disabled = false;
    ($("exportTop") as HTMLButtonElement).disabled = false;
    S.exportAbort = null;
    redraw(0);
  }
}
$("exportBtn").addEventListener("click", doExport);
$("exportTop").addEventListener("click", doExport);
$("exportCancel").addEventListener("click", () => S.exportAbort?.abort());
window.addEventListener("resize", () => drawRibbon());

// test / automation hook
(window as unknown as { netglow: unknown }).netglow = { S, loadDemo, handleFiles, redraw, play, stop };

const q = new URLSearchParams(location.search).get("demo");
if (q) loadDemo(q);
