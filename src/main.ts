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
import { imageView, type ImageSettings, type ImageView } from "./image";
import { DEFAULT_STYLE, type PicPos, type Style } from "./render";
import { computeSaliency, type SaliencyMap } from "./saliency";
import { CompareRenderer, type CompareLayout } from "./compare";
import type { RenderOpts } from "./render";
import { legendHTML } from "./legend";
import { DEFAULT_SETTINGS, clearStored, clone, diff, fromHash, loadStored, merge, store, toHash, type Settings } from "./settings";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

interface LabelsFile {
  title?: string;
  description?: string;
  inputs?: string[];
  outputs?: string[];
  icons?: (string | null)[];
  output_mode?: OutputMode;
  input_units?: (string | null)[] | null;
  /** compare demos: a second model in the same folder, and names for the two panels */
  compare?: { model: string; a?: string; b?: string };
}

/** the second model in compare mode, run on the same recording */
interface Cmp {
  model: ModelInfo;
  plan: Plan;
  name: string;
  /** panel names */
  a: string;
  b: string;
  obsList: NdArray[];
  trace: Trace | null;
  viz: Viz | null;
  labels: Labels | null;
  sal: SaliencyMap | null;
  salFor: Viz | null;
}

const DEMOS = [
  { id: "lunar-mlp", title: "LunarLander, MLP", sub: "PPO policy reading 8 numbers" },
  { id: "lunar-cnn", title: "LunarLander, CNN", sub: "a CNN pilot that flies from pixels" },
  { id: "lunar-mixed", title: "LunarLander, pixels + numbers", sub: "two inputs: frames and 6 plain numbers" },
  { id: "lunar-resmlp", title: "LunarLander, residual MLP", sub: "a skip connection around a two-layer block" },
  { id: "lunar-lstm", title: "LunarLander, LSTM", sub: "never sees its speed, remembers it instead" },
  { id: "lunar-compare", title: "LunarLander, early vs trained", sub: "two checkpoints of one policy, side by side" },
];

const S = {
  model: null as ModelInfo | null,
  modelError: "",
  modelName: "",
  plan: null as Plan | null,
  rec: null as Map<string, NdArray> | null,
  recName: "",
  /** the input shown as a picture (the image input, else the first one) */
  obs: null as NdArray | null,
  /** one recording array per model input, in plan.inputs order */
  obsList: [] as NdArray[],
  labelsFile: null as LabelsFile | null,
  iconFiles: new Map<string, HTMLImageElement>(), // by lowercased basename without extension
  iconOverride: new Map<number, HTMLImageElement>(),
  nameOverride: { inputs: new Map<number, string>(), outputs: new Map<number, string>() },
  trace: null as Trace | null,
  viz: null as Viz | null,
  renderer: null as Renderer | CompareRenderer | null,
  cmp: null as Cmp | null,
  cmpLayout: "auto" as CompareLayout,
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
  style: { ...DEFAULT_STYLE } as Style,
  /** settings from a share link, applied to the first model that loads */
  pendingHash: null as unknown,
  showMaps: true,
  img: { range: "auto", layout: "auto", newest: "last" } as ImageSettings,
  imageView: null as ImageView | null,
  transparent: false,
  exportAbort: null as AbortController | null,
  /** saliency heatmap of the current viz (computed when first switched on) */
  sal: null as SaliencyMap | null,
  salFor: null as Viz | null,
  lastUrl: "",
  lastBlob: null as Blob | null,
  lastExt: "",
  /** the bundled demo on screen (null for the visitor's own files): embeds can point at it */
  demoId: null as string | null,
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
  S.demoId = null;
  S.model = null;
  S.plan = null;
  S.modelName = name;
  markFile("model", name);
  S.modelError = "";
  S.trace = null;
  S.viz = null;
  try {
    const m = parseModel(bytes);
    S.model = m;
    S.plan = buildPlan(m); // throws UnsupportedGraph with a clear message
  } catch (e) {
    S.modelError = e instanceof Error ? e.message : String(e);
    throw e;
  }
  S.modeTouched = false;
}

async function loadRecording(bytes: Uint8Array, name: string) {
  let rec: Map<string, NdArray>;
  if (name.endsWith(".npz") || (bytes[0] === 0x50 && bytes[1] === 0x4b)) rec = await parseNpz(bytes);
  else if (name.endsWith(".npy")) rec = new Map([["obs", parseNpy(bytes)]]);
  else rec = parseJsonRecording(new TextDecoder().decode(bytes));
  const keys = [...rec.keys()];
  if (!rec.has("obs") && !rec.has("inputs") && !keys.some((k) => k.startsWith("obs_")) && !(S.plan && S.plan.inputs.every((i) => rec.has(i.name))))
    throw new Error(`The recording needs an "obs" array (one row per tick), or one "obs_<input name>" array per model input. Found: ${keys.join(", ") || "nothing"}.`);
  S.rec = rec;
  S.obs = null;
  S.obsList = [];
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

/** the recording array for each model input: obs_<name>, then <name>, then plain obs for a single-input model */
function resolveObs(plan: Plan, rec: Map<string, NdArray>): NdArray[] {
  const one = plan.inputs.length === 1;
  return plan.inputs.map((inp) => {
    const a = rec.get(`obs_${inp.name}`) ?? rec.get(inp.name) ?? (one ? (rec.get("obs") ?? rec.get("inputs")) : undefined);
    if (!a)
      throw new Error(`The model has ${plan.inputs.length} inputs (${plan.inputs.map((i) => i.name).join(", ")}); the recording needs an "obs_${inp.name}" array. Found: ${[...rec.keys()].join(", ")}.`);
    return a;
  });
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
      if (Array.isArray(j.obs) || Array.isArray(j.inputs?.[0]) || Object.keys(j).some((k) => k.startsWith("obs_"))) recs.push(f);
      else {
        S.labelsFile = j;
        markFile("labels", f.name + (icons.length ? ` + ${icons.length} icons` : ""));
      }
    }
    if (models[0]) {
      await loadModelBytes(new Uint8Array(await models[0].arrayBuffer()), models[0].name);
      applySaved();
    }
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
    status(`Loading demo ${id}…`);
    busy(true, "Loading the demo…", 0.1);
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
    S.demoId = id;
    await loadRecording(rec, `${id}/recording.npz`);
    S.showFrame = !!S.rec?.get("frames");
    ($("showFrame") as HTMLInputElement).checked = S.showFrame;
    applySaved();
    const cf = S.labelsFile?.compare;
    S.cmp = cf ? newCmp(await get(cf.model), `${id}/${cf.model}`, cf.a, cf.b) : null;
    syncCompareUI();
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
    if (S.modelError) status(S.modelError, true);
    else if (S.rec) status("Recording loaded. Now drop the ONNX model.");
    return;
  }
  if (!S.rec) {
    status(`Model loaded: ${S.plan.layers.length} weighted layers. Now drop a recording.`);
    $("docTitle").textContent = S.labelsFile?.title ?? S.modelName;
    return;
  }
  if (rerun || !S.trace) {
    S.obsList = resolveObs(S.plan, S.rec);
    const imgIdx = S.plan.inputs.findIndex((i) => i.isImage);
    S.obs = S.obsList[Math.max(0, imgIdx)];
    if (!S.modeTouched) {
      const lf = S.labelsFile?.output_mode;
      const acts = S.rec.get("actions");
      const nOut = S.plan.layers[S.plan.layers.length - 1].out;
      S.mode = lf ?? (S.plan.softmaxInGraph ? "categorical" : acts && acts.dtype.startsWith("f") && acts.shape.length > 1 ? "continuous" : nOut > 1 ? "categorical" : "continuous");
      ($("mode") as HTMLSelectElement).value = S.mode;
    }
    busy(true, "Running the model in this tab…", 0);
    S.trace = await runTrace(S.model, S.plan, S.obsList, (f) => busy(true, "Running the model in this tab…", f), S.rec);
    busy(false);
  }
  const prev = S.viz ? { inputs: S.viz.cols.filter((c) => c.kind === "input").map((c) => c.order), output: S.viz.cols[S.viz.cols.length - 1].order } : undefined;
  S.viz = buildViz(S.plan, S.trace, S.rec, S.obsList, { nodesPerColumn: S.nodes, maxInputs: 24, mode: S.mode }, rerun ? undefined : prev);
  S.labels = resolveLabels();
  if (S.cmp) await buildCompare(rerun);
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
  syncStyleUI();
  renderLegend();
  renderEmbed();
  $("empty").hidden = true;
  $("docTitle").textContent = S.labelsFile?.title ?? S.modelName.replace(/\.onnx$/, "");
  for (const id of ["play", "scrub", "exportBtn", "exportTop"]) ($(id) as HTMLButtonElement).disabled = false;
  const sc = $("scrub") as HTMLInputElement;
  sc.max = String(S.viz.T - 1);
  syncFrameSection();
  drawRibbon();
  redraw(0);
  // web fonts may still be loading on first open; draw again once they are in
  document.fonts?.ready.then(() => S.playing || redraw(0));
}

function resolveLabels(viz: Viz = S.viz!, plan: Plan = S.plan!): Labels {
  const lf = S.labelsFile ?? {};
  const rec = S.rec!;
  const nOut = viz.cols[viz.cols.length - 1].n;
  const recIn = rec.get("input_names")?.strings;
  const recOut = rec.get("output_names")?.strings;
  // one flat list of names over every model input, in model order; an image input takes one name
  const inCols = viz.cols.filter((c) => c.kind === "input");
  const multi = inCols.length > 1;
  const nIn = Math.max(0, ...inCols.map((c) => (c.base ?? 0) + (c.image ? 1 : c.n)));
  const inputs: string[] = new Array(nIn);
  for (const c of inCols) {
    const b = c.base ?? 0;
    const tensor = plan.inputs[c.input ?? 0].name;
    if (c.image) inputs[b] = S.nameOverride.inputs.get(b) ?? (multi ? (lf.inputs?.[b] ?? tensor) : "image");
    else
      for (let i = 0; i < c.n; i++)
        inputs[b + i] = S.nameOverride.inputs.get(b + i) ?? lf.inputs?.[b + i] ?? recIn?.[b + i] ?? (multi ? `${tensor}[${i}]` : `in ${i}`);
  }
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
    image: S.imageView,
    style: S.style,
    showMaps: S.showMaps,
    frames: S.rec?.get("frames") ?? null,
    obs: S.obs ?? S.obsList[0],
    hz: S.hz,
    speed: S.speed,
    fps: Number(($("fps") as HTMLSelectElement).value),
    saliency: ensureSaliency(),
  };
}

// ---------- compare mode ----------
function newCmp(bytes: Uint8Array, name: string, a?: string, b?: string): Cmp {
  const model = parseModel(bytes);
  const plan = buildPlan(model);
  return { model, plan, name, a: a ?? "", b: b ?? "", obsList: [], trace: null, viz: null, labels: null, sal: null, salFor: null };
}

/** run the second model on the recording (when needed) and build its picture with the same options */
async function buildCompare(rerun: boolean) {
  const c = S.cmp!;
  try {
    if (rerun || !c.trace) {
      c.obsList = resolveObs(c.plan, S.rec!);
      busy(true, "Running the second model…", 0);
      c.trace = await runTrace(c.model, c.plan, c.obsList, (f) => busy(true, "Running the second model…", f), S.rec!);
      busy(false);
      if (c.trace.T !== S.trace!.T) throw new Error(`it ran ${c.trace.T} ticks, the first model ${S.trace!.T}`);
    }
    const prev = c.viz ? { inputs: c.viz.cols.filter((x) => x.kind === "input").map((x) => x.order), output: c.viz.cols[c.viz.cols.length - 1].order } : undefined;
    c.viz = buildViz(c.plan, c.trace, S.rec!, c.obsList, { nodesPerColumn: S.nodes, maxInputs: 24, mode: S.mode }, rerun ? undefined : prev);
    c.labels = resolveLabels(c.viz, c.plan);
  } catch (e) {
    busy(false);
    S.cmp = null;
    syncCompareUI();
    status(`Second model: ${e instanceof Error ? e.message : e}`, true);
  }
}

function cmpSaliency(c: Cmp): SaliencyMap | null {
  const sp = c.plan.saliency;
  if (!S.style.saliency || !sp || sp.error || !c.viz || !c.trace) return null;
  if (c.salFor === c.viz && c.sal) return c.sal;
  const img = c.obsList[Math.max(0, c.plan.inputs.findIndex((i) => i.isImage))];
  const target = new Int32Array(c.viz.T).map((_, t) => (c.viz!.chosen ? c.viz!.chosen[t] : 0));
  c.sal = computeSaliency(c.model, sp, c.plan.outputTensor, c.trace.tensors, toFloat(img), img.shape.slice(1), target);
  c.salFor = c.viz;
  return c.sal;
}

/** the picture for render options o: one network, or the two compared ones */
function makeDrawable(o: RenderOpts): Renderer | CompareRenderer {
  const c = S.cmp;
  if (!c?.viz || !c.labels) return new Renderer(S.viz!, S.labels!, o);
  const imgB = c.viz.imageInput ? c.obsList[Math.max(0, c.plan.inputs.findIndex((i) => i.isImage))] : c.obsList[0];
  const short = (n: string) => n.replace(/\.onnx$/, "").split("/").pop();
  return new CompareRenderer(
    { viz: S.viz!, labels: S.labels!, name: `A · ${c.a || short(S.modelName)}` },
    { viz: c.viz, labels: c.labels, name: `B · ${c.b || short(c.name)}`, opt: { image: c.viz.imageInput ? imageView(imgB, S.img) : null, obs: imgB, saliency: cmpSaliency(c) } },
    o, S.cmpLayout, matchMedia("(max-width: 700px)").matches,
  );
}

function syncCompareUI() {
  const on = !!S.cmp;
  document.querySelectorAll<HTMLElement>(".compare-on").forEach((el) => (el.hidden = !on));
  $("nameCompare").textContent = on ? S.cmp!.name : ".onnx, run on the same recording";
  document.querySelector('.file-row[data-kind="compare"]')!.classList.toggle("loaded", on);
  if (!on) $("cmpNote").textContent = "";
}

function renderCompareNote() {
  const c = S.cmp;
  if (!c?.viz || !(S.renderer instanceof CompareRenderer)) return;
  const ag = S.renderer.agree.same;
  const m = c.viz.match.argmax;
  $("cmpNote").textContent =
    (ag !== null ? `Same choice as A on ${Math.round(ag * 100)}% of the ${c.viz.T} ticks.` : "Values outputs: the strip shows how close they are.") +
    (m ? ` B matches the logged actions on ${((100 * m.agree) / m.total).toFixed(1)}%.` : "");
}

/** gradient × input of the chosen output for every tick, computed once per viz when the heatmap is on */
function ensureSaliency(): SaliencyMap | null {
  const sp = S.plan?.saliency;
  if (!S.style.saliency || !sp || sp.error || !S.viz || !S.trace || !S.model || !S.obs) return null;
  if (S.salFor === S.viz && S.sal) return S.sal;
  const V = S.viz;
  const out = S.trace.tensors.get(S.plan!.outputTensor)!;
  const target = new Int32Array(V.T);
  for (let t = 0; t < V.T; t++) {
    if (V.chosen) target[t] = V.chosen[t];
    else {
      let b = 0;
      for (let i = 1; i < out.size; i++) if (Math.abs(out.data[t * out.size + i]) > Math.abs(out.data[t * out.size + b])) b = i;
      target[t] = b;
    }
  }
  S.sal = computeSaliency(S.model, sp, S.plan!.outputTensor, S.trace.tensors, toFloat(S.obs), S.obs.shape.slice(1), target);
  S.salFor = V;
  return S.sal;
}

function makeRenderer() {
  if (!S.viz || !S.labels) return;
  S.imageView = S.viz.imageInput && S.obs ? imageView(S.obs, S.img) : null;
  const salErr = S.plan?.saliency?.error;
  $("imageDescribe").textContent = S.imageView ? `Reading the input as ${S.imageView.describe}.${salErr ? ` ${salErr}` : ""}` : "";
  document.querySelectorAll<HTMLElement>(".image-only").forEach((el) => (el.hidden = !S.imageView));
  document.querySelectorAll<HTMLElement>(".maps-only").forEach((el) => (el.hidden = !S.viz!.cols.some((c) => c.maps)));
  const o = renderOpts();
  S.renderer = makeDrawable(o);
  // a stacked comparison is taller than the chosen size
  const [cw, ch] = S.renderer instanceof CompareRenderer ? [S.renderer.width, S.renderer.height] : [o.width, o.height];
  view.width = cw;
  view.height = ch;
  $("well").style.setProperty("--canvas-ratio", `${cw} / ${ch}`);
  renderCompareNote();
  // the canvas is the picture; say in words what it shows
  const ins = S.viz.cols.filter((c) => c.kind === "input").map((c) => (c.image ? `an image input (${S.labels!.inputs[c.base ?? 0]})` : `${c.sel.length} numbers`));
  view.setAttribute("aria-label", `Network activation view: ${ins.join(" and ")} through ${S.viz.cols.filter((c) => c.kind === "hidden").length} hidden layers to ${S.viz.cols[S.viz.cols.length - 1].sel.length} outputs. Reorder rows with Alt + arrow keys in the Inputs and Outputs lists.`);
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
/** the side preview only when the canvas does not already show the recording */
function syncFrameSection() {
  $("frameSection").hidden = !S.rec?.get("frames") || S.showFrame;
}
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
  // compare mode: a row on top, red where the two models choose differently
  const agree = S.renderer instanceof CompareRenderer ? S.renderer.agree : null;
  const agH = agree ? 5 * devicePixelRatio : 0;
  const rowH = (h - misH - agH) / n;
  const on = S.theme.on;
  for (let x = 0; x < w; x++) {
    const t = Math.min(viz.T - 1, Math.floor((x / w) * viz.T));
    for (let slot = 0; slot < n; slot++) {
      const ii = out.order[slot];
      const v = out.v[t * n + ii];
      if (v < 0.04) continue;
      c.globalAlpha = Math.min(1, v);
      c.fillStyle = viz.chosen && viz.chosen[t] === out.sel[ii] ? on : "#9698a0";
      c.fillRect(x, agH + slot * rowH, 1, rowH - 1);
    }
    // episode starts (recurrent models reset their state there): a thin full-height mark
    if (viz.starts && t > 0 && viz.starts[t] && Math.floor(((x - 1) / w) * viz.T) < t) {
      c.globalAlpha = 0.9;
      c.fillStyle = "#f0f0f0";
      c.fillRect(x, 0, Math.max(1, devicePixelRatio), h);
    }
    if (agree && agree.v[t] < 0.5) {
      c.globalAlpha = 1;
      c.fillStyle = "#e5566f";
      c.fillRect(x, 0, 1, agH - devicePixelRatio);
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
  if (S.viz!.starts) rows.push(["Episodes (state resets)", String(S.viz!.starts.reduce((a, b) => a + b, 0))]);
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
  const inList = $("inputLabels");
  // every input column, in the order drawn; data-in is the name's index in labels.inputs, data-col the column
  inList.innerHTML = viz.cols
    .map((c, ci) => (c.kind !== "input" ? "" : c.order
      .map((ii) => {
        const u = c.image ? 0 : c.sel[ii];
        const g = (c.base ?? 0) + u;
        const shown = c.image ? "img" : String(u);
        return `<li><span class="idx">${shown}</span><input type="text" name="input-${g}" data-in="${g}" data-col="${ci}" data-unit="${c.image ? "" : u}" value="${esc(L.inputs[g] ?? "")}" aria-label="Name of input ${esc(L.inputs[g] ?? String(g))}" spellcheck="false" autocomplete="off" /><span></span></li>`;
      })
      .join("")))
    .join("");
  const outCol = viz.cols[viz.cols.length - 1];
  const outList = $("outputLabels");
  outList.innerHTML = outCol.order
    .map((ii) => {
      const u = outCol.sel[ii];
      const icon = L.icons[u];
      return `<li><span class="idx">${u}</span><input type="text" name="output-${u}" data-out="${u}" value="${esc(L.outputs[u])}" aria-label="Name of output ${u}" spellcheck="false" autocomplete="off" />
        <label class="icon-pick" title="Set icon">${icon ? `<img src="${icon.src}" alt="" />` : `<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M8 3v10M3 8h10" stroke="currentColor" stroke-width="1.5"/></svg>`}<input type="file" accept=".svg,.png,.jpg,.jpeg,.webp" data-icon="${u}" aria-label="Icon for output ${u}" /></label></li>`;
    })
    .join("");
}

// ---------- events ----------
/** Alt + arrow in a name field moves that row: the keyboard version of dragging on the canvas */
function keyReorder(listId: string, colOf: (el: HTMLInputElement) => number, attr: "in" | "out") {
  $(listId).addEventListener("keydown", (e) => {
    const el = e.target as HTMLInputElement;
    if (!e.altKey || (e.key !== "ArrowUp" && e.key !== "ArrowDown") || el.dataset[attr] === undefined || !S.viz) return;
    if (attr === "in" && el.dataset.unit === "") return; // the image node is a single row
    e.preventDefault();
    const ci = colOf(el);
    const col = S.viz.cols[ci];
    const unit = Number(attr === "in" ? el.dataset.unit : el.dataset[attr]);
    const slot = col.order.findIndex((ii) => col.sel[ii] === unit);
    const to = slot + (e.key === "ArrowUp" ? -1 : 1);
    if (to < 0 || to >= col.order.length) return;
    const [it] = col.order.splice(slot, 1);
    col.order.splice(to, 0, it);
    if (ci === S.viz.cols.length - 1) orderHidden(S.viz);
    renderLabelsPanel();
    drawRibbon();
    redraw(0);
    ($(listId).querySelector(`input[data-${attr}="${el.dataset[attr]}"]`) as HTMLInputElement | null)?.focus();
  });
}
keyReorder("inputLabels", (el) => Number(el.dataset.col), "in");
keyReorder("outputLabels", () => S.viz!.cols.length - 1, "out");

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

$("fileCompare").addEventListener("change", async (e) => {
  const input = e.target as HTMLInputElement;
  const f = input.files?.[0];
  input.value = "";
  if (!f) return;
  try {
    S.cmp = newCmp(new Uint8Array(await f.arrayBuffer()), f.name);
    syncCompareUI();
    if (S.trace) await rebuild(false);
  } catch (err) {
    S.cmp = null;
    syncCompareUI();
    status(`Second model: ${err instanceof Error ? err.message : err}`, true);
  }
});
$("cmpLayout").addEventListener("change", (e) => {
  S.cmpLayout = (e.target as HTMLSelectElement).value as CompareLayout;
  makeRenderer();
  redraw(0);
});
$("cmpOff").addEventListener("click", () => {
  S.cmp = null;
  syncCompareUI();
  makeRenderer();
  drawRibbon();
  redraw(0);
});

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
  renderLegend();
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
for (const [id, key] of [["pxRange", "range"], ["pxLayout", "layout"], ["pxNewest", "newest"]] as const)
  $(id).addEventListener("change", (e) => {
    (S.img as unknown as Record<string, string>)[key] = (e.target as HTMLSelectElement).value;
    makeRenderer();
    redraw(0);
  });
$("showMaps").addEventListener("change", (e) => {
  S.showMaps = (e.target as HTMLInputElement).checked;
  if (S.renderer) S.renderer.opt.showMaps = S.showMaps;
  redraw(0);
});
$("showFrame").addEventListener("change", (e) => {
  S.showFrame = (e.target as HTMLInputElement).checked;
  if (S.renderer) S.renderer.opt.showFrame = S.showFrame;
  syncFrameSection();
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
    if (S.viz.cols[drag.col].kind === "output") orderHidden(S.viz);
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
      make: makeDrawable,
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
    S.lastBlob = res.blob;
    S.lastExt = res.ext;
    ($("playerPage") as HTMLButtonElement).disabled = false;
    const name = `${(S.labels.title || "netglow").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")}.${res.ext}`;
    const a = document.createElement("a");
    a.href = S.lastUrl;
    a.download = name;
    a.textContent = `Download ${name}`;
    const p = $("exportResult");
    p.textContent = "";
    p.append(a, ` · ${res.note}, ${view.width}×${view.height}, ${res.frames} frames, ${(res.blob.size / 1e6).toFixed(1)} MB, ${((performance.now() - t0) / 1000).toFixed(1)} s`);
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

// ---------- "What am I looking at" ----------
const LEGEND_KEY = "netglow:legend-seen:v1";
const embed = new URLSearchParams(location.search).get("embed") === "1";
function renderLegend() {
  const V = S.viz;
  $("legendList").innerHTML = legendHTML(S.theme, S.style.pos || S.theme.pos, S.style.neg || S.theme.neg, {
    image: !!V?.imageInput, maps: !!V?.cols.some((c) => c.maps) && S.showMaps, skip: !!V?.edges.some((e) => e.kind === "skip"),
    recurrent: !!V?.edges.some((e) => e.kind === "recurrent"), saliency: !!S.style.saliency && !!S.plan?.saliency && !S.plan.saliency.error, compare: !!S.cmp?.viz,
  });
}
function showLegend(on: boolean) {
  $("legend").hidden = !on;
  $("legendBtn").setAttribute("aria-expanded", String(on));
  if (on) {
    renderLegend();
    ($("legendOk") as HTMLButtonElement).focus({ preventScroll: true });
  } else {
    try {
      localStorage.setItem(LEGEND_KEY, "1");
    } catch {
      /* private mode: it shows again next time */
    }
  }
}
$("legendBtn").addEventListener("click", () => showLegend($("legend").hidden));
$("legendClose").addEventListener("click", () => showLegend(false));
$("legendOk").addEventListener("click", () => {
  showLegend(false);
  $("legendBtn").focus();
});
$("legend").addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    showLegend(false);
    $("legendBtn").focus();
  }
});
let legendSeen = false;
try {
  legendSeen = localStorage.getItem(LEGEND_KEY) === "1";
} catch {
  legendSeen = false;
}

// ---------- embed ----------
/** an <iframe> of the hosted app in ?embed=1 mode for a bundled demo; for your own files, a self-contained player page */
function embedCode(): string {
  const [w, h] = [view.width, view.height];
  const title = (S.labels?.title ?? "netglow").replace(/"/g, "&quot;");
  const dw = Math.min(720, w);
  const dh = Math.round((dw * h) / w);
  if (S.demoId) {
    const src = `${location.origin}${location.pathname}?demo=${encodeURIComponent(S.demoId)}&embed=1${toHash(collectSettings())}`;
    return `<iframe src="${src}" width="${dw}" height="${dh}" style="border:0;max-width:100%;aspect-ratio:${w}/${h};height:auto" loading="lazy" title="netglow: ${title}"></iframe>`;
  }
  return `<iframe src="netglow-player.html" width="${dw}" height="${dh}" style="border:0;max-width:100%;aspect-ratio:${w}/${h};height:auto" loading="lazy" title="netglow: ${title}"></iframe>`;
}
function renderEmbed() {
  if (!S.viz) return;
  ($("embedCode") as HTMLTextAreaElement).value = embedCode();
  $("embedNote").textContent = S.demoId
    ? "Plays this demo with this look on any page (the hosted app in a chrome-less embed mode)."
    : "Your files stay on your machine, so the embed plays an export: export first (MP4 or WebM stay smallest), then download the player page and host it next to your page.";
}
$("copyEmbed").addEventListener("click", async () => {
  renderEmbed();
  const code = ($("embedCode") as HTMLTextAreaElement).value;
  try {
    await navigator.clipboard.writeText(code);
    $("embedNote").textContent = "Embed code copied.";
  } catch {
    ($("embedCode") as HTMLTextAreaElement).select();
    $("embedNote").textContent = "Select the code above and copy it.";
  }
});
$("playerPage").addEventListener("click", async () => {
  if (!S.lastBlob) return;
  const data = await new Promise<string>((res) => {
    const r = new FileReader();
    r.onload = () => res(String(r.result));
    r.readAsDataURL(S.lastBlob!);
  });
  const title = esc(S.labels?.title ?? "netglow");
  const media = S.lastExt === "gif" ? `<img src="${data}" alt="${title}">` : `<video src="${data}" autoplay loop muted playsinline aria-label="${title}"></video>`;
  const html = `<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${title}</title>
<style>html,body{margin:0;height:100%;background:${S.transparent ? "transparent" : S.style.bg || S.theme.bg}}body{display:grid;place-items:center}img,video{max-width:100%;max-height:100vh;display:block}</style></head>
<body>${media}<script>if(matchMedia("(prefers-reduced-motion: reduce)").matches){const v=document.querySelector("video");if(v){v.pause();v.controls=true}}</script></body></html>\n`;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([html], { type: "text/html" }));
  a.download = "netglow-player.html";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
if (embed) {
  document.body.classList.add("embed");
  const link = document.createElement("a");
  link.className = "embed-link";
  link.href = location.href.replace(/[?&]embed=1/, "");
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = "netglow ↗";
  $("well").appendChild(link);
  // click or Space (the window's key handler) plays and pauses; the canvas takes focus so Space reaches it
  view.tabIndex = 0;
  view.addEventListener("click", () => (S.playing ? stop() : play()));
}

// test / automation hook
(window as unknown as { netglow: unknown }).netglow = { S, loadDemo, handleFiles, redraw, play, stop, showLegend };

// first open: load the MLP demo so the page shows a live network straight away (?demo=none keeps it empty)
const q = new URLSearchParams(location.search).get("demo") ?? "lunar-mlp";
if (q !== "none")
  loadDemo(q).then(() => {
    if (!embed && !legendSeen && S.viz) showLegend(true);
    if (S.viz && !matchMedia("(prefers-reduced-motion: reduce)").matches && !new URLSearchParams(location.search).has("paused")) play();
  });

// ---------- settings: layout, size and look; persisted per model, shareable ----------

type Ctl =
  | { key: keyof Style; label: string; type: "range"; min: number; max: number; step: number; unit?: string; only?: "image" | "frames" | "maps" }
  | { key: keyof Style; label: string; type: "select"; options: [string, string][]; only?: "image" | "frames" | "maps" }
  | { key: keyof Style; label: string; type: "check"; only?: "image" | "frames" | "maps" }
  | { key: keyof Style; label: string; type: "color" };

const POS: [PicPos, string][] = [["above", "Above"], ["left", "Beside, left"], ["right", "Beside, right"], ["tl", "Corner, top left"], ["tr", "Corner, top right"], ["bl", "Corner, bottom left"], ["br", "Corner, bottom right"], ["off", "Off"]];
const GROUPS: { title: string; ctls: Ctl[] }[] = [
  {
    title: "Layout and size",
    ctls: [
      { key: "nodeScale", label: "Node size", type: "range", min: 0.4, max: 2.5, step: 0.05, unit: "×" },
      { key: "edgeMin", label: "Thinnest edge", type: "range", min: 0.25, max: 6, step: 0.25, unit: "px" },
      { key: "edgeMax", label: "Strongest edge", type: "range", min: 0.5, max: 10, step: 0.25, unit: "px" },
      { key: "labelSize", label: "Label text", type: "range", min: 9, max: 36, step: 1, unit: "px" },
      { key: "valueSize", label: "Value text", type: "range", min: 8, max: 32, step: 1, unit: "px" },
      { key: "mapScale", label: "Feature-map tiles", type: "range", min: 0.4, max: 2.5, step: 0.05, unit: "×", only: "maps" },
      { key: "imageScale", label: "Image node", type: "range", min: 0.4, max: 2.5, step: 0.05, unit: "×", only: "image" },
      { key: "colSpacing", label: "Column spacing", type: "range", min: 0.4, max: 1.4, step: 0.02, unit: "×" },
      { key: "rowSpacing", label: "Row spacing", type: "range", min: 0.4, max: 1.8, step: 0.02, unit: "×" },
      { key: "padding", label: "Canvas padding", type: "range", min: 0, max: 96, step: 2, unit: "px" },
      { key: "titles", label: "Column titles", type: "check" },
    ],
  },
  {
    title: "Pictures",
    ctls: [
      { key: "aiPos", label: "What the AI sees", type: "select", options: POS, only: "image" },
      { key: "aiScale", label: "AI view size", type: "range", min: 0.3, max: 2, step: 0.05, unit: "×", only: "image" },
      { key: "recPos", label: "Recording", type: "select", options: [["auto", "Next to the AI view"], ...POS], only: "frames" },
      { key: "recScale", label: "Recording size", type: "range", min: 0.3, max: 2, step: 0.05, unit: "×", only: "frames" },
      { key: "saliency", label: "Saliency heatmap (gradient × input)", type: "check", only: "image" },
      { key: "saliencyOpacity", label: "Heatmap opacity", type: "range", min: 0.1, max: 1, step: 0.05, unit: "×", only: "image" },
    ],
  },
  {
    title: "Look",
    ctls: [
      { key: "glow", label: "Glow", type: "range", min: 0, max: 3, step: 0.05, unit: "×" },
      { key: "pulse", label: "Decision pulse", type: "check" },
      { key: "pulseIntensity", label: "Pulse strength", type: "range", min: 0, max: 5, step: 0.1, unit: "×" },
      { key: "shape", label: "Node shape", type: "select", options: [["circle", "Circle"], ["square", "Square"]] },
      { key: "showValues", label: "Input values", type: "check" },
      { key: "showProbs", label: "Output probabilities", type: "check" },
      { key: "pos", label: "Exciting edges", type: "color" },
      { key: "neg", label: "Inhibiting edges", type: "color" },
      { key: "bg", label: "Background", type: "color" },
    ],
  },
];

const themeColour = (key: "pos" | "neg" | "bg") => S.theme[key];

function buildStyleUI() {
  const host = $("styleGroups");
  host.innerHTML =
    GROUPS.map(
      (g) => `<details class="group"><summary>${g.title}</summary><div class="group-body">${g.ctls
        .map((c) => {
          const id = `st-${c.key}`;
          const only = "only" in c && c.only ? ` data-only="${c.only}"` : "";
          if (c.type === "range")
            return `<div class="field"${only}><label for="${id}">${c.label}</label><div class="range-row wide"><input id="${id}" type="range" min="${c.min}" max="${c.max}" step="${c.step}" data-key="${c.key}" /><output for="${id}" class="num" id="${id}-out"></output></div></div>`;
          if (c.type === "select")
            return `<div class="field"${only}><label for="${id}">${c.label}</label><select id="${id}" data-key="${c.key}">${c.options.map(([v, l]) => `<option value="${v}">${l}</option>`).join("")}</select></div>`;
          if (c.type === "check") return `<label class="check"${only}><input id="${id}" type="checkbox" data-key="${c.key}" /> <span>${c.label}</span></label>`;
          return `<div class="field colour-row"><label for="${id}">${c.label}</label><input id="${id}" type="color" data-key="${c.key}" /></div>`;
        })
        .join("")}${g.title === "Look" ? `<button type="button" class="btn small" id="themeColours">Use the theme's colours</button>` : ""}</div></details>`,
    ).join("") +
    `<details class="group"><summary>Save and share</summary><div class="group-body share">
      <p class="hint">Settings are kept for each model in this browser.</p>
      <div class="btn-row"><button type="button" class="btn small" id="copyLink">Copy link with this look</button><button type="button" class="btn small" id="resetLook">Reset to defaults</button></div>
      <div class="btn-row"><button type="button" class="btn small" id="exportLook">Export JSON</button><label class="btn small file-mini">Import JSON<input type="file" accept=".json,application/json" id="importLook" /></label></div>
      <p class="hint" id="shareNote" aria-live="polite"></p>
    </div></details>`;
  host.addEventListener("input", (e) => {
    const el = e.target as HTMLInputElement;
    const key = el.dataset.key as keyof Style | undefined;
    if (!key) return;
    const cur = S.style[key];
    const v = el.type === "checkbox" ? el.checked : typeof cur === "number" ? Number(el.value) : el.value;
    (S.style as unknown as Record<string, unknown>)[key] = v;
    if (key === "edgeMin" && S.style.edgeMax < S.style.edgeMin) S.style.edgeMax = S.style.edgeMin;
    if (key === "edgeMax" && S.style.edgeMin > S.style.edgeMax) S.style.edgeMin = S.style.edgeMax;
    syncStyleUI();
    styleChanged();
  });
  $("themeColours").addEventListener("click", () => {
    S.style.pos = S.style.neg = S.style.bg = "";
    syncStyleUI();
    styleChanged();
  });
  $("resetLook").addEventListener("click", () => {
    if (S.model) clearStored(modelKey());
    const d = clone(DEFAULT_SETTINGS);
    d.showFrame = S.modelName.startsWith("lunar-") && !!S.rec?.get("frames");
    applySettings(d);
    note("Back to the defaults.");
  });
  $("copyLink").addEventListener("click", async () => {
    const h = toHash(collectSettings());
    history.replaceState(null, "", location.pathname + location.search + h);
    try {
      await navigator.clipboard.writeText(location.href);
      note("Link copied. It carries the look, not the model or recording.");
    } catch {
      note("Link is in the address bar. It carries the look, not the model or recording.");
    }
  });
  $("exportLook").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(collectSettings(), null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "netglow-settings.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
  $("importLook").addEventListener("change", async (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    (e.target as HTMLInputElement).value = "";
    if (!f) return;
    try {
      applySettings(merge(collectSettings(), JSON.parse(await f.text())));
      note(`Applied ${f.name}.`);
    } catch {
      note(`${f.name} is not a settings file.`);
    }
  });
}

function note(msg: string) {
  $("shareNote").textContent = msg;
}

/** reflect S.style in the controls, and show only the rows that apply to this model */
function syncStyleUI() {
  const hasImage = !!S.viz?.imageInput;
  const hasFrames = !!S.rec?.get("frames");
  const hasMaps = !!S.viz?.cols.some((c) => c.maps);
  document.querySelectorAll<HTMLElement>("#styleGroups [data-only]").forEach((el) => {
    const o = el.dataset.only;
    el.hidden = o === "image" ? !hasImage : o === "frames" ? !hasFrames : !hasMaps;
  });
  for (const g of GROUPS)
    for (const c of g.ctls) {
      const el = $<HTMLInputElement>(`st-${c.key}`);
      const v = S.style[c.key];
      if (c.type === "check") el.checked = !!v;
      else if (c.type === "color") el.value = (v as string) || themeColour(c.key as "pos" | "neg" | "bg");
      else el.value = String(v);
      if (c.type === "range") $(`st-${c.key}-out`).textContent = `${+Number(v).toFixed(2)}${c.unit === "px" ? "" : c.unit ?? ""}`;
    }
}

function styleChanged() {
  if (S.renderer) {
    S.renderer.opt.style = S.style;
    S.renderer.opt.saliency = ensureSaliency();
    renderLegend();
    renderEmbed();
    if (S.renderer instanceof CompareRenderer && S.cmp) S.renderer.b.opt.saliency = cmpSaliency(S.cmp);
    redraw(0);
  }
  saveSoon();
}

function modelKey() {
  return `${S.modelName}:${S.model?._bytes.length ?? 0}`;
}

function collectSettings(): Settings {
  return {
    nodes: S.nodes,
    topK: S.topK,
    theme: S.theme.id,
    mode: S.modeTouched ? S.mode : null,
    speed: S.speed,
    showFrame: S.showFrame,
    transparent: S.transparent,
    showMaps: S.showMaps,
    img: { ...S.img },
    format: ($("format") as HTMLSelectElement).value as Settings["format"],
    fps: Number(($("fps") as HTMLSelectElement).value),
    size: ($("size") as HTMLSelectElement).value,
    style: { ...S.style },
  };
}

/** set every control and state field from a settings object, then redraw (and re-select nodes if needed) */
function applySettings(st: Settings, quiet = false) {
  const needViz = st.nodes !== S.nodes || (st.mode ?? null) !== (S.modeTouched ? S.mode : null);
  S.nodes = st.nodes;
  S.topK = st.topK;
  S.theme = THEMES.find((t) => t.id === st.theme) ?? THEMES[0];
  S.speed = st.speed;
  S.showFrame = st.showFrame;
  S.transparent = st.transparent;
  S.showMaps = st.showMaps;
  S.img = { ...st.img };
  S.style = { ...st.style };
  if (st.mode) {
    S.mode = st.mode;
    S.modeTouched = true;
  } else S.modeTouched = false;
  const set = (id: string, v: string | boolean) => {
    const el = $<HTMLInputElement>(id);
    if (typeof v === "boolean") el.checked = v;
    else el.value = v;
  };
  set("nodes", String(st.nodes));
  $("nodesOut").textContent = String(st.nodes);
  set("topk", String(st.topK));
  $("topkOut").textContent = String(st.topK);
  themeSeg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String((x as HTMLElement).dataset.theme === S.theme.id)));
  if (st.mode) set("mode", st.mode);
  set("speed", String(st.speed));
  set("showFrame", st.showFrame);
  set("transparent", st.transparent);
  set("showMaps", st.showMaps);
  set("pxRange", st.img.range);
  set("pxLayout", st.img.layout);
  set("pxNewest", st.img.newest);
  set("format", st.format);
  set("fps", String(st.fps));
  set("size", st.size);
  $("exportBtn").textContent = `Export ${FORMAT_LABEL[st.format]}`;
  syncStyleUI();
  if (quiet) return;
  if (needViz && S.trace) rebuild(false).catch(fail);
  else if (S.viz) {
    makeRenderer();
    syncFrameSection();
    drawRibbon();
    redraw(0);
  }
  saveSoon();
}

/** after a model loads: a share link wins, then what this browser stored for the model */
function applySaved() {
  const src = S.pendingHash ?? (S.model ? loadStored(modelKey()) : null);
  S.pendingHash = null;
  if (src) applySettings(merge(collectSettings(), src), true);
}

let saveTimer = 0;
function saveSoon() {
  clearTimeout(saveTimer);
  saveTimer = window.setTimeout(() => {
    if (S.model) store(modelKey(), collectSettings());
  }, 300);
}

buildStyleUI();
syncStyleUI();
// any other control in the settings panel persists too
$("settingsPanel").addEventListener("change", saveSoon);
themeSeg.addEventListener("click", () => {
  syncStyleUI();
  saveSoon();
});
S.pendingHash = fromHash(location.hash);
void diff;
