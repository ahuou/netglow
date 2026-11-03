// Live mode: a Python process (netglow.live / Recorder(live=True)) streams the model once and then every tick's
// inputs over a WebSocket; this tab runs the model on each tick as it arrives and animates a rolling window.
//
// Vocabulary:
//   tick     one decision step: the inputs the policy saw (plus the outputs / action it logged, for the match check)
//   window   the last few seconds of ticks; normalisation, node selection and the match check use only these
//   lag      the view trails the newest tick by a fraction of a second so playback stays smooth between updates
//
// Wire format (binary messages): u32 LE header length, UTF-8 JSON header, then the raw array bytes back to back.
//   {"type":"model","fps":50,"labels":{...},"inputs":[...],"size":n}          payload: the .onnx file
//   {"type":"tick","i":17,"arrays":[{"key":"obs_obs","dtype":"<f4","shape":[8],"size":32}, ...]}
//
// The model runs incrementally (each tick once, on its own onnxruntime session); every update hands the window to
// main.ts as an ordinary recording + trace and asks it to rebuild the picture, so nothing else needs to know about live.

import * as ort from "onnxruntime-web/wasm";
import { withExtraOutputs, type ModelInfo } from "./onnx/proto";
import type { Plan } from "./analyze";
import { makeInput, type Trace } from "./run";
import type { NdArray } from "./io/npy";
import { writeNpz } from "./io/npz-write";
import "./live.css";

export interface LiveHost {
  S: {
    model: ModelInfo | null;
    plan: Plan | null;
    rec: Map<string, NdArray> | null;
    recName: string;
    obs: NdArray | null;
    obsList: NdArray[];
    trace: Trace | null;
    viz: { T: number } | null;
    labelsFile: unknown;
    t: number;
    hz: number;
    playing: boolean;
    showFrame: boolean;
    mode: string;
    modeTouched: boolean;
    iconFiles: Map<string, unknown>;
    iconOverride: Map<number, unknown>;
    nameOverride: { inputs: Map<number, string>; outputs: Map<number, string> };
    /** compare mode's second model (main.ts); its trace is dropped so it reruns on each window */
    cmp: { trace: unknown } | null;
  };
  loadModel(bytes: Uint8Array, name: string): Promise<void>;
  rebuild(rerun: boolean): Promise<void>;
  redraw(dt: number): void;
  stop(): void;
  status(msg: string, error?: boolean): void;
  markFile(kind: string, name: string): void;
}

interface Tick {
  i: number;
  arrays: Map<string, NdArray>;
  bytes: number;
  trace?: Map<string, Float32Array>;
  /** per fetched tensor: its per-tick dims (batch removed) */
  dims?: Map<string, number[]>;
  /** recurrent models: the state fed into this tick, per state input */
  stateIn?: Map<string, Float32Array>;
}

const KIND: Record<string, string> = { f4: "f4", f8: "f8", i1: "i1", i2: "i2", i4: "i4", i8: "i8", u1: "u1", b1: "u1", u2: "u2", u4: "u4", u8: "u8" };
const CTOR: Record<string, new (b: ArrayBuffer) => NdArray["data"]> = {
  f4: Float32Array, f8: Float64Array, i1: Int8Array, i2: Int16Array, i4: Int32Array, i8: BigInt64Array, u1: Uint8Array, u2: Uint16Array, u4: Uint32Array, u8: BigUint64Array,
};
const RECORD_CAP = 512 * 2 ** 20; // bytes of streamed ticks kept for "Record"

function decode(buf: ArrayBuffer) {
  const dv = new DataView(buf);
  const hlen = dv.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
  return { header, body: new Uint8Array(buf, 4 + hlen) };
}

function decodeTick(header: { i: number; arrays: { key: string; dtype: string; shape: number[]; size: number }[] }, body: Uint8Array): Tick {
  const arrays = new Map<string, NdArray>();
  let off = 0;
  for (const a of header.arrays) {
    const kind = KIND[a.dtype.slice(1)];
    if (!kind) throw new Error(`Live stream: unsupported dtype ${a.dtype} for ${a.key}`);
    const raw = body.slice(off, off + a.size).buffer; // copy: typed arrays need an aligned buffer
    arrays.set(a.key, { dtype: kind, shape: a.shape, data: new CTOR[kind](raw) });
    off += a.size;
  }
  return { i: header.i, arrays, bytes: off };
}

/** per-tick arrays (same shape) -> one array with a leading tick axis */
function stack(parts: NdArray[]): NdArray {
  const a0 = parts[0];
  const n = a0.data.length;
  const C = CTOR[a0.dtype];
  const out = new C(new ArrayBuffer(n * parts.length * a0.data.BYTES_PER_ELEMENT)) as NdArray["data"];
  parts.forEach((p, k) => (out as unknown as { set(a: ArrayLike<number | bigint>, o: number): void }).set(p.data as ArrayLike<number | bigint>, k * n));
  return { dtype: a0.dtype, shape: [parts.length, ...a0.shape], data: out };
}

export function initLive(host: LiveHost) {
  const S = host.S;
  // ---------- UI (built here so index.html stays untouched) ----------
  const sec = document.createElement("section");
  sec.id = "liveSection";
  sec.innerHTML = `
    <h2>Live</h2>
    <form class="live-form" id="liveForm">
      <input id="liveUrl" type="text" value="ws://localhost:8765" aria-label="Live stream address" spellcheck="false" autocomplete="off" />
      <button class="btn small" id="liveBtn" type="submit">Connect live</button>
    </form>
    <p class="hint" id="liveInfo">Stream ticks from Python with <code>netglow.live(model)</code>; this tab runs the model on each one.</p>
    <div class="btn-row" id="liveRecRow" hidden>
      <button class="btn small" id="liveRec" type="button" aria-pressed="false">Record</button>
      <span class="live-saved" id="liveSaved"></span>
    </div>`;
  const anchor = document.getElementById("demoList")?.closest("section");
  if (anchor) anchor.after(sec);
  else document.querySelector(".panel.left")?.appendChild(sec);
  const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
  const urlIn = $<HTMLInputElement>("liveUrl");
  const btn = $<HTMLButtonElement>("liveBtn");
  const info = $("liveInfo");
  const recBtn = $<HTMLButtonElement>("liveRec");

  // ---------- state ----------
  let ws: WebSocket | null = null;
  let want = false; // user asked to be connected (reconnect on drops)
  let retry = 0;
  let inbox: Promise<void> = Promise.resolve();
  let session: ort.InferenceSession | null = null;
  let modelBytes: Uint8Array | null = null;
  let plan: Plan | null = null;
  let hz = 30;
  let fpsFromModel = false;
  let pending: Tick[] = [];
  const state = new Map<string, Float32Array>(); // recurrent state carried from tick to tick
  let win: Tick[] = [];
  let W = 300;
  let newest = -1;
  let first = true;
  let busy = false;
  let lastUpdate = 0;
  let interval = 120;
  let g = 0; // displayed position, in global ticks
  let raf = 0;
  let lastTs = 0;
  let arrivals: number[] = [];
  let recording = false;
  let recTicks: Tick[] = [];
  let recBytes = 0;
  let recFull = false;

  const resolve = (v: string) => {
    v = v.trim();
    if (/^\d+$/.test(v)) return `ws://localhost:${v}`;
    if (v === "here" || v === "") return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}`;
    return v;
  };
  const setInfo = (s: string) => (info.textContent = s);

  function connect(url: string) {
    want = true;
    btn.textContent = "Disconnect";
    urlIn.value = url;
    open(url);
  }
  function open(url: string) {
    setInfo(`Connecting to ${url}…`);
    let sock: WebSocket;
    try {
      sock = new WebSocket(url);
    } catch (e) {
      setInfo(`Could not open ${url}: ${e instanceof Error ? e.message : e}`);
      return;
    }
    ws = sock;
    sock.binaryType = "arraybuffer";
    sock.onopen = () => {
      retry = 0;
      setInfo(`Connected to ${url}. Waiting for the model…`);
    };
    sock.onmessage = (e) => {
      const msg = e.data as ArrayBuffer;
      inbox = inbox.then(() => handle(msg)).catch((err) => {
        console.error(err);
        host.status(err instanceof Error ? err.message : String(err), true);
      });
    };
    sock.onclose = () => {
      if (ws !== sock) return;
      ws = null;
      if (!want) return;
      retry++;
      const https = location.protocol === "https:" && /^ws:\/\/(localhost|127\.0\.0\.1|\[::1\])/.test(url);
      setInfo(
        `No stream at ${url} yet; retrying.` +
          (https && retry > 1 ? " If Python is running, this browser may block https → localhost: allow local access when asked, or open http://localhost:PORT/ that netglow prints." : ""),
      );
      setTimeout(() => want && !ws && open(url), Math.min(3000, 500 * retry));
    };
  }
  async function disconnect() {
    want = false;
    btn.textContent = "Connect live";
    ws?.close();
    ws = null;
    if (recording) finishRecording();
    $("liveRecRow").hidden = true;
    await inbox;
    while (busy) await new Promise((r) => setTimeout(r, 20)); // let a running update finish
    cancelAnimationFrame(raf);
    raf = 0;
    if (S.rec && S.model && win.length) {
      // the last window stays loaded as an ordinary recording: scrub, play and export it
      S.trace = null;
      await host.rebuild(true);
      setInfo(`Disconnected. The last ${win.length} ticks stay loaded; play or export them.`);
    } else setInfo("Disconnected.");
  }

  async function handle(buf: ArrayBuffer) {
    if (!want) return; // messages still queued after Disconnect
    const { header, body } = decode(buf);
    if (header.type === "model") await onModel(header, body.slice(0, header.size));
    else if (header.type === "tick" && session) onTick(decodeTick(header, body));
  }

  async function onModel(h: { fps?: number | null; labels?: Record<string, unknown> }, bytes: Uint8Array) {
    host.stop();
    cancelAnimationFrame(raf);
    raf = 0;
    await session?.release();
    session = null;
    pending = [];
    win = [];
    state.clear();
    first = true;
    newest = -1;
    arrivals = [];
    const labels = { ...(h.labels ?? {}) };
    delete labels.icons; // icon paths can't be resolved from a stream
    S.labelsFile = labels;
    S.iconFiles.clear();
    S.iconOverride.clear();
    S.nameOverride.inputs.clear();
    S.nameOverride.outputs.clear();
    S.rec = null;
    S.trace = null;
    S.viz = null;
    modelBytes = bytes;
    await host.loadModel(bytes, "live · model.onnx");
    plan = S.plan!;
    session = await ort.InferenceSession.create(withExtraOutputs(S.model!, plan.fetch), { executionProviders: ["wasm"], graphOptimizationLevel: "disabled" });
    fpsFromModel = !!h.fps;
    hz = h.fps || 30;
    W = Math.max(30, Math.min(600, Math.round(hz * 10)));
    $("liveRecRow").hidden = false;
    host.markFile("recording", "live stream");
    setInfo(`Model received (${plan.layers.length} weighted layers). Waiting for ticks…`);
  }

  function onTick(t: Tick) {
    pending.push(t);
    newest = Math.max(newest, t.i);
    const now = performance.now();
    arrivals.push(now);
    if (arrivals.length > 60) arrivals.shift();
    if (!fpsFromModel && arrivals.length > 10) {
      const span = (arrivals[arrivals.length - 1] - arrivals[0]) / 1000;
      if (span > 0) hz = (arrivals.length - 1) / span;
    }
    if (recording) {
      if (recBytes + t.bytes <= RECORD_CAP) {
        recTicks.push(t);
        recBytes += t.bytes;
      } else recFull = true;
      recBtn.textContent = `Stop and save (${recTicks.length}${recFull ? ", full" : ""})`;
    }
    schedule();
  }

  /** ticks -> recording arrays: stacked per key; per-tick state hand-overs become one init_<input> (the state fed into
   *  the first tick, so the slice replays exactly) and episode_starts gets a 1 on its first tick */
  function asRecording(ticks: Tick[]): Map<string, NdArray> {
    const rec = new Map<string, NdArray>();
    for (const key of ticks[ticks.length - 1].arrays.keys()) {
      if (key.startsWith("init_")) continue;
      const parts = ticks.map((t) => t.arrays.get(key));
      if (parts.every((x) => x && x.data.length === parts[0]!.data.length)) rec.set(key, stack(parts as NdArray[]));
    }
    if (plan?.states.length) {
      const st = new Uint8Array(ticks.length);
      ticks.forEach((t, k) => (st[k] = k === 0 || Number(t.arrays.get("episode_starts")?.data[0] ?? 0) ? 1 : 0));
      rec.set("episode_starts", { dtype: "u1", shape: [ticks.length], data: st });
      for (const s of plan.states) {
        const v = ticks[0].stateIn?.get(s.input);
        if (v) rec.set(`init_${s.input}`, { dtype: "f4", shape: [v.length], data: v.slice() });
      }
    }
    rec.set("fps", { dtype: "f8", shape: [], data: Float64Array.of(hz) });
    return rec;
  }

  function schedule() {
    if (busy) return;
    const wait = Math.max(0, lastUpdate + interval - performance.now());
    busy = true;
    setTimeout(() => void update().finally(() => (busy = false)), wait);
  }

  async function runTicks(ticks: Tick[]) {
    const p = plan!;
    const fixed = (d: number | string | undefined) => (typeof d === "number" && d > 0 ? d : 0);
    const B = Math.max(0, ...p.inputs.map((i) => fixed(i.shape[0]))) === 1 || p.states.length ? 1 : 64;
    for (let s = 0; s < ticks.length; s += B) {
      const chunk = ticks.slice(s, s + B);
      const feeds: Record<string, ort.Tensor> = {};
      for (const inp of p.inputs) {
        const key = `obs_${inp.name}`;
        const parts = chunk.map((t) => t.arrays.get(key) ?? (p.inputs.length === 1 ? t.arrays.get("obs") : undefined));
        if (parts.some((x) => !x)) throw new Error(`Live tick is missing "${key}" (got ${[...chunk[0].arrays.keys()].join(", ")}).`);
        const nd = stack(parts as NdArray[]);
        const want = inp.shape.slice(1);
        const known = want.length > 0 && want.every((d) => fixed(d) > 0);
        const tickShape = known ? (want as number[]) : nd.shape.slice(1);
        const size = nd.shape.slice(1).reduce((a, b) => a * b, 1);
        feeds[inp.name] = makeInput(inp.elemType, nd, 0, chunk.length, size, [chunk.length, ...tickShape]);
      }
      // recurrent: one tick per run; the state goes back to its initial value at an episode start. A viewer that joins
      // mid-episode takes the state Python fed into that tick (init_<input>), else starts from zeros.
      const t = chunk[0];
      for (const st of p.states) {
        const sent = t.arrays.get(`init_${st.input}`);
        const isStart = !!Number(t.arrays.get("episode_starts")?.data[0] ?? 0);
        if (isStart || !state.has(st.input)) {
          const v = new Float32Array(st.size);
          if (sent && sent.data.length === st.size) for (let i = 0; i < st.size; i++) v[i] = Number(sent.data[i]);
          state.set(st.input, v);
        }
        (t.stateIn ??= new Map()).set(st.input, state.get(st.input)!);
        feeds[st.input] = new ort.Tensor("float32", state.get(st.input)!.slice(), st.dims);
      }
      const res = await session!.run(feeds, p.fetch);
      for (const st of p.states) state.set(st.input, Float32Array.from(res[st.output].data as Float32Array));
      for (const name of p.fetch) {
        const r = res[name];
        const per = r.data.length / chunk.length;
        const src = r.data as ArrayLike<number | bigint>;
        chunk.forEach((t, k) => {
          const a = new Float32Array(per);
          for (let j = 0; j < per; j++) a[j] = Number(src[k * per + j]);
          (t.trace ??= new Map()).set(name, a);
          (t.dims ??= new Map()).set(name, p.tickDims.get(name) ?? r.dims.slice(1).map(Number));
        });
      }
    }
  }

  async function update() {
    if (!want || !session || !plan || !pending.length) return;
    const t0 = performance.now();
    const batch = pending;
    pending = [];
    await runTicks(batch);
    win.push(...batch);
    if (win.length > W) win = win.slice(win.length - W);
    // the window as an ordinary recording + trace
    const rec = new Map<string, NdArray>();
    for (const [k, v] of asRecording(win)) rec.set(k, v);
    const tensors: Trace["tensors"] = new Map();
    const dims0 = win[0].dims!;
    for (const name of plan.fetch) {
      const per = win[0].trace!.get(name)!.length;
      const data = new Float32Array(per * win.length);
      win.forEach((t, k) => data.set(t.trace!.get(name)!, k * per));
      tensors.set(name, { dims: dims0.get(name)!, size: per, data });
    }
    S.rec = rec;
    S.recName = "live stream";
    S.obsList = plan.inputs.map((i) => rec.get(`obs_${i.name}`) ?? rec.get("obs")!);
    S.obs = S.obsList[Math.max(0, plan.inputs.findIndex((i) => i.isImage))];
    S.trace = { tensors, T: win.length, ms: performance.now() - t0 };
    if (plan.states.length) {
      // the window replays exactly: it starts from the state its first tick was fed (trace.init / init_<input>)
      S.trace.starts = rec.get("episode_starts")!.data as Uint8Array;
      S.trace.init = new Map(plan.states.map((st) => [st.input, win[0].stateIn!.get(st.input)!]));
    }
    if (S.cmp) S.cmp.trace = null; // compare: the second model reruns on each new window
    S.hz = hz;
    if (first) {
      first = false;
      if (!S.modeTouched) {
        const lf = (S.labelsFile as { output_mode?: string } | null)?.output_mode;
        const acts = rec.get("actions");
        const nOut = plan.layers[plan.layers.length - 1].out;
        S.mode = lf ?? (plan.softmaxInGraph ? "categorical" : acts && acts.dtype.startsWith("f") && acts.shape.length > 1 ? "continuous" : nOut > 1 ? "categorical" : "continuous");
        const sel = document.getElementById("mode") as HTMLSelectElement | null;
        if (sel) sel.value = S.mode;
      }
      S.showFrame = rec.has("frames");
      const cb = document.getElementById("showFrame") as HTMLInputElement | null;
      if (cb) cb.checked = S.showFrame;
      g = newest - lagTicks();
    }
    const hzIn = document.getElementById("hz") as HTMLInputElement | null;
    if (hzIn && document.activeElement !== hzIn) hzIn.value = String(+hz.toFixed(2));
    await host.rebuild(false);
    const cost = performance.now() - t0;
    interval = Math.max(80, Math.min(600, cost * 3));
    lastUpdate = performance.now();
    setInfo(`Live: tick ${newest} · ${hz.toFixed(hz < 10 ? 1 : 0)} Hz · last ${win.length} ticks in view · update ${Math.round(cost)} ms`);
    if (!raf) {
      lastTs = 0;
      raf = requestAnimationFrame(frame);
    }
    if (pending.length) setTimeout(schedule, 0);
  }

  const lagTicks = () => hz * (interval / 1000 + 0.15);

  function frame(ts: number) {
    raf = requestAnimationFrame(frame);
    if (S.playing) host.stop(); // the live clock drives playback, not the file player
    const dt = lastTs ? Math.min(0.1, (ts - lastTs) / 1000) : 0;
    lastTs = ts;
    if (!win.length || !S.viz) return;
    const lo = win[0].i;
    const hi = win[win.length - 1].i;
    const target = newest - lagTicks();
    g += dt * hz;
    g += (target - g) * Math.min(1, dt * 2); // drift back to the target lag when ticks come early or late
    if (Math.abs(target - g) > hz * 2) g = target;
    g = Math.max(lo, Math.min(hi, g));
    S.t = Math.max(0, (g - lo) / hz);
    host.redraw(dt);
  }

  // ---------- record what was streamed ----------
  function finishRecording() {
    recording = false;
    recBtn.textContent = "Record";
    recBtn.setAttribute("aria-pressed", "false");
    if (!recTicks.length) return;
    const blob = writeNpz(asRecording(recTicks));
    const n = recTicks.length;
    recTicks = [];
    recBytes = 0;
    recFull = false;
    download(blob, "recording.npz");
    const saved = $("liveSaved");
    saved.textContent = `Saved ${n} ticks as recording.npz. `;
    if (modelBytes) {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([modelBytes as BlobPart]));
      a.download = "model.onnx";
      a.textContent = "model.onnx";
      saved.append("With its model: ", a);
    }
  }
  function download(blob: Blob, name: string) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
  }

  $<HTMLFormElement>("liveForm").addEventListener("submit", (e) => {
    e.preventDefault();
    if (want) void disconnect();
    else connect(resolve(urlIn.value));
  });
  recBtn.addEventListener("click", () => {
    if (recording) return finishRecording();
    recording = true;
    recTicks = [];
    recBytes = 0;
    recBtn.setAttribute("aria-pressed", "true");
    recBtn.textContent = "Stop and save (0)";
    $("liveSaved").textContent = "";
  });

  const q = new URLSearchParams(location.search).get("live");
  if (q !== null) connect(resolve(q));
  return { connect: (u: string) => connect(resolve(u)), disconnect, get window() { return win.length; }, get newest() { return newest; } };
}
