// Runs the model over every tick of the recording with onnxruntime-web (WebAssembly, in this tab) and
// collects the intermediate tensors the plan asked for.
//
// Vocabulary:
//   batched run   - feed-forward models: 64 ticks per session.run, the ticks do not depend on each other
//   stepped run   - recurrent models: one tick per session.run, the state outputs of tick t are fed back as the
//                   state inputs of tick t+1; at an episode start the state goes back to its initial value
//   episode start - recording key "episode_starts" (1 on the first tick of each episode), or "dones" (1 on the
//                   last tick, so the next one starts); tick 0 always starts. Initial state: "init_<input name>"
//                   if the recording has it, else zeros

import * as ort from "onnxruntime-web/wasm";
import { ELEM, type ModelInfo, withExtraOutputs } from "./onnx/proto";
import type { Plan } from "./analyze";
import type { NdArray } from "./io/npy";

ort.env.wasm.numThreads = 1; // threads need cross-origin isolation, which plain static hosting lacks

export interface Trace {
  /** per tensor: per-tick dims (batch removed) and data for all ticks, tick-major */
  tensors: Map<string, { dims: number[]; size: number; data: Float32Array }>;
  T: number;
  ms: number;
  /** recurrent models: 1 where the state was reset to its initial value (always tick 0) */
  starts?: Uint8Array;
  /** recurrent models: the initial value of each state input, flat */
  init?: Map<string, Float32Array>;
}

/** episode starts of a recording: episode_starts (1 = first tick) or dones (1 = last tick); tick 0 always starts */
export function episodeStarts(rec: Map<string, NdArray> | null, T: number): Uint8Array {
  const s = new Uint8Array(T);
  s[0] = 1;
  const es = rec?.get("episode_starts") ?? rec?.get("starts");
  const dn = rec?.get("dones");
  if (es && es.shape[0] === T) for (let t = 0; t < T; t++) s[t] = s[t] || (Number(es.data[t]) ? 1 : 0);
  else if (dn && dn.shape[0] === T) for (let t = 0; t + 1 < T; t++) if (Number(dn.data[t])) s[t + 1] = 1;
  return s;
}

export function makeInput(elem: number, src: NdArray, start: number, count: number, size: number, shape: number[]): ort.Tensor {
  const n = count * size;
  const off = start * size;
  const d = src.data;
  const pick = <A extends { length: number; [i: number]: number | bigint }>(A: A) => {
    for (let i = 0; i < n; i++) (A as unknown as number[])[i] = d[off + i] as number;
    return A;
  };
  if (elem === ELEM.FLOAT) {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = Number(d[off + i]);
    return new ort.Tensor("float32", a, shape);
  }
  if (elem === ELEM.UINT8) return new ort.Tensor("uint8", pick(new Uint8Array(n)), shape);
  if (elem === ELEM.INT8) return new ort.Tensor("int8", pick(new Int8Array(n)), shape);
  if (elem === ELEM.INT32) return new ort.Tensor("int32", pick(new Int32Array(n)), shape);
  if (elem === ELEM.DOUBLE) {
    const a = new Float64Array(n);
    for (let i = 0; i < n; i++) a[i] = Number(d[off + i]);
    return new ort.Tensor("float64", a, shape);
  }
  if (elem === ELEM.INT64) {
    const a = new BigInt64Array(n);
    for (let i = 0; i < n; i++) a[i] = BigInt(Math.round(Number(d[off + i])));
    return new ort.Tensor("int64", a, shape);
  }
  throw new Error(`Unsupported input type ${elem}`);
}

/** obs: one recording array per model input, in plan.inputs order; rec: the recording (episode starts, initial state) */
export async function runTrace(model: ModelInfo, plan: Plan, obs: NdArray[], onProgress?: (f: number) => void, rec?: Map<string, NdArray>): Promise<Trace> {
  const t0 = performance.now();
  const bytes = withExtraOutputs(model, plan.fetch);
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "disabled" });
  // optimisation is disabled so fused nodes keep the tensor names we fetch
  const T = obs[0].shape[0];
  const fixed = (d: number | string | undefined) => (typeof d === "number" && d > 0 ? d : 0);
  const feedsOf = plan.inputs.map((inp, i) => {
    const o = obs[i];
    if (o.shape[0] !== T) throw new Error(`Recording arrays disagree on the tick count: ${o.shape[0]} for "${inp.name}" vs ${T}.`);
    const perTick = o.shape.slice(1);
    const size = perTick.reduce((a, b) => a * b, 1);
    const want = inp.shape.slice(1);
    const known = want.length > 0 && want.every((d) => fixed(d) > 0);
    const wantSize = known ? (want as number[]).reduce((a, b) => a * b, 1) : size;
    if (wantSize !== size)
      throw new Error(`Recording input "${inp.name}" has ${perTick.join("×")} values per tick but the model expects ${want.join("×")}.`);
    return { inp, o, size, tickShape: known ? (want as number[]) : perTick };
  });
  const fixedBatch = Math.max(0, ...plan.inputs.map((i) => fixed(i.shape[0])));
  const recurrent = plan.states.length > 0;
  const B = fixedBatch === 1 || recurrent ? 1 : fixedBatch > 1 ? -1 : 64;
  if (B < 0) throw new Error(`The model has a fixed batch size of ${fixedBatch}; export it with batch 1 or a dynamic batch axis.`);
  const tensors: Trace["tensors"] = new Map();
  // recurrent state: initial values, and the current value fed each tick
  const starts = recurrent ? episodeStarts(rec ?? null, T) : undefined;
  const init = new Map<string, Float32Array>();
  for (const st of plan.states) {
    const given = rec?.get(`init_${st.input}`);
    const v = new Float32Array(st.size);
    if (given) {
      if (given.data.length !== st.size) throw new Error(`Recording "init_${st.input}" has ${given.data.length} values; the state input "${st.input}" takes ${st.size}.`);
      for (let i = 0; i < st.size; i++) v[i] = Number(given.data[i]);
    }
    init.set(st.input, v);
  }
  const cur = new Map<string, Float32Array>();
  for (let s = 0; s < T; s += B) {
    const c = Math.min(B, T - s);
    const feeds: Record<string, ort.Tensor> = Object.fromEntries(feedsOf.map((f) => [f.inp.name, makeInput(f.inp.elemType, f.o, s, c, f.size, [c, ...f.tickShape])]));
    for (const st of plan.states) {
      if (starts![s] || !cur.has(st.input)) cur.set(st.input, init.get(st.input)!.slice());
      feeds[st.input] = new ort.Tensor("float32", cur.get(st.input)!, st.dims);
    }
    const res = await session.run(feeds, plan.fetch);
    for (const st of plan.states) cur.set(st.input, Float32Array.from(res[st.output].data as Float32Array));
    for (const name of plan.fetch) {
      const r = res[name];
      if (!r) throw new Error(`onnxruntime did not return tensor "${name}".`);
      let e = tensors.get(name);
      if (!e) {
        const dims = plan.tickDims.get(name) ?? r.dims.slice(1).map(Number);
        const sz = dims.reduce((a, b) => a * b, 1);
        if (sz * c !== r.data.length) throw new Error(`Tensor "${name}" has ${r.data.length / c} values per tick, expected ${sz}.`);
        e = { dims, size: sz, data: new Float32Array(T * sz) };
        tensors.set(name, e);
      }
      const src = r.data as ArrayLike<number | bigint>;
      const dst = e.data;
      const off = s * e.size;
      for (let i = 0; i < c * e.size; i++) dst[off + i] = Number(src[i]);
    }
    onProgress?.(Math.min(1, (s + c) / T));
    if (s % (B * (recurrent ? 64 : 8)) === 0) await new Promise((r) => setTimeout(r, 0));
  }
  await session.release();
  return { tensors, T, ms: performance.now() - t0, starts, init: recurrent ? init : undefined };
}
