// .npy / .npz / .json readers for recordings. No dependency: .npz is a zip, inflated with the browser's
// built-in DecompressionStream("deflate-raw").
//
// Vocabulary:
//   .npy  - NumPy's single-array file: magic "\x93NUMPY", a header dict ({'descr','fortran_order','shape'}), raw data
//   .npz  - a zip of .npy files, one per key (np.savez / np.savez_compressed)
//   descr - NumPy dtype string, e.g. "<f4" (little-endian float32), "|u1" (uint8), "<U12" (unicode, 12 chars)

export interface NdArray {
  dtype: string; // normalised: f4 f8 i1 i2 i4 i8 u1 u2 u4 u8 b1 U
  shape: number[];
  data: Float32Array | Float64Array | Int8Array | Int16Array | Int32Array | BigInt64Array | Uint8Array | Uint16Array | Uint32Array | BigUint64Array;
  strings?: string[]; // for unicode arrays
}

export function parseNpy(buf: Uint8Array): NdArray {
  if (buf[0] !== 0x93 || String.fromCharCode(...buf.subarray(1, 6)) !== "NUMPY") throw new Error("Not a .npy file");
  const major = buf[6];
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const hstart = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(buf.subarray(hstart, hstart + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*True/.test(header);
  const shapeStr = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? "";
  if (!descr) throw new Error("Bad .npy header");
  if (fortran) throw new Error("Fortran-ordered arrays are not supported; save with np.ascontiguousarray.");
  if (descr.startsWith("|O")) throw new Error("Object arrays (pickles) are not supported; save plain numeric arrays.");
  const shape = shapeStr.split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const n = shape.reduce((a, b) => a * b, 1);
  const off = hstart + hlen;
  const body = buf.slice(off); // copy -> aligned
  const ab = body.buffer;
  const endian = descr[0];
  if (endian === ">") throw new Error("Big-endian arrays are not supported.");
  const kind = descr.slice(1);
  let data: NdArray["data"];
  let dtype = kind;
  if (kind === "f4") data = new Float32Array(ab, 0, n);
  else if (kind === "f8") data = new Float64Array(ab, 0, n);
  else if (kind === "f2") {
    const h = new Uint16Array(ab, 0, n);
    const f = new Float32Array(n);
    for (let i = 0; i < n; i++) f[i] = half(h[i]);
    data = f;
    dtype = "f4";
  } else if (kind === "i1") data = new Int8Array(ab, 0, n);
  else if (kind === "i2") data = new Int16Array(ab, 0, n);
  else if (kind === "i4") data = new Int32Array(ab, 0, n);
  else if (kind === "i8") data = new BigInt64Array(ab, 0, n);
  else if (kind === "u1" || kind === "b1") data = new Uint8Array(ab, 0, n);
  else if (kind === "u2") data = new Uint16Array(ab, 0, n);
  else if (kind === "u4") data = new Uint32Array(ab, 0, n);
  else if (kind === "u8") data = new BigUint64Array(ab, 0, n);
  else if (kind.startsWith("U")) {
    const w = Number(kind.slice(1));
    const cps = new Uint32Array(ab, 0, n * w);
    const strings: string[] = [];
    for (let i = 0; i < n; i++) {
      let s = "";
      for (let j = 0; j < w; j++) {
        const c = cps[i * w + j];
        if (!c) break;
        s += String.fromCodePoint(c);
      }
      strings.push(s);
    }
    return { dtype: "U", shape, data: new Uint8Array(0), strings };
  } else throw new Error(`Unsupported dtype ${descr}`);
  return { dtype, shape, data };
}

function half(h: number) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

async function inflateRaw(data: Uint8Array): Promise<Uint8Array> {
  const ds = new DecompressionStream("deflate-raw");
  const stream = new Blob([data as BlobPart]).stream().pipeThrough(ds);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Reads a zip via its central directory (handles zip64 sizes, which numpy writes for big arrays). */
export async function parseZip(buf: Uint8Array): Promise<Map<string, Uint8Array>> {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let eocd = -1;
  for (let i = buf.length - 22; i >= Math.max(0, buf.length - 65557); i--) {
    if (dv.getUint32(i, true) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error("Not a zip / .npz file");
  let count = dv.getUint16(eocd + 10, true);
  let cdOff = dv.getUint32(eocd + 16, true);
  if (cdOff === 0xffffffff || count === 0xffff) {
    const loc = eocd - 20;
    if (dv.getUint32(loc, true) === 0x07064b50) {
      const z64 = Number(dv.getBigUint64(loc + 8, true));
      count = Number(dv.getBigUint64(z64 + 32, true));
      cdOff = Number(dv.getBigUint64(z64 + 48, true));
    }
  }
  const files = new Map<string, Uint8Array>();
  let p = cdOff;
  for (let k = 0; k < count; k++) {
    if (dv.getUint32(p, true) !== 0x02014b50) throw new Error("Corrupt zip central directory");
    const method = dv.getUint16(p + 10, true);
    let csize = dv.getUint32(p + 20, true);
    const nlen = dv.getUint16(p + 28, true);
    const xlen = dv.getUint16(p + 30, true);
    const clen = dv.getUint16(p + 32, true);
    let local = dv.getUint32(p + 42, true);
    const name = new TextDecoder().decode(buf.subarray(p + 46, p + 46 + nlen));
    // zip64 extra field: original size, compressed size, local header offset (only those set to 0xffffffff)
    let usize = dv.getUint32(p + 24, true);
    let x = p + 46 + nlen;
    const xend = x + xlen;
    while (x + 4 <= xend) {
      const id = dv.getUint16(x, true);
      const sz = dv.getUint16(x + 2, true);
      if (id === 0x0001) {
        let q = x + 4;
        if (usize === 0xffffffff) {
          usize = Number(dv.getBigUint64(q, true));
          q += 8;
        }
        if (csize === 0xffffffff) {
          csize = Number(dv.getBigUint64(q, true));
          q += 8;
        }
        if (local === 0xffffffff) local = Number(dv.getBigUint64(q, true));
      }
      x += 4 + sz;
    }
    const lnlen = dv.getUint16(local + 26, true);
    const lxlen = dv.getUint16(local + 28, true);
    const dataStart = local + 30 + lnlen + lxlen;
    const raw = buf.subarray(dataStart, dataStart + csize);
    if (method === 0) files.set(name, raw);
    else if (method === 8) files.set(name, await inflateRaw(raw));
    else throw new Error(`Unsupported zip compression method ${method} in ${name}`);
    p += 46 + nlen + xlen + clen;
  }
  return files;
}

export async function parseNpz(buf: Uint8Array): Promise<Map<string, NdArray>> {
  const files = await parseZip(buf);
  const out = new Map<string, NdArray>();
  for (const [name, bytes] of files) {
    if (!name.endsWith(".npy")) continue;
    out.set(name.slice(0, -4), parseNpy(bytes));
  }
  return out;
}

/** JSON recording: { "obs": [[...], ...], "actions": [...], ... } -> arrays (nested lists become shapes). */
export function parseJsonRecording(text: string): Map<string, NdArray> {
  const j = JSON.parse(text);
  const out = new Map<string, NdArray>();
  for (const [k, v] of Object.entries(j)) {
    if (typeof v === "number") {
      out.set(k, { dtype: "f8", shape: [], data: Float64Array.of(v) });
      continue;
    }
    if (!Array.isArray(v)) continue;
    const shape: number[] = [];
    let cur: unknown = v;
    while (Array.isArray(cur)) {
      shape.push(cur.length);
      cur = cur[0];
    }
    if (typeof cur === "string") {
      out.set(k, { dtype: "U", shape, data: new Uint8Array(0), strings: (v as unknown[]).flat(Infinity) as string[] });
      continue;
    }
    const flat = (v as unknown[]).flat(Infinity) as number[];
    const isInt = flat.every((x) => Number.isInteger(x));
    out.set(k, { dtype: isInt ? "i8" : "f8", shape, data: isInt ? BigInt64Array.from(flat.map(BigInt)) : Float64Array.from(flat) });
  }
  return out;
}

export function toFloat(a: NdArray): Float32Array {
  if (a.data instanceof Float32Array) return a.data;
  if (a.data instanceof BigInt64Array || a.data instanceof BigUint64Array) return Float32Array.from(a.data as ArrayLike<bigint>, Number);
  return Float32Array.from(a.data as ArrayLike<number>);
}
