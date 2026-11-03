// Writing .npz files (an uncompressed zip of .npy files) so a live stream can be saved as a normal recording.
//
// Vocabulary:
//   stored  - zip method 0: the bytes go in as they are (no deflate); numpy reads it like np.savez
//   CRC-32  - the checksum every zip entry carries; computed here with the usual 256-entry table

import type { NdArray } from "./npy";

let table: Uint32Array | null = null;
function crc32(b: Uint8Array): number {
  if (!table) {
    table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      table[n] = c >>> 0;
    }
  }
  let c = 0xffffffff;
  for (let i = 0; i < b.length; i++) c = table[(c ^ b[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const DESCR: Record<string, string> = { f4: "<f4", f8: "<f8", i1: "|i1", i2: "<i2", i4: "<i4", i8: "<i8", u1: "|u1", u2: "<u2", u4: "<u4", u8: "<u8", b1: "|b1" };

/** one array as .npy bytes (format 1.0, C order) */
export function npyBytes(a: NdArray): Uint8Array {
  const descr = DESCR[a.dtype];
  if (!descr) throw new Error(`Cannot write dtype ${a.dtype}`);
  const shape = a.shape.length === 1 ? `(${a.shape[0]},)` : `(${a.shape.join(", ")})`;
  let h = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shape}, }`;
  const pad = 64 - ((10 + h.length + 1) % 64);
  h += " ".repeat(pad % 64) + "\n";
  const hb = new TextEncoder().encode(h);
  const body = new Uint8Array(a.data.buffer, a.data.byteOffset, a.data.byteLength);
  const out = new Uint8Array(10 + hb.length + body.length);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, hb.length & 0xff, hb.length >> 8]);
  out.set(hb, 10);
  out.set(body, 10 + hb.length);
  return out;
}

/** arrays -> .npz (stored zip). Keys become "<key>.npy". Total size must stay under 4 GB (no zip64). */
export function writeNpz(arrays: Map<string, NdArray>): Blob {
  const parts: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let off = 0;
  for (const [key, a] of arrays) {
    const name = new TextEncoder().encode(`${key}.npy`);
    const data = npyBytes(a);
    const crc = crc32(data);
    const local = new Uint8Array(30 + name.length);
    const lv = new DataView(local.buffer);
    lv.setUint32(0, 0x04034b50, true);
    lv.setUint16(4, 20, true);
    lv.setUint32(14, crc, true);
    lv.setUint32(18, data.length, true);
    lv.setUint32(22, data.length, true);
    lv.setUint16(26, name.length, true);
    local.set(name, 30);
    const cd = new Uint8Array(46 + name.length);
    const cv = new DataView(cd.buffer);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(4, 20, true);
    cv.setUint16(6, 20, true);
    cv.setUint32(16, crc, true);
    cv.setUint32(20, data.length, true);
    cv.setUint32(24, data.length, true);
    cv.setUint16(28, name.length, true);
    cv.setUint32(42, off, true);
    cd.set(name, 46);
    parts.push(local, data);
    central.push(cd);
    off += local.length + data.length;
  }
  const cdSize = central.reduce((s, c) => s + c.length, 0);
  const end = new Uint8Array(22);
  const ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, central.length, true);
  ev.setUint16(10, central.length, true);
  ev.setUint32(12, cdSize, true);
  ev.setUint32(16, off, true);
  return new Blob([...parts, ...central, end] as BlobPart[], { type: "application/octet-stream" });
}
