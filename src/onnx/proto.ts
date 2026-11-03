// Minimal ONNX protobuf reader and writer, enough to (1) read the graph (nodes, weights, inputs, outputs)
// and (2) append extra graph outputs so onnxruntime returns intermediate tensors.
//
// Vocabulary:
//   varint      - protobuf's variable-length integer encoding (7 bits per byte, high bit = "more follows")
//   wire type   - how a field is encoded: 0 varint, 1 fixed 64-bit, 2 length-delimited bytes, 5 fixed 32-bit
//   initializer - a constant tensor stored in the graph (the weights)
//   value info  - a tensor's name plus (optionally) its type and shape; graph outputs are value infos
//
// Field numbers follow onnx/onnx.proto (ModelProto.graph = 7, GraphProto.output = 12, ...).

export interface TensorProto {
  name: string;
  dims: number[];
  dataType: number;
  /** Decoded values as float32 (weights only need float math here). Empty for unsupported types. */
  data: Float32Array;
  external: boolean;
}

export interface AttributeProto {
  name: string;
  f?: number;
  i?: number;
  s?: string;
  t?: TensorProto;
  floats?: number[];
  ints?: number[];
}

export interface NodeProto {
  name: string;
  opType: string;
  domain: string;
  inputs: string[];
  outputs: string[];
  attrs: Map<string, AttributeProto>;
}

export interface ValueInfo {
  name: string;
  elemType: number; // 0 = unknown
  shape: (number | string)[] | null; // number = fixed dim, string = symbolic (e.g. "batch")
}

export interface ModelInfo {
  irVersion: number;
  opset: number;
  producer: string;
  nodes: NodeProto[];
  initializers: Map<string, TensorProto>;
  inputs: ValueInfo[];
  outputs: ValueInfo[];
  valueInfo: Map<string, ValueInfo>;
  /** raw bytes and byte range of the graph field, kept for rewriting */
  _bytes: Uint8Array;
  _graphField: { start: number; end: number; dataStart: number; dataEnd: number };
}

export const ELEM = { FLOAT: 1, UINT8: 2, INT8: 3, UINT16: 4, INT16: 5, INT32: 6, INT64: 7, BOOL: 9, FLOAT16: 10, DOUBLE: 11 } as const;

class Reader {
  pos: number;
  constructor(public buf: Uint8Array, start = 0, public end = buf.length) {
    this.pos = start;
  }
  eof() {
    return this.pos >= this.end;
  }
  varint(): number {
    // returns a JS number; int64 values beyond 2^53 lose precision, fine for dims and attrs
    let result = 0;
    let mul = 1;
    for (;;) {
      const b = this.buf[this.pos++];
      result += (b & 0x7f) * mul;
      if (b < 0x80) break;
      mul *= 128;
    }
    return result;
  }
  signedVarint(): number {
    // int64 negative numbers are encoded as 10-byte two's complement varints
    const start = this.pos;
    const v = this.varint();
    if (this.pos - start >= 10 || v >= 2 ** 63) return v - 2 ** 64;
    return v;
  }
  tag(): { field: number; wire: number } {
    const t = this.varint();
    return { field: Math.floor(t / 8), wire: t & 7 };
  }
  bytes(): { start: number; end: number } {
    const len = this.varint();
    const start = this.pos;
    this.pos += len;
    return { start, end: this.pos };
  }
  skip(wire: number) {
    if (wire === 0) this.varint();
    else if (wire === 1) this.pos += 8;
    else if (wire === 2) this.bytes();
    else if (wire === 5) this.pos += 4;
    else throw new Error(`unsupported protobuf wire type ${wire}`);
  }
}

const td = new TextDecoder();
const str = (buf: Uint8Array, r: { start: number; end: number }) => td.decode(buf.subarray(r.start, r.end));

function readPackedVarints(r: Reader, wire: number, out: number[]) {
  if (wire === 2) {
    const { start, end } = r.bytes();
    const p = new Reader(r.buf, start, end);
    while (!p.eof()) out.push(p.signedVarint());
  } else out.push(r.signedVarint());
}

function readTensor(buf: Uint8Array, start: number, end: number): TensorProto {
  const r = new Reader(buf, start, end);
  const t: TensorProto = { name: "", dims: [], dataType: 0, data: new Float32Array(0), external: false };
  let raw: { start: number; end: number } | null = null;
  const floatData: number[] = [];
  const intData: number[] = [];
  const doubleData: number[] = [];
  while (!r.eof()) {
    const { field, wire } = r.tag();
    if (field === 1) readPackedVarints(r, wire, t.dims);
    else if (field === 2) t.dataType = r.varint();
    else if (field === 4) {
      if (wire === 2) {
        const b = r.bytes();
        const dv = new DataView(buf.buffer, buf.byteOffset + b.start, b.end - b.start);
        for (let o = 0; o + 4 <= dv.byteLength; o += 4) floatData.push(dv.getFloat32(o, true));
      } else {
        floatData.push(new DataView(buf.buffer, buf.byteOffset + r.pos, 4).getFloat32(0, true));
        r.pos += 4;
      }
    } else if (field === 5 || field === 7) readPackedVarints(r, wire, intData);
    else if (field === 8) t.name = str(buf, r.bytes());
    else if (field === 9) raw = r.bytes();
    else if (field === 10) {
      if (wire === 2) {
        const b = r.bytes();
        const dv = new DataView(buf.buffer, buf.byteOffset + b.start, b.end - b.start);
        for (let o = 0; o + 8 <= dv.byteLength; o += 8) doubleData.push(dv.getFloat64(o, true));
      } else {
        doubleData.push(new DataView(buf.buffer, buf.byteOffset + r.pos, 8).getFloat64(0, true));
        r.pos += 8;
      }
    } else if (field === 14) t.external = r.varint() === 1;
    else r.skip(wire);
  }
  const n = t.dims.reduce((a, b) => a * b, 1);
  if (raw) {
    const len = raw.end - raw.start;
    const copy = buf.slice(raw.start, raw.end); // copy: guarantees alignment for typed views
    const ab = copy.buffer;
    const dt = t.dataType;
    let src: ArrayLike<number> | null = null;
    if (dt === ELEM.FLOAT) src = new Float32Array(ab, 0, len / 4);
    else if (dt === ELEM.DOUBLE) src = new Float64Array(ab, 0, len / 8);
    else if (dt === ELEM.INT64) src = Array.from(new BigInt64Array(ab, 0, len / 8), Number);
    else if (dt === ELEM.INT32) src = new Int32Array(ab, 0, len / 4);
    else if (dt === ELEM.INT16) src = new Int16Array(ab, 0, len / 2);
    else if (dt === ELEM.INT8) src = new Int8Array(ab, 0, len);
    else if (dt === ELEM.UINT8 || dt === ELEM.BOOL) src = new Uint8Array(ab, 0, len);
    else if (dt === ELEM.FLOAT16) src = Array.from(new Uint16Array(ab, 0, len / 2), halfToFloat);
    if (src) t.data = Float32Array.from(src as ArrayLike<number>);
  } else if (floatData.length) t.data = Float32Array.from(floatData);
  else if (doubleData.length) t.data = Float32Array.from(doubleData);
  else if (intData.length) t.data = Float32Array.from(intData);
  if (t.data.length !== n && t.data.length === 0 && n > 0 && !t.external) {
    // unsupported encoding (strings, etc.): leave empty, callers treat as "no weights"
  }
  return t;
}

function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

function readAttr(buf: Uint8Array, start: number, end: number): AttributeProto {
  const r = new Reader(buf, start, end);
  const a: AttributeProto = { name: "" };
  while (!r.eof()) {
    const { field, wire } = r.tag();
    if (field === 1) a.name = str(buf, r.bytes());
    else if (field === 2) {
      a.f = new DataView(buf.buffer, buf.byteOffset + r.pos, 4).getFloat32(0, true);
      r.pos += 4;
    } else if (field === 3) a.i = r.signedVarint();
    else if (field === 4) a.s = str(buf, r.bytes());
    else if (field === 5) {
      const b = r.bytes();
      a.t = readTensor(buf, b.start, b.end);
    } else if (field === 7) {
      a.floats ??= [];
      if (wire === 2) {
        const b = r.bytes();
        const dv = new DataView(buf.buffer, buf.byteOffset + b.start, b.end - b.start);
        for (let o = 0; o + 4 <= dv.byteLength; o += 4) a.floats.push(dv.getFloat32(o, true));
      } else {
        a.floats.push(new DataView(buf.buffer, buf.byteOffset + r.pos, 4).getFloat32(0, true));
        r.pos += 4;
      }
    } else if (field === 8) {
      a.ints ??= [];
      readPackedVarints(r, wire, a.ints);
    } else r.skip(wire);
  }
  return a;
}

function readNode(buf: Uint8Array, start: number, end: number): NodeProto {
  const r = new Reader(buf, start, end);
  const n: NodeProto = { name: "", opType: "", domain: "", inputs: [], outputs: [], attrs: new Map() };
  while (!r.eof()) {
    const { field, wire } = r.tag();
    if (field === 1) n.inputs.push(str(buf, r.bytes()));
    else if (field === 2) n.outputs.push(str(buf, r.bytes()));
    else if (field === 3) n.name = str(buf, r.bytes());
    else if (field === 4) n.opType = str(buf, r.bytes());
    else if (field === 5) {
      const b = r.bytes();
      const a = readAttr(buf, b.start, b.end);
      n.attrs.set(a.name, a);
    } else if (field === 7) n.domain = str(buf, r.bytes());
    else r.skip(wire);
  }
  return n;
}

function readValueInfo(buf: Uint8Array, start: number, end: number): ValueInfo {
  const r = new Reader(buf, start, end);
  const v: ValueInfo = { name: "", elemType: 0, shape: null };
  while (!r.eof()) {
    const { field, wire } = r.tag();
    if (field === 1) v.name = str(buf, r.bytes());
    else if (field === 2) {
      // TypeProto { tensor_type = 1 { elem_type = 1, shape = 2 { dim = 1 { dim_value = 1 | dim_param = 2 } } } }
      const tb = r.bytes();
      const tr = new Reader(buf, tb.start, tb.end);
      while (!tr.eof()) {
        const t = tr.tag();
        if (t.field !== 1) {
          tr.skip(t.wire);
          continue;
        }
        const ttb = tr.bytes();
        const ttr = new Reader(buf, ttb.start, ttb.end);
        while (!ttr.eof()) {
          const tt = ttr.tag();
          if (tt.field === 1) v.elemType = ttr.varint();
          else if (tt.field === 2) {
            v.shape = [];
            const sb = ttr.bytes();
            const sr = new Reader(buf, sb.start, sb.end);
            while (!sr.eof()) {
              const st = sr.tag();
              if (st.field !== 1) {
                sr.skip(st.wire);
                continue;
              }
              const db = sr.bytes();
              const dr = new Reader(buf, db.start, db.end);
              let dim: number | string = "?";
              while (!dr.eof()) {
                const dt = dr.tag();
                if (dt.field === 1) dim = dr.signedVarint();
                else if (dt.field === 2) dim = str(buf, dr.bytes());
                else dr.skip(dt.wire);
              }
              v.shape.push(dim);
            }
          } else ttr.skip(tt.wire);
        }
      }
    } else r.skip(wire);
  }
  return v;
}

export function parseModel(bytes: Uint8Array): ModelInfo {
  const r = new Reader(bytes);
  let graph: ModelInfo["_graphField"] | null = null;
  let irVersion = 0;
  let opset = 0;
  let producer = "";
  while (!r.eof()) {
    const fieldStart = r.pos;
    const { field, wire } = r.tag();
    if (field === 1) irVersion = r.varint();
    else if (field === 2) producer = str(bytes, r.bytes());
    else if (field === 7) {
      const b = r.bytes();
      graph = { start: fieldStart, end: b.end, dataStart: b.start, dataEnd: b.end };
    } else if (field === 8) {
      // OperatorSetIdProto { domain = 1, version = 2 }: keep the default-domain version
      const b = r.bytes();
      const or = new Reader(bytes, b.start, b.end);
      let domain = "";
      let version = 0;
      while (!or.eof()) {
        const t = or.tag();
        if (t.field === 1) domain = str(bytes, or.bytes());
        else if (t.field === 2) version = or.varint();
        else or.skip(t.wire);
      }
      if (domain === "" || domain === "ai.onnx") opset = version;
    } else r.skip(wire);
  }
  if (!graph) throw new Error("Not an ONNX model: no graph found.");
  const g = new Reader(bytes, graph.dataStart, graph.dataEnd);
  const nodes: NodeProto[] = [];
  const initializers = new Map<string, TensorProto>();
  const inputs: ValueInfo[] = [];
  const outputs: ValueInfo[] = [];
  const valueInfo = new Map<string, ValueInfo>();
  while (!g.eof()) {
    const { field, wire } = g.tag();
    if (field === 1) {
      const b = g.bytes();
      nodes.push(readNode(bytes, b.start, b.end));
    } else if (field === 5) {
      const b = g.bytes();
      const t = readTensor(bytes, b.start, b.end);
      initializers.set(t.name, t);
    } else if (field === 11 || field === 12 || field === 13) {
      const b = g.bytes();
      const v = readValueInfo(bytes, b.start, b.end);
      if (field === 11) inputs.push(v);
      else if (field === 12) outputs.push(v);
      else valueInfo.set(v.name, v);
    } else g.skip(wire);
  }
  // Constant nodes are weights too (some exporters emit them instead of initializers)
  for (const n of nodes) {
    if (n.opType === "Constant" && n.attrs.get("value")?.t) {
      const t = n.attrs.get("value")!.t!;
      initializers.set(n.outputs[0], { ...t, name: n.outputs[0] });
    }
  }
  if (initializers.size && [...initializers.values()].some((t) => t.external))
    throw new Error("This model stores its weights in external data files. Re-export with the weights embedded (one .onnx file).");
  return { irVersion, opset, producer, nodes, initializers, inputs, outputs, valueInfo, _bytes: bytes, _graphField: graph };
}

// ---- writer -----------------------------------------------------------------------------------

function encVarint(v: number, out: number[]) {
  while (v >= 0x80) {
    out.push((v % 128) | 0x80);
    v = Math.floor(v / 128);
  }
  out.push(v);
}

function encLen(field: number, payload: Uint8Array): Uint8Array {
  const head: number[] = [];
  encVarint(field * 8 + 2, head);
  encVarint(payload.length, head);
  const out = new Uint8Array(head.length + payload.length);
  out.set(head, 0);
  out.set(payload, head.length);
  return out;
}

function concat(parts: Uint8Array[]): Uint8Array {
  const n = parts.reduce((a, p) => a + p.length, 0);
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

/**
 * Returns a copy of the model with extra graph outputs appended (protobuf allows repeated fields anywhere,
 * so appending to the end of the graph message is valid). Each output declares only its element type
 * (no shape); onnxruntime infers the rest.
 */
export function withExtraOutputs(model: ModelInfo, names: string[], elemType = ELEM.FLOAT): Uint8Array {
  const te = new TextEncoder();
  const existing = new Set(model.outputs.map((o) => o.name));
  const extra: Uint8Array[] = [];
  for (const name of names) {
    if (existing.has(name)) continue;
    existing.add(name);
    const tensorType = encLen(1, new Uint8Array([0x08, elemType])); // TypeProto.Tensor { elem_type }
    const typeProto = encLen(2, tensorType); // ValueInfoProto.type
    const vi = concat([encLen(1, te.encode(name)), typeProto]);
    extra.push(encLen(12, vi)); // GraphProto.output
  }
  const b = model._bytes;
  const gf = model._graphField;
  const newGraph = concat([b.subarray(gf.dataStart, gf.dataEnd), ...extra]);
  return concat([b.subarray(0, gf.start), b.subarray(gf.end), encLen(7, newGraph)]);
}

