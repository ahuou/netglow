// Copies onnxruntime-web's WebAssembly runtime into public/ort so the app serves it itself (no CDN).
import { copyFileSync, mkdirSync } from "node:fs";
const src = "node_modules/onnxruntime-web/dist/";
mkdirSync("public/ort", { recursive: true });
for (const f of ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]) copyFileSync(src + f, "public/ort/" + f);
