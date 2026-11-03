// Load a netglow folder (model.onnx + recording.npz [+ labels.json]) as a user would and print what the app made of it,
// as one JSON line. Used by the Python tests. usage: node test/folder.mjs <dir> [url]
import { chromium } from "@playwright/test";
import { existsSync } from "node:fs";
const dir = process.argv[2];
const url = process.argv[3] ?? process.env.URL ?? "http://localhost:5181/?demo=none";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1280, height: 800 } });
const errors = [];
p.on("pageerror", (e) => errors.push(e.message));
await p.goto(url);
await p.setInputFiles("#fileModel", `${dir}/model.onnx`);
if (existsSync(`${dir}/labels.json`)) await p.setInputFiles("#fileLabels", `${dir}/labels.json`);
await p.setInputFiles("#fileRec", `${dir}/recording.npz`);
await p.waitForFunction(() => window.netglow?.S?.viz || document.querySelector("#status.error"), null, { timeout: 60000 });
const r = await p.evaluate(() => {
  const S = window.netglow.S;
  return {
    status: document.querySelector("#status").textContent,
    error: document.querySelector("#status").classList.contains("error"),
    match: document.querySelector("#matchPill").textContent,
    T: S.viz?.T ?? 0,
    inputs: S.plan?.inputs.map((i) => `${i.name}:${i.isImage ? "image" : "vector"}`) ?? [],
    layers: S.plan?.layers.length ?? 0,
    labels: S.labels ? { inputs: S.labels.inputs, outputs: S.labels.outputs } : null,
    title: document.getElementById("docTitle").textContent,
  };
});
r.pageErrors = errors;
console.log(JSON.stringify(r));
await b.close();
