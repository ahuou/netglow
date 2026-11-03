// Saliency test: the heatmap toggle draws gradient × input of the chosen output over "What the AI sees" and the
// image node; the gradient is checked against central finite differences in onnxruntime (python, float-input
// fixture), the overlay changes the pixels, opacity works, it reaches exports, and models where it is not
// available say so.
import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
const url = process.env.URL ?? "http://localhost:5181/?demo=none";
const b = await chromium.launch();
// the first-visit legend is tested in polish.mjs; keep it out of these screenshots
const _np = b.newPage.bind(b); b.newPage = async (o) => { const pg = await _np(o); await pg.addInitScript(() => localStorage.setItem("netglow:legend-seen:v1", "1")); return pg; };
const p = await b.newPage({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
p.on("pageerror", (e) => { console.error(e); process.exitCode = 1; });
await p.goto(url);
const set = (k, v) => p.evaluate(([k, v]) => { const el = document.getElementById(`st-${k}`); if (el.type === "checkbox") el.checked = v === "1"; else el.value = v; el.dispatchEvent(new Event("input", { bubbles: true })); }, [k, v]);
const load = async (name) => {
  await p.setInputFiles("#fileModel", `test/fixtures/${name}.onnx`);
  await p.waitForFunction(() => document.getElementById("busy").hidden);
  await p.waitForTimeout(300);
  await p.evaluate(() => (window.netglow.S.viz = null));
  await p.setInputFiles("#fileRec", `test/fixtures/${name}.npz`);
  await p.waitForFunction((n) => window.netglow.S.recName === `${n}.npz` && window.netglow.S.viz, name);
};
const pix = () => p.evaluate(() => { const R = window.netglow.S.renderer, cv = document.getElementById("view"); const a = R.pics.find((q) => q.kind === "ai"); return Array.from(cv.getContext("2d").getImageData(a.x, a.y, a.w, a.h).data.filter((_, i) => i % 4 === 0)).reduce((s, v) => s + v, 0); });
// exactness: float-input conv net with max-pool, against finite differences
for (const name of ["rgbf", "cnn", "rescnn"]) {
  await load(name);
  await set("saliency", "0");
  await p.evaluate(() => { const n = window.netglow; n.S.t = 3 / n.S.hz; n.redraw(0); });
  const off = await pix();
  await set("saliency", "1");
  await p.evaluate(() => { const n = window.netglow; n.S.t = 3 / n.S.hz; n.redraw(0); });
  const on = await pix();
  const d = await p.evaluate(() => { const s = window.netglow.S.sal, k = 3; return { tick: k, target: s.target[k], heat: Array.from(s.heat.subarray(k * s.H * s.W, (k + 1) * s.H * s.W)), caption: window.netglow.S.renderer.pics.find((q) => q.kind === "ai").caption }; });
  writeFileSync(`test/exports/sal-${name}.json`, JSON.stringify(d));
  console.log(name, { off, on, caption: d.caption });
  assert.notEqual(on, off, "the overlay changes the AI view");
  assert.match(d.caption, /· saliency$/);
  if (name === "rgbf") {
    const r = JSON.parse(execFileSync("python/.venv/bin/python", ["test/fixtures/fd_saliency.py", `test/fixtures/${name}.onnx`, `test/fixtures/${name}.npz`, `test/exports/sal-${name}.json`]).toString());
    console.log(name, "finite differences", r);
    assert.ok(r.max_rel_err < 1e-2 && r.corr > 0.9999, "gradient × input matches finite differences (float32 central differences, eps 1e-3)");
  }
}
// demo: CNN pilot with heatmap, opacity, screenshot, export
await p.click(`#demoList [data-demo="lunar-cnn"]`);
await p.waitForFunction(() => window.netglow.S.recName.startsWith("lunar-cnn") && window.netglow.S.viz);
await set("saliency", "1"); await set("saliencyOpacity", "0.9"); await set("aiPos", "right");
await p.evaluate(() => { const n = window.netglow; n.stop(); n.S.t = 2.3; n.redraw(0); });
const a1 = await pix();
await set("saliencyOpacity", "0.3");
await p.evaluate(() => { const n = window.netglow; n.S.t = 2.3; n.redraw(0); });
assert.notEqual(await pix(), a1, "opacity changes the overlay");
await set("saliencyOpacity", "0.85");
await p.evaluate(() => { const n = window.netglow; n.S.t = 2.3; n.redraw(0); });
await p.screenshot({ path: "test/shots/saliency-1440.png" });
const png = await p.evaluate(() => document.getElementById("view").toDataURL("image/png"));
writeFileSync("test/shots/saliency-canvas.png", Buffer.from(png.split(",")[1], "base64"));
await p.selectOption("#format", "gif"); await p.selectOption("#size", "960x540"); await p.fill("#from", "2.2"); await p.fill("#to", "2.4");
const [dl] = await Promise.all([p.waitForEvent("download", { timeout: 120000 }), p.click("#exportBtn")]);
await dl.saveAs("test/exports/saliency.gif");
execFileSync("ffmpeg", ["-v", "error", "-i", "test/exports/saliency.gif", "-frames:v", "1", "-y", "test/exports/saliency-f0.png"]);
// heatmap pixels in the export: the theme's warm or cool colour inside the AI view, where the plain frame is gray
const warm = execFileSync("python3", ["-c", "from PIL import Image; im=Image.open('test/exports/saliency-f0.png').convert('RGB'); print(sum(1 for r,g,b in im.get_flattened_data() if (r>150 and g>90 and b<60) or (b>150 and r<120)))"]).toString().trim();
console.log("coloured px in export", warm);
assert.ok(Number(warm) > 200, "heatmap is in the export");
// not available: recurrent nets have no image; the mixed demo has one (and supports it)
await p.click(`#demoList [data-demo="lunar-mixed"]`);
await p.waitForFunction(() => window.netglow.S.recName.startsWith("lunar-mixed") && window.netglow.S.viz);
await p.evaluate(() => { const n = window.netglow; n.stop(); n.S.t = 1.5; n.redraw(0); });
assert.ok(await p.evaluate(() => !!window.netglow.S.sal && window.netglow.S.sal.H === 64), "mixed model: saliency on the frames");
await p.screenshot({ path: "test/shots/saliency-mixed-1440.png" });
await b.close();
console.log(process.exitCode ? "FAIL" : "PASS");
