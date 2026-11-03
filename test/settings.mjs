// Settings test: sizes / look change the render and the export; defaults reset; settings persist per model,
// round-trip through JSON and the share-link hash.
import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
const base = process.env.URL ?? "http://localhost:5181/";
const b = await chromium.launch();
const ctx = await b.newContext({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
const p = await ctx.newPage();
p.on("pageerror", (e) => { console.error(e); process.exitCode = 1; });
const load = async (url) => {
  await p.goto(url);
  await p.setInputFiles("#fileModel", "test/fixtures/cnn.onnx");
  await p.setInputFiles("#fileRec", "test/fixtures/cnn.npz");
  await p.waitForFunction(() => window.netglow?.S?.viz && window.netglow.S.renderer);
  await p.evaluate(() => { const n = window.netglow; n.S.t = 2; n.redraw(0); });
};
const snap = () => p.evaluate(() => { const r = window.netglow.S.renderer; return { r: r.r, X: r.X.slice(), map: r.mapSize, img: r.imgRect, pics: r.pics.map((q) => [q.kind, Math.round(q.x), Math.round(q.y), Math.round(q.w)]), data: document.getElementById("view").toDataURL() }; });
const setRange = (key, v) => p.evaluate(([k, v]) => { const el = document.getElementById(`st-${k}`); el.value = v; el.dispatchEvent(new Event("input", { bubbles: true })); }, [key, String(v)]);
const exportGif = async (out) => {
  await p.selectOption("#format", "gif"); await p.selectOption("#size", "960x540"); await p.fill("#to", "1");
  const [dl] = await Promise.all([p.waitForEvent("download"), p.click("#exportBtn")]);
  await dl.saveAs(out);
  return execFileSync("ffprobe", ["-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height", "-of", "csv=p=0", out]).toString().trim();
};
await load(`${base}?demo=none`);
const d0 = await snap();
const dim0 = await exportGif("test/exports/settings-default.gif");
// change sizes and look
await setRange("nodeScale", 2); await setRange("mapScale", 0.5); await setRange("imageScale", 2); await setRange("colSpacing", 0.6);
await setRange("labelSize", 26); await setRange("padding", 60); await setRange("glow", 0);
await p.selectOption("#st-aiPos", "tr", { force: true });
await p.evaluate(() => { const el = document.getElementById("st-pos"); el.value = "#ff00ff"; el.dispatchEvent(new Event("input", { bubbles: true })); });
const d1 = await snap();
assert.ok(d1.r > d0.r * 1.5, "node size grows");
assert.ok(d1.map < d0.map * 0.75, "feature-map tiles shrink");
assert.ok(d1.img[2] > d0.img[2] * 1.4, "image node grows");
assert.ok(d1.X.at(-1) - d1.X[0] < (d0.X.at(-1) - d0.X[0]) * 0.75, "columns closer");
assert.equal(d1.pics[0][0], "ai"); assert.ok(d1.pics[0][1] > 600, "AI view moved to the top-right corner");
assert.notEqual(d1.data, d0.data, "render changed");
const dim1 = await exportGif("test/exports/settings-custom.gif");
assert.equal(dim0, "960,540"); assert.equal(dim1, "960,540");
const f0 = readFileSync("test/exports/settings-default.gif"), f1 = readFileSync("test/exports/settings-custom.gif");
assert.ok(!f0.equals(f1), "export content changed with the settings");
// magenta exciting edges are in the export
execFileSync("ffmpeg", ["-v", "error", "-i", "test/exports/settings-custom.gif", "-frames:v", "1", "-y", "test/exports/settings-custom-f0.png"]);
const png = execFileSync("python3", ["-c", "from PIL import Image; im=Image.open('test/exports/settings-custom-f0.png').convert('RGB'); print(sum(1 for r,g,b in im.get_flattened_data() if r>200 and b>200 and g<90))"]).toString().trim();
assert.ok(Number(png) > 50, `magenta edges in export (${png} px)`);
// export size setting changes export dimensions
await p.selectOption("#format", "gif"); await p.selectOption("#size", "1080x1080");
const [dl] = await Promise.all([p.waitForEvent("download"), p.click("#exportBtn")]);
await dl.saveAs("test/exports/settings-square.gif");
assert.equal(execFileSync("ffprobe", ["-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", "test/exports/settings-square.gif"]).toString().trim(), "1080,1080");
// persisted per model: reload, same model -> same settings
await p.waitForTimeout(500);
await load(`${base}?demo=none`);
const persisted = await p.evaluate(() => window.netglow.S.style);
assert.equal(persisted.nodeScale, 2); assert.equal(persisted.aiPos, "tr"); assert.equal(persisted.pos, "#ff00ff");
// export JSON, reset, import JSON
const [js] = await Promise.all([p.waitForEvent("download"), p.evaluate(() => document.getElementById("exportLook").click())]);
await js.saveAs("test/exports/settings.json");
await p.evaluate(() => document.getElementById("resetLook").click()); await p.evaluate(() => { const n = window.netglow; n.S.t = 2; n.redraw(0); });
const reset = await snap();
assert.equal(reset.r, d0.r); assert.equal(reset.data, d0.data, "reset restores the default render exactly");
await p.setInputFiles("#importLook", "test/exports/settings.json");
await p.waitForFunction(() => window.netglow.S.style.nodeScale === 2);
// share link: hash carries the look into a fresh browser
const href = await p.evaluate(async () => { document.getElementById("copyLink").click(); await new Promise((r) => setTimeout(r, 100)); return location.href; });
assert.match(href, /#look=/);
const ctx2 = await b.newContext({ viewport: { width: 1440, height: 900 } });
const p2 = await ctx2.newPage();
await p2.goto(href.replace("?demo=none", "?demo=none"));
await p2.setInputFiles("#fileModel", "test/fixtures/mlp.onnx");
await p2.setInputFiles("#fileRec", "test/fixtures/mlp.npz");
await p2.waitForFunction(() => window.netglow?.S?.viz);
assert.equal(await p2.evaluate(() => window.netglow.S.style.labelSize), 26);
writeFileSync("test/exports/settings-ok.txt", "ok");
await b.close();
console.log(process.exitCode ? "FAIL" : "PASS");
