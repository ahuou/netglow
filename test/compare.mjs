// Compare mode: the early-vs-trained demo (two PPO checkpoints on one recording) and a user-loaded second model
// with a different architecture. Checks both panels are drawn and synced, the agreement strip and timeline row,
// side by side on desktop and stacked on a phone, no overlaps inside either panel, and that exports carry both.
import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
const url = process.env.URL ?? "http://localhost:5181/?demo=none";
const b = await chromium.launch();
const state = (p) => p.evaluate(() => {
  const S = window.netglow.S, R = S.renderer;
  const over = (B) => { const bad = []; for (let i = 0; i < B.length; i++) for (let j = i + 1; j < B.length; j++) { const a = B[i], c = B[j]; if (a.x0 < c.x1 - 0.5 && c.x0 < a.x1 - 0.5 && a.y0 < c.y1 - 0.5 && c.y0 < a.y1 - 0.5) bad.push(`${a.kind} / ${c.kind}`); } return bad; };
  return { compare: !!R.agree, stacked: R.stacked, rects: R.rects, same: R.agree?.same, T: R.agree?.v.length, ticksA: R.a?.viz.T, ticksB: R.b?.viz.T,
    bad: R.a ? [...over(R.a.boxes), ...over(R.b.boxes)] : [], note: document.getElementById("cmpNote").textContent, match: document.getElementById("matchPill").textContent };
});
// desktop: side by side
const p = await b.newPage({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
p.on("pageerror", (e) => { console.error(e); process.exitCode = 1; });
await p.goto(url.replace("demo=none", "demo=lunar-compare&paused"));
await p.waitForFunction(() => window.netglow?.S?.renderer?.agree, null, { timeout: 60000 });
await p.evaluate(() => { const n = window.netglow; n.stop(); n.S.t = 2.4; n.redraw(0); });
const d = await state(p);
console.log("demo", JSON.stringify(d));
assert.ok(d.compare && !d.stacked, "side by side on desktop");
assert.equal(d.match, "Match 100%", "A is the model that flew");
assert.equal(d.ticksA, d.ticksB);
assert.ok(Math.abs(d.same - 0.2434) < 0.01, "agreement as computed in python (24.3%)");
assert.match(d.note, /Same choice as A on 24%/);
assert.equal(d.bad.length, 0, "no overlaps in either panel");
// both panels lit: the output pills of A and B are drawn in their own halves
const halves = await p.evaluate(() => { const cv = document.getElementById("view"), g = cv.getContext("2d"), R = window.netglow.S.renderer;
  return R.rects.map(([x, y, w, h]) => { const dd = g.getImageData(x, y, w, h).data; let n = 0; for (let i = 0; i < dd.length; i += 4) if (dd[i] > 200 && dd[i + 1] > 150) n++; return n; }); });
console.log("lit px per panel", halves);
assert.ok(halves.every((n) => n > 500), "both panels drawn");
await p.screenshot({ path: "test/shots/compare-1440.png" });
writeFileSync("test/shots/compare-canvas.png", Buffer.from((await p.evaluate(() => document.getElementById("view").toDataURL("image/png"))).split(",")[1], "base64"));
// export carries both panels and the strip
await p.selectOption("#format", "gif"); await p.selectOption("#size", "960x540"); await p.fill("#from", "2"); await p.fill("#to", "2.4");
const [dl] = await Promise.all([p.waitForEvent("download", { timeout: 120000 }), p.click("#exportBtn")]);
await dl.saveAs("test/exports/compare.gif");
execFileSync("ffmpeg", ["-v", "error", "-i", "test/exports/compare.gif", "-frames:v", "1", "-y", "test/exports/compare-f0.png"]);
const lit = execFileSync("python3", ["-c", "from PIL import Image; im=Image.open('test/exports/compare-f0.png').convert('RGB'); w,h=im.size; px=im.load(); print([sum(1 for x in range(a,b) for y in range(h) if px[x,y][0]>200 and px[x,y][1]>150) for a,b in ((0,w//2),(w//2,w))])"]).toString().trim();
console.log("export lit px per half", lit);
assert.ok(JSON.parse(lit).every((n) => n > 300), "both panels in the export");
// stacked layout by choice
await p.selectOption("#cmpLayout", "stacked");
await p.evaluate(() => { const n = window.netglow; n.S.t = 2.4; n.redraw(0); });
assert.ok((await state(p)).stacked);
assert.equal(await p.evaluate(() => document.getElementById("view").height), 1162, "stacked: two full 540 p panels plus header and strip");
await p.selectOption("#format", "mp4"); await p.fill("#from", "2"); await p.fill("#to", "2.2");
const [dl2] = await Promise.all([p.waitForEvent("download", { timeout: 120000 }), p.click("#exportBtn")]);
await dl2.saveAs("test/exports/compare-stacked.mp4");
const dims2 = execFileSync("ffprobe", ["-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height", "-of", "csv=p=0", "test/exports/compare-stacked.mp4"]).toString().trim();
console.log("stacked mp4", dims2);
assert.equal(dims2, "960,1162", "the export is the stacked composite");
await p.selectOption("#cmpLayout", "auto");
// a user-loaded second model with another architecture on the same recording
await p.goto(url);
await p.setInputFiles("#fileModel", "test/fixtures/mlp.onnx");
await p.setInputFiles("#fileRec", "test/fixtures/mlp.npz");
await p.waitForFunction(() => window.netglow.S.viz && window.netglow.S.recName === "mlp.npz");
await p.setInputFiles("#fileCompare", "test/fixtures/resmlp.onnx");
await p.waitForFunction(() => window.netglow.S.renderer?.agree);
const u = await state(p);
console.log("upload", JSON.stringify({ same: u.same, note: u.note, bad: u.bad.length }));
assert.ok(u.compare && u.same >= 0 && u.same <= 1);
assert.equal(await p.evaluate(() => window.netglow.S.renderer.b.viz.edges.some((e) => e.kind === "skip")), true, "B is the residual net");
await p.click("#cmpOff");
assert.equal(await p.evaluate(() => !!window.netglow.S.renderer.agree), false, "compare off");
// phone: stacked
const q = await b.newPage({ viewport: { width: 390, height: 844 } });
q.on("pageerror", (e) => { console.error(e); process.exitCode = 1; });
await q.goto(url.replace("demo=none", "demo=lunar-compare&paused"));
await q.waitForFunction(() => window.netglow?.S?.renderer?.agree, null, { timeout: 60000 });
await q.evaluate(() => { const n = window.netglow; n.stop(); n.S.t = 2.4; n.redraw(0); });
const ph = await state(q);
assert.ok(ph.stacked, "stacked on a phone");
assert.equal(ph.bad.length, 0);
await q.screenshot({ path: "test/shots/compare-390.png", fullPage: true });
await b.close();
console.log(process.exitCode ? "FAIL" : "PASS");
