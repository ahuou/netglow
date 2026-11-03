// Mixed-input test: the lunar-mixed demo has two ONNX inputs (frames [N,2,64,64] uint8 and vector [N,6] float)
// joined by a Concat. Checks the match, the column structure (image lane + vector lane feeding the merged dense
// column), that both branches carry edges, drag reorder of the vector inputs, the render, and a short GIF export.
import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
const url = process.env.URL ?? "http://localhost:5181/?demo=none";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
p.on("pageerror", (e) => { console.error(e); process.exitCode = 1; });
await p.goto(url);
await p.click(`#demoList [data-demo="lunar-mixed"]`);
await p.waitForFunction(() => window.netglow?.S?.viz && window.netglow.S.recName.startsWith("lunar-mixed"), null, { timeout: 60000 });
await p.evaluate(() => { const n = window.netglow; n.stop(); n.S.t = (n.S.viz.T * 0.4) / n.S.hz; n.redraw(0); });
const r = await p.evaluate(() => {
  const S = window.netglow.S, R = S.renderer, V = S.viz;
  const ins = V.cols.map((c, ci) => ({ ci, c })).filter((x) => x.c.kind === "input");
  const merged = V.cols.findIndex((c) => c.lane === -1);
  const cv = document.getElementById("view");
  const [x, y, w, h] = R.imgRect.map(Math.round);
  const d = cv.getContext("2d").getImageData(x, y, w, h).data;
  let mn = 765, mx = 0;
  for (let i = 0; i < d.length; i += 4) { const g = d[i] + d[i + 1] + d[i + 2]; mn = Math.min(mn, g); mx = Math.max(mx, g); }
  const into = V.edges.filter((e) => e.to === merged).map((e) => ({ from: e.from, strong: Math.max(...Array.from(e.c, Math.abs)) / e.scale }));
  return {
    inputs: S.plan.inputs.map((i) => `${i.name}:${i.isImage ? "image" : "vector"}`),
    lanes: V.lanes, stages: V.stages,
    inputCols: ins.map((x) => ({ ci: x.ci, image: !!x.c.image, n: x.c.sel.length, lane: x.c.lane })),
    merged, into, maps: V.cols.filter((c) => c.maps).length, spread: mx - mn,
    labels: S.labels.inputs, match: document.getElementById("matchPill").textContent,
    panelRows: document.querySelectorAll("#inputLabels li").length,
  };
});
console.log(JSON.stringify(r));
assert.equal(r.match, "Match 100%");
assert.deepEqual(r.inputs, ["frames:image", "vector:vector"]);
assert.equal(r.lanes, 2);
assert.equal(r.inputCols.length, 2);
assert.ok(r.inputCols[0].image && r.inputCols[0].n === 1, "image node");
assert.equal(r.inputCols[1].n, 6, "six labelled vector inputs");
assert.deepEqual(r.labels, ["camera", "x speed", "y speed", "angle", "spin", "left leg", "right leg"]);
assert.ok(r.maps >= 2, "feature maps on both conv columns");
assert.ok(r.spread > 60, "image node shows real pixels");
const vecCol = r.inputCols[1].ci;
assert.equal(new Set(r.into.map((e) => e.from)).size, 2, "the merged dense column gets edges from both branches");
assert.ok(r.into.some((e) => e.from === vecCol), "vector inputs feed the merged dense column directly");
assert.equal(r.panelRows, 7, "label panel lists the image and the 6 numbers");
// drag the first vector input to the third slot
const before = await p.evaluate((c) => window.netglow.S.viz.cols[c].order.join(","), vecCol);
const pts = await p.evaluate((c) => {
  const R = window.netglow.S.renderer, v = document.getElementById("view").getBoundingClientRect();
  const sc = v.width / document.getElementById("view").width, col = R.viz.cols[c];
  const y = (slot) => v.top + R.Y[c][col.order[slot]] * sc;
  return { x: v.left + R.X[c] * sc, y0: y(0), y2: y(2) };
}, vecCol);
await p.mouse.move(pts.x, pts.y0); await p.mouse.down(); await p.mouse.move(pts.x, pts.y2, { steps: 8 }); await p.mouse.up();
const after = await p.evaluate((c) => window.netglow.S.viz.cols[c].order.join(","), vecCol);
console.log({ before, after });
assert.notEqual(after, before, "vector inputs are draggable");
await p.evaluate(() => { const n = window.netglow; n.S.t = (n.S.viz.T * 0.4) / n.S.hz; n.redraw(0); });
await p.screenshot({ path: "test/shots/mixed-1440.png" });
const png = await p.evaluate(() => document.getElementById("view").toDataURL("image/png"));
writeFileSync("test/shots/mixed-canvas.png", Buffer.from(png.split(",")[1], "base64"));
// short GIF export
await p.selectOption("#format", "gif"); await p.selectOption("#size", "960x540"); await p.fill("#to", "1");
const [dl] = await Promise.all([p.waitForEvent("download", { timeout: 120000 }), p.click("#exportBtn")]);
await dl.saveAs("test/exports/mixed.gif");
const dims = execFileSync("ffprobe", ["-v", "error", "-select_streams", "v", "-count_frames", "-show_entries", "stream=width,height,nb_read_frames", "-of", "csv=p=0", "test/exports/mixed.gif"]).toString().trim();
console.log("gif", dims);
assert.match(dims, /^960,540,\d+$/);
assert.ok(Number(dims.split(",")[2]) >= 20, "about a second of frames");
await b.close();
console.log(process.exitCode ? "FAIL" : "PASS");
