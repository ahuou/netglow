// Screenshot of the settings panel with every group open, and a customised CNN look.
import { chromium } from "@playwright/test";
const b = await chromium.launch();
// the first-visit legend is tested in polish.mjs; keep it out of these screenshots
const _np = b.newPage.bind(b); b.newPage = async (o) => { const pg = await _np(o); await pg.addInitScript(() => localStorage.setItem("netglow:legend-seen:v1", "1")); return pg; };
const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
await p.goto((process.env.URL ?? "http://localhost:5181/").replace(/\?.*$/, "") + "?demo=lunar-cnn&paused");
await p.waitForFunction(() => window.netglow?.S?.viz);
await p.evaluate(() => { const n = window.netglow; n.stop(); n.S.t = 2.3; n.redraw(0); });
await p.screenshot({ path: "test/shots/settings-closed.png" });
await p.evaluate(() => document.querySelectorAll("#styleGroups details").forEach((d) => (d.open = true)));
await p.evaluate(() => document.getElementById("styleGroups").scrollIntoView());
await p.screenshot({ path: "test/shots/settings-open.png" });
const set = (k, v) => p.evaluate(([k, v]) => { const el = document.getElementById(`st-${k}`); if (el.type === "checkbox") el.checked = v === "1"; else el.value = v; el.dispatchEvent(new Event("input", { bubbles: true })); }, [k, v]);
await set("aiPos", "right"); await set("recPos", "bl"); await set("recScale", "0.8"); await set("nodeScale", "1.3"); await set("shape", "square");
await set("edgeMax", "4"); await set("pos", "#ff7a1a"); await set("neg", "#36d1c4"); await set("labelSize", "19"); await set("padding", "40");
await p.evaluate(() => { const n = window.netglow; n.S.t = 2.3; n.redraw(0); });
const png = await p.evaluate(() => document.getElementById("view").toDataURL());
(await import("node:fs")).writeFileSync("test/shots/custom-look-canvas.png", Buffer.from(png.split(",")[1], "base64"));
await p.screenshot({ path: "test/shots/custom-look.png" });
// extreme: everything big, columns squeezed, pictures beside on both sides; column titles must still not overlap
await set("aiScale", "1.4"); await set("nodeScale", "2.5"); await set("mapScale", "2.5");
await set("imageScale", "2.5"); await set("colSpacing", "0.4"); await set("padding", "96"); await set("labelSize", "36"); await set("valueSize", "32");
await p.evaluate(() => { const n = window.netglow; n.S.t = 2.3; n.redraw(0); });
const boxes = await p.evaluate(() => window.netglow.S.renderer.titleBoxes ?? []);
const over = boxes.some((a, i) => boxes.some((c, j) => j > i && a.x0 < c.x1 && c.x0 < a.x1 && a.y0 < c.y1 && c.y0 < a.y1));
const ext = await p.evaluate(() => document.getElementById("view").toDataURL());
(await import("node:fs")).writeFileSync("test/shots/extreme-look-canvas.png", Buffer.from(ext.split(",")[1], "base64"));
await p.screenshot({ path: "test/shots/extreme-look.png" });
console.log(boxes.length && !over ? "PASS column titles do not overlap" : `FAIL column titles overlap (${boxes.length} boxes)`);
if (!boxes.length || over) process.exitCode = 1;

// every reserved box (pictures with captions, titles, input / output label boxes, image node, node columns) is
// disjoint from every other one and inside the canvas, at the extreme settings and with the pictures elsewhere
const overlaps = () => p.evaluate(() => {
  const R = window.netglow.S.renderer, cv = document.getElementById("view");
  const B = R.boxes, bad = [];
  for (let i = 0; i < B.length; i++) {
    const a = B[i];
    if (a.x0 < -0.5 || a.y0 < -0.5 || a.x1 > cv.width + 0.5 || a.y1 > cv.height + 0.5) bad.push(`${a.kind} leaves the canvas`);
    for (let j = i + 1; j < B.length; j++) {
      const c = B[j];
      if (a.x0 < c.x1 - 0.5 && c.x0 < a.x1 - 0.5 && a.y0 < c.y1 - 0.5 && c.y0 < a.y1 - 0.5) bad.push(`${a.kind} / ${c.kind}`);
    }
  }
  return { n: B.length, kinds: [...new Set(B.map((x) => x.kind.split(" ")[0]))], fit: R.fit, bad };
});
const layouts = [
  ["cnn extreme, AI right, recording bottom left", {}],
  ["cnn extreme, AI left, recording right", { aiPos: "left", recPos: "right" }],
  ["cnn extreme, both above", { aiPos: "above", recPos: "auto" }],
  ["cnn extreme, corners", { aiPos: "tl", recPos: "br", rowSpacing: "1.8", colSpacing: "1.4" }],
];
for (const [name, extra] of layouts) {
  for (const [k, v] of Object.entries(extra)) await set(k, v);
  await p.evaluate(() => { const n = window.netglow; n.S.t = 2.3; n.redraw(0); });
  const r = await overlaps();
  console.log(r.bad.length ? "FAIL" : "PASS", `${name}: ${r.n} boxes (${r.kinds.join(", ")}), fit ${r.fit.toFixed(2)}`, r.bad.slice(0, 6).join("; "));
  if (r.bad.length || r.n < 12) process.exitCode = 1;
}
// the mixed demo (two inputs, two lanes) and the MLP at the same extreme look
for (const demo of ["lunar-mixed", "lunar-mlp"]) {
  await p.click(`#demoList [data-demo="${demo}"]`);
  await p.waitForFunction((d) => window.netglow.S.recName.startsWith(d) && window.netglow.S.viz, demo);
  for (const [k, v] of Object.entries({ aiPos: "right", recPos: "bl", aiScale: "1.4", nodeScale: "2.5", mapScale: "2.5", imageScale: "2.5", colSpacing: "0.4", padding: "96", labelSize: "36", valueSize: "32", rowSpacing: "1" })) await set(k, v);
  await p.evaluate(() => { const n = window.netglow; n.stop(); n.S.t = 1.5; n.redraw(0); });
  const r = await overlaps();
  console.log(r.bad.length ? "FAIL" : "PASS", `${demo} extreme: ${r.n} boxes, fit ${r.fit.toFixed(2)}`, r.bad.slice(0, 6).join("; "));
  if (r.bad.length) process.exitCode = 1;
  const png = await p.evaluate(() => document.getElementById("view").toDataURL());
  (await import("node:fs")).writeFileSync(`test/shots/extreme-${demo}-canvas.png`, Buffer.from(png.split(",")[1], "base64"));
}
await b.close();
console.log(process.exitCode ? "FAIL" : "PASS");
