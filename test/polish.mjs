// Polish: the first-visit legend (shown once, dismissible, reopens from ?, lists what this net shows), the
// layout using free height at extreme sizes, and embedding (?embed=1 chrome-less player, embed code, player page).
import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
const base = (process.env.URL ?? "http://localhost:5181/?demo=none").replace(/\?.*$/, "");
const b = await chromium.launch();
const err = (pg) => pg.on("pageerror", (e) => { console.error(e); process.exitCode = 1; });
// ---- legend
for (const [w, h] of [[1440, 900], [390, 844]]) {
  const ctx = await b.newContext({ viewport: { width: w, height: h }, reducedMotion: "reduce" });
  const p = await ctx.newPage(); err(p);
  await p.goto(`${base}?demo=lunar-resmlp`);
  await p.waitForFunction(() => window.netglow?.S?.viz && !document.getElementById("legend").hidden, null, { timeout: 60000 });
  const items = await p.$$eval("#legendList dd", (d) => d.map((x) => x.textContent));
  console.log(w, "legend:", items.map((t) => t.split(" ").slice(0, 2).join(" ")));
  assert.ok(items.length >= 5 && items.some((t) => /^Skip/.test(t)), "legend lists nodes, glow, edges, outputs, and the skip");
  assert.equal(await p.evaluate(() => document.activeElement.id), "legendOk", "focus moves to the dismiss button");
  await p.screenshot({ path: `test/shots/legend-${w}.png`, fullPage: w < 600 });
  // the legend never sits inside the canvas, so exports never show it
  await p.click("#legendOk");
  assert.ok(await p.$eval("#legend", (e) => e.hidden));
  await p.reload();
  await p.waitForFunction(() => window.netglow?.S?.viz);
  await p.waitForTimeout(300);
  assert.ok(await p.$eval("#legend", (e) => e.hidden), "seen once: not shown again");
  if (w > 600) {
    await p.click("#legendBtn");
    assert.ok(!(await p.$eval("#legend", (e) => e.hidden)), "? reopens it");
    await p.keyboard.press("Escape");
    assert.ok(await p.$eval("#legend", (e) => e.hidden), "Esc closes it");
  }
  await ctx.close();
}
// ---- extreme sizes: the network uses the free height instead of shrinking into the middle
{
  const ctx = await b.newContext({ viewport: { width: 1440, height: 900 } });
  await ctx.addInitScript(() => localStorage.setItem("netglow:legend-seen:v1", "1"));
  const p = await ctx.newPage(); err(p);
  await p.goto(`${base}?demo=lunar-cnn&paused`);
  await p.waitForFunction(() => window.netglow?.S?.viz);
  const set = (k, v) => p.evaluate(([k, v]) => { const el = document.getElementById(`st-${k}`); el.value = v; el.dispatchEvent(new Event("input", { bubbles: true })); }, [k, v]);
  for (const [k, v] of Object.entries({ aiPos: "right", recPos: "bl", aiScale: "1.4", nodeScale: "2.5", mapScale: "2.5", imageScale: "2.5", colSpacing: "0.4", padding: "96", labelSize: "36", valueSize: "32" })) await set(k, v);
  await p.evaluate(() => { const n = window.netglow; n.stop(); n.S.t = 2.3; n.redraw(0); });
  const r = await p.evaluate(() => { const R = window.netglow.S.renderer, V = R.viz; const hid = V.cols.map((c, i) => i).filter((i) => V.cols[i].kind === "hidden" && V.cols[i].sel.length > 3);
    return { fit: R.fit, span: Math.max(...hid.map((i) => Math.max(...R.Y[i]) - Math.min(...R.Y[i]))), H: document.getElementById("view").height }; });
  console.log("extreme", r);
  assert.ok(r.fit < 0.7, "the width forced a shrink");
  assert.ok(r.span > 0.55 * r.H, "the columns still spread over most of the height");
  const png = await p.evaluate(() => document.getElementById("view").toDataURL("image/png"));
  (await import("node:fs")).writeFileSync("test/shots/extreme-look-after.png", Buffer.from(png.split(",")[1], "base64"));
  // ---- embed code for a demo carries the demo and the look
  const code = await p.$eval("#embedCode", (e) => e.value);
  console.log(code.slice(0, 120));
  assert.ok(code.startsWith(`<iframe src="${base}?demo=lunar-cnn&embed=1#look=`) && code.includes(`width="720" height="405"`), code);
  assert.match(code, /title="netglow: LunarLander · CNN from pixels"/);
  // the look survives into the embed
  const src = code.match(/src="([^"]+)"/)[1];
  const q = await ctx.newPage(); err(q);
  await q.setViewportSize({ width: 720, height: 405 });
  await q.goto(src);
  await q.waitForFunction(() => window.netglow?.S?.viz && window.netglow.S.playing, null, { timeout: 60000 });
  const e = await q.evaluate(() => ({ bar: getComputedStyle(document.querySelector(".bar")).display, panel: getComputedStyle(document.querySelector(".panel")).display,
    canvas: document.getElementById("view").getBoundingClientRect().width, nodeScale: window.netglow.S.style.nodeScale, legend: document.getElementById("legend").hidden }));
  console.log("embed", e);
  assert.equal(e.bar, "none"); assert.equal(e.panel, "none");
  assert.ok(e.canvas > 650, "the picture fills the frame");
  assert.equal(e.nodeScale, 2.5, "the look came along");
  await q.waitForTimeout(500);
  await q.screenshot({ path: "test/shots/embed-720.png" });
  await q.click("#view");
  assert.equal(await q.evaluate(() => window.netglow.S.playing), false, "click pauses");
  // ---- own files: export, then a self-contained player page
  await p.goto(`${base}?demo=none`);
  await p.setInputFiles("#fileModel", "test/fixtures/mlp.onnx");
  await p.setInputFiles("#fileRec", "test/fixtures/mlp.npz");
  await p.waitForFunction(() => window.netglow.S.viz && window.netglow.S.recName === "mlp.npz");
  assert.match(await p.$eval("#embedCode", (e) => e.value), /src="netglow-player.html"/);
  assert.ok(await p.$eval("#playerPage", (e) => e.disabled), "player page needs an export first");
  await p.selectOption("#format", "gif"); await p.selectOption("#size", "960x540"); await p.fill("#to", "0.5");
  await Promise.all([p.waitForEvent("download"), p.click("#exportBtn")]);
  const [dl] = await Promise.all([p.waitForEvent("download"), p.click("#playerPage")]);
  await dl.saveAs("test/exports/netglow-player.html");
  const html = readFileSync("test/exports/netglow-player.html", "utf8");
  assert.match(html, /^<!doctype html>/); assert.match(html, /<img src="data:image\/gif;base64,/);
  const pp = await ctx.newPage(); err(pp);
  await pp.goto("file://" + process.cwd() + "/test/exports/netglow-player.html");
  const im = await pp.$eval("img", (i) => [i.naturalWidth, i.naturalHeight]);
  console.log("player page image", im);
  assert.deepEqual(im, [960, 540]);
  await ctx.close();
}
await b.close();
console.log(process.exitCode ? "FAIL" : "PASS");
