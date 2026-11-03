// Recurrent test: LSTM and GRU policies stepped one tick at a time with the state threaded through, reset at the
// recording's episode_starts. Checks the match (100 % only if the state is carried and reset right), that netglow's
// own recomputation of the cell agrees with onnxruntime, the loop edges (exact R_c[j,k] * h(t-1)[k], zero right
// after a reset), the layout, and the LSTM LunarLander demo at desktop and phone width.
import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
const url = process.env.URL ?? "http://localhost:5181/?demo=none";
const b = await chromium.launch();
// the first-visit legend is tested in polish.mjs; keep it out of these screenshots
const _np = b.newPage.bind(b); b.newPage = async (o) => { const pg = await _np(o); await pg.addInitScript(() => localStorage.setItem("netglow:legend-seen:v1", "1")); return pg; };
const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
p.on("pageerror", (e) => { console.error(e); process.exitCode = 1; });
await p.goto(url);
for (const [name, H] of [["lstm", 32], ["gru", 24]]) {
  await p.setInputFiles("#fileModel", `test/fixtures/${name}.onnx`);
  await p.waitForFunction(() => document.getElementById("busy").hidden);
  await p.waitForTimeout(300);
  await p.evaluate(() => (window.netglow.S.viz = null));
  await p.setInputFiles("#fileRec", `test/fixtures/${name}.npz`);
  await p.waitForFunction((n) => window.netglow.S.recName === `${n}.npz` && window.netglow.S.modelName === `${n}.onnx` && window.netglow.S.viz, name);
  await p.evaluate(() => { const n = window.netglow; n.S.t = (n.S.viz.T * 0.5) / n.S.hz; n.redraw(0); });
  const r = await p.evaluate(() => {
    const S = window.netglow.S, V = S.viz, P = S.plan, tr = S.trace;
    const ci = V.cols.findIndex((c) => c.recurrent);
    const e = V.edges.find((x) => x.kind === "recurrent");
    const l = P.layers.find((x) => x.kind === "rnn"), cell = l.rnn;
    const h = tr.tensors.get(cell.hOut), ns = V.cols[ci].sel.length;
    let worst = 0, atReset = 0;
    const starts = [...V.starts.keys()].filter((t) => V.starts[t]);
    for (let t = 0; t < V.T; t++) for (let jj = 0; jj < ns; jj++) for (let kk = 0; kk < ns; kk++) {
      const v = e.c[(t * ns + jj) * ns + kk];
      if (V.starts[t]) { atReset = Math.max(atReset, Math.abs(v)); continue; }
      if (cell.type !== "lstm") continue;
      const j = V.cols[ci].sel[jj], k = V.cols[ci].sel[kk];
      const want = cell.Rc[j * cell.H + k] * h.data[(t - 1) * cell.H + k];
      worst = Math.max(worst, Math.abs(v - want));
    }
    const B = S.renderer.boxes, bad = [];
    for (let i = 0; i < B.length; i++) for (let j = i + 1; j < B.length; j++) {
      const a = B[i], c = B[j];
      if (a.x0 < c.x1 - 0.5 && c.x0 < a.x1 - 0.5 && a.y0 < c.y1 - 0.5 && c.y0 < a.y1 - 0.5) bad.push(`${a.kind} / ${c.kind}`);
    }
    return { match: document.getElementById("matchPill").textContent, states: P.states.map((s) => `${s.input}->${s.output}`), check: V.rnnCheck,
      label: V.cols[ci].label, from: e.from, to: e.to, worst, atReset, starts, bad, maxLoop: Math.max(...Array.from(e.c, Math.abs)) };
  });
  console.log(name, JSON.stringify(r));
  assert.equal(r.match, "Match 100%");
  assert.ok(r.check < 1e-4, "netglow's recomputed cell agrees with onnxruntime");
  assert.equal(r.from, r.to, "loop edges go from the column into itself");
  assert.deepEqual(r.starts, [0, 80], "two episodes");
  assert.equal(r.atReset, 0, "no loop contribution right after a reset (state is zero)");
  assert.ok(r.maxLoop > 0, "loop edges carry something");
  assert.ok(r.worst < 1e-5, "LSTM loop edges are R_c[j,k] * h(t-1)[k]");
  assert.equal(r.bad.length, 0, "no overlapping boxes");
  assert.match(r.label, new RegExp(`^${name} \\d · ${H}$`));
  const png = await p.evaluate(() => document.getElementById("view").toDataURL("image/png"));
  writeFileSync(`test/shots/recurrent-${name}-canvas.png`, Buffer.from(png.split(",")[1], "base64"));
}
for (const [w, h] of [[1440, 900], [390, 844]]) {
  const q = await b.newPage({ viewport: { width: w, height: h } });
  q.on("pageerror", (e) => { console.error(e); process.exitCode = 1; });
  await q.goto(url.replace("demo=none", "demo=lunar-lstm&paused"));
  await q.waitForFunction(() => window.netglow?.S?.viz && window.netglow.S.recName.startsWith("lunar-lstm"), null, { timeout: 60000 });
  await q.evaluate(() => { const n = window.netglow; n.stop(); n.S.t = 3.2; n.redraw(0); });
  assert.equal(await q.textContent("#matchPill"), "Match 100%");
  await q.screenshot({ path: `test/shots/lstm-demo-${w}.png`, fullPage: w < 600 });
  if (w > 600) {
    const png = await q.evaluate(() => document.getElementById("view").toDataURL("image/png"));
    writeFileSync("test/shots/lstm-demo-canvas.png", Buffer.from(png.split(",")[1], "base64"));
  }
  await q.close();
}
await b.close();
console.log(process.exitCode ? "FAIL" : "PASS");
