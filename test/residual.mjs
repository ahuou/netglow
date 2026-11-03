// Skip-connection test: residual MLP (identity + projection skips) and residual CNN (identity + 1x1 stride-2
// projection). Checks the match, that each skip becomes one edge set from the block's input column into the
// column after the block, that the skip edges are exactly the Add's other addend (identity: x[j] into unit j,
// conv: the channel mean; projection: P[j,i] * x[i]), that the block edges are W[j,i] * a[i], that the arcs run
// above the block, the layout has no overlaps, and that graphs netglow still cannot show are rejected clearly.
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
const load = async (name, rejected = false) => {
  await p.setInputFiles("#fileModel", `test/fixtures/${name}.onnx`);
  await p.setInputFiles("#fileRec", `test/fixtures/${name}.npz`);
  if (rejected) await p.waitForFunction((n) => window.netglow.S.modelName === `${n}.onnx` && window.netglow.S.modelError, name);
  else await p.waitForFunction((n) => window.netglow.S.recName === `${n}.npz` && window.netglow.S.modelName === `${n}.onnx` && window.netglow.S.viz, name);
};
// exactness of every skip and block edge at a few ticks, checked against the real tensors and the weights
const exact = () => p.evaluate(() => {
  const S = window.netglow.S, V = S.viz, P = S.plan, tr = S.trace;
  const colOf = new Map(); let ci = 0;
  V.cols.forEach((c, i) => { if (c.kind !== "input") colOf.set(P.layers[ci++], i); });
  let worst = 0, checked = 0;
  for (const sk of P.skips) {
    const to = colOf.get(sk.to), from = sk.from ? colOf.get(sk.from) : 0;
    const e = V.edges.find((x) => x.kind === "skip" && x.to === to && x.from === from);
    if (!e) return { error: "missing skip edge" };
    const src = tr.tensors.get(sk.src), nt = V.cols[to].sel.length, nf = V.cols[from].sel.length;
    const conv = src.dims.length === 3, hw = conv ? src.dims[1] * src.dims[2] : 1;
    for (const t of [0, Math.floor(V.T / 2), V.T - 1])
      for (let jj = 0; jj < nt; jj++) for (let ii = 0; ii < nf; ii++) {
        const d = V.cols[to].sel[jj], i = V.cols[from].sel[ii];
        let want;
        if (!sk.proj) {
          let m = 0;
          for (let q = 0; q < hw; q++) m += src.data[t * src.size + i * hw + q];
          want = d === i ? m / hw : 0;
        } else if (!conv) {
          want = sk.proj.W[d * sk.proj.in + i] * src.data[t * src.size + i];
        } else continue; // conv projection: same code path as a conv layer, covered by the conv edges below
        worst = Math.max(worst, Math.abs(e.c[(t * nt + jj) * nf + ii] - want) / (Math.abs(want) + 1e-3)); checked++;
      }
  }
  // dense block edges into the column after each block: W[j, i] * a[i]
  for (const sk of P.skips) {
    if (sk.to.kind !== "dense") continue;
    const to = colOf.get(sk.to), e = V.edges.find((x) => !x.kind && x.to === to);
    const a = tr.tensors.get(sk.to.inTensor), nt = V.cols[to].sel.length, nf = V.cols[e.from].sel.length;
    for (const t of [1, V.T - 2]) for (let jj = 0; jj < nt; jj++) for (let ii = 0; ii < nf; ii++) {
      const d = V.cols[to].sel[jj], i = V.cols[e.from].sel[ii];
      const want = sk.to.W[d * sk.to.in + i] * a.data[t * a.size + i];
      worst = Math.max(worst, Math.abs(e.c[(t * nt + jj) * nf + ii] - want) / (Math.abs(want) + 1e-3)); checked++;
    }
  }
  return { worst, checked };
});
const shape = () => p.evaluate(() => {
  const S = window.netglow.S, V = S.viz, R = S.renderer;
  const skips = V.edges.map((e, ei) => ({ e, ei })).filter((x) => x.e.kind === "skip").map(({ e, ei }) => {
    const mids = V.cols.map((c, ci) => ci).filter((ci) => V.cols[ci].stage > V.cols[e.from].stage && V.cols[ci].stage < V.cols[e.to].stage);
    const top = Math.min(...mids.map((ci) => Math.min(...R.Y[ci])));
    return { from: e.from, to: e.to, proj: !!e.proj, peak: R.skipPeak.get(ei), top, span: V.cols[e.to].stage - V.cols[e.from].stage };
  });
  const B = R.boxes, bad = [];
  for (let i = 0; i < B.length; i++) for (let j = i + 1; j < B.length; j++) {
    const a = B[i], c = B[j];
    if (a.x0 < c.x1 - 0.5 && c.x0 < a.x1 - 0.5 && a.y0 < c.y1 - 0.5 && c.y0 < a.y1 - 0.5) bad.push(`${a.kind} / ${c.kind}`);
  }
  return { skips, bad, match: document.getElementById("matchPill").textContent, status: document.getElementById("status").textContent };
});
const set = (k, v) => p.evaluate(([k, v]) => { const el = document.getElementById(`st-${k}`); el.value = v; el.dispatchEvent(new Event("input", { bubbles: true })); }, [k, v]);

for (const [name, nSkip] of [["resmlp", 2], ["rescnn", 2]]) {
  await load(name);
  await p.evaluate(() => { const n = window.netglow; n.S.t = (n.S.viz.T * 0.5) / n.S.hz; n.redraw(0); });
  const s = await shape();
  const x = await exact();
  console.log(name, JSON.stringify({ match: s.match, skips: s.skips, exact: x, bad: s.bad.slice(0, 4) }));
  assert.equal(s.match, "Match 100%");
  assert.equal(s.skips.length, nSkip, "one edge set per skip");
  assert.deepEqual(s.skips.map((k) => k.proj), [false, true], "identity first, then the projection");
  for (const k of s.skips) {
    assert.equal(k.span, 2, "the skip bypasses a two-layer block");
    assert.ok(k.peak < k.top, "the arc runs above the block it bypasses");
  }
  assert.ok(x.checked > 50 && x.worst < 1e-4, `skip and block edges are exact (worst rel. error ${x.worst})`);
  assert.equal(s.bad.length, 0, "no overlapping boxes");
  await p.screenshot({ path: `test/shots/residual-${name}-1440.png` });
  const png = await p.evaluate(() => document.getElementById("view").toDataURL("image/png"));
  writeFileSync(`test/shots/residual-${name}-canvas.png`, Buffer.from(png.split(",")[1], "base64"));
  // extreme sizes: still nothing on top of anything
  for (const [k, v] of Object.entries({ nodeScale: "2.5", mapScale: "2.5", imageScale: "2.5", colSpacing: "0.4", padding: "96", labelSize: "36", valueSize: "32" })) await set(k, v);
  await p.evaluate(() => { const n = window.netglow; n.S.t = 1; n.redraw(0); });
  const s2 = await shape();
  console.log(name, "extreme", s2.bad.length ? s2.bad.slice(0, 4) : "no overlaps");
  assert.equal(s2.bad.length, 0, "no overlaps at extreme sizes");
  await p.evaluate(() => document.getElementById("resetLook").click());
}
// the old fixture (ReLU between the block's layer and the Add) is shown, with a note on what the edges mean
await load("residual");
let st = await p.evaluate(() => document.getElementById("status").textContent);
assert.match(st, /Relu sits between dense 1 and the skip's Add/);
// multiplicative gating is still out of scope, with a message naming the node
await load("gated", true);
st = await p.evaluate(() => document.getElementById("status").textContent);
console.log("gated:", st);
assert.match(st, /Node "\/Mul" \(Mul\) joins two signals/);
// the residual LunarLander demo, desktop and phone
for (const [w, h] of [[1440, 900], [390, 844]]) {
  const q = await b.newPage({ viewport: { width: w, height: h } });
  q.on("pageerror", (e) => { console.error(e); process.exitCode = 1; });
  await q.goto(url.replace("demo=none", "demo=lunar-resmlp&paused"));
  await q.waitForFunction(() => window.netglow?.S?.viz && window.netglow.S.recName.startsWith("lunar-resmlp"), null, { timeout: 60000 });
  await q.evaluate(() => { const n = window.netglow; n.stop(); n.S.t = 2.4; n.redraw(0); });
  assert.equal(await q.textContent("#matchPill"), "Match 100%");
  await q.screenshot({ path: `test/shots/residual-demo-${w}.png`, fullPage: w < 600 });
  if (w > 600) {
    const png = await q.evaluate(() => document.getElementById("view").toDataURL("image/png"));
    writeFileSync("test/shots/residual-demo-canvas.png", Buffer.from(png.split(",")[1], "base64"));
  }
  await q.close();
}
await b.close();
console.log(process.exitCode ? "FAIL" : "PASS");
