import { chromium } from "@playwright/test";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
await p.goto("http://localhost:5181/?demo=none");
await p.setInputFiles("#fileModel", "test/fixtures/mlp.onnx");
await p.setInputFiles("#fileRec", "test/fixtures/mlp.npz");
await p.waitForFunction(() => window.netglow?.S?.viz);
const before = await p.evaluate(() => window.netglow.S.viz.cols.at(-1).order.join(","));
// find canvas position of output slot 0 and drag it to slot 3
const pts = await p.evaluate(() => {
  const r = window.netglow.S.renderer, v = document.getElementById("view").getBoundingClientRect();
  const c = r.viz.cols.length - 1, sc = v.width / document.getElementById("view").width;
  const y = (slot) => v.top + r.Y[c][r.viz.cols[c].order[slot]] * sc;
  return { x: v.left + r.X[c] * sc, y0: y(0), y3: y(3) };
});
await p.mouse.move(pts.x, pts.y0); await p.mouse.down(); await p.mouse.move(pts.x, pts.y3, { steps: 8 }); await p.mouse.up();
const after = await p.evaluate(() => window.netglow.S.viz.cols.at(-1).order.join(","));
await p.focus('#outputLabels input[data-out="1"]'); await p.keyboard.press("Alt+ArrowDown");
const after2 = await p.evaluate(() => window.netglow.S.viz.cols.at(-1).order.join(","));
console.log({ before, after, after2 });
await b.close();
