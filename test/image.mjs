// Image-node test: image models get one image node with a real thumbnail, the "What the AI sees" view and
// feature maps; pixel-range detection handles uint8 stacks and mean/std-normalised RGB floats.
import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
const url = process.env.URL ?? "http://localhost:5181/?demo=none";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
p.on("pageerror", (e) => { console.error(e); process.exitCode = 1; });
await p.goto(url);
const check = async (name, expect) => {
  await p.setInputFiles("#fileModel", `test/fixtures/${name}.onnx`);
  await p.setInputFiles("#fileRec", `test/fixtures/${name}.npz`);
  await p.waitForFunction((n) => window.netglow.S.recName === `${n}.npz` && window.netglow.S.viz, name);
  await p.evaluate(() => { const n = window.netglow; n.S.t = (n.S.viz.T * 0.5) / n.S.hz; n.redraw(0); });
  const r = await p.evaluate(() => {
    const S = window.netglow.S, R = S.renderer, cv = document.getElementById("view");
    const [x, y, w, h] = R.imgRect.map(Math.round);
    const d = cv.getContext("2d").getImageData(x, y, w, h).data;
    let mn = 255, mx = 0;
    for (let i = 0; i < d.length; i += 4) { const g = d[i] + d[i + 1] + d[i + 2]; mn = Math.min(mn, g); mx = Math.max(mx, g); }
    return { inputs: S.viz.cols[0].sel.length, edgeSrc: S.viz.edges[0].c.length / (S.viz.T * S.viz.cols[1].sel.length), pics: R.pics.map((q) => q.kind),
      maps: S.viz.cols.filter((c) => c.maps).length, spread: mx - mn, describe: document.getElementById("imageDescribe").textContent, match: document.getElementById("matchPill").textContent };
  });
  console.log(name, r);
  assert.equal(r.inputs, 1, "one image node");
  assert.equal(r.edgeSrc, 1, "edges summed into one source");
  assert.ok(r.pics.includes("ai"), "AI view drawn");
  assert.ok(r.maps >= 1, "feature maps computed");
  assert.ok(r.spread > 60, "thumbnail shows real pixels");
  assert.match(r.describe, expect);
  assert.equal(r.match, "Match 100%");
};
await check("cnn", /2 stacked grayscale frames, 0 to 255/);
await check("rgbf", /RGB, stretched per channel/);
await p.screenshot({ path: "test/shots/image-rgbf.png" });
await b.close();
console.log(process.exitCode ? "FAIL" : "PASS");
