// Save the full-resolution canvas for a fixture or demo. usage: node test/canvas.mjs <mlp|cnn|rgbf|demo:id> out.png [extra: aiView=beside,maps=0,size=1080x1080,frame=0]
import { chromium } from "@playwright/test";
import { writeFileSync } from "node:fs";
const [which, out, extra = ""] = process.argv.slice(2);
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
await p.goto("http://localhost:5181/?demo=none");
if (which.startsWith("demo:")) await p.click(`#demoList [data-demo="${which.slice(5)}"]`);
else { await p.setInputFiles("#fileModel", `test/fixtures/${which}.onnx`); await p.setInputFiles("#fileRec", `test/fixtures/${which}.npz`); }
await p.waitForFunction(() => window.netglow?.S?.viz);
for (const kv of extra.split(",").filter(Boolean)) {
  const [k, v] = kv.split("=");
  if (k === "aiView") await p.selectOption("#st-aiPos", v === "beside" ? "left" : v, { force: true });
  if (k === "maps") await p.setChecked("#showMaps", v === "1");
  if (k === "frame") await p.setChecked("#showFrame", v === "1");
  if (k === "size") await p.selectOption("#size", v);
  if (k === "theme") await p.click(`[data-theme="${v}"]`);
}
await p.evaluate(() => { const n = window.netglow; n.S.t = (n.S.viz.T * 0.45) / n.S.hz; n.redraw(0); });
const png = await p.evaluate(() => document.getElementById("view").toDataURL("image/png"));
writeFileSync(out, Buffer.from(png.split(",")[1], "base64"));
await b.close();
