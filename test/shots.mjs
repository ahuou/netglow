// Screenshots: empty state at 1440 and 390, each demo (MLP, CNN, mixed inputs) at 1440 and 390.
import { chromium } from "@playwright/test";
const url = process.env.URL ?? "http://localhost:5181/?demo=none";
const b = await chromium.launch();
for (const [w, h] of [[1440, 900], [390, 844]]) {
  const p = await b.newPage({ viewport: { width: w, height: h } });
  await p.goto(url); await p.waitForTimeout(400);
  await p.screenshot({ path: `test/shots/empty-${w}.png`, fullPage: w < 600 });
  await p.click("#demoList [data-demo='lunar-mlp']");
  await p.waitForFunction(() => window.netglow?.S?.viz);
  await p.evaluate(() => { const n = window.netglow; n.stop(); n.S.t = 2.1; n.redraw(0); });
  await p.screenshot({ path: `test/shots/demo-${w}.png`, fullPage: w < 600 });
  await p.click("#demoList [data-demo='lunar-cnn']");
  await p.waitForFunction(() => window.netglow?.S?.recName?.startsWith("lunar-cnn") && window.netglow.S.viz);
  await p.evaluate(() => { const n = window.netglow; n.S.t = (n.S.viz.T * 0.45) / n.S.hz; n.redraw(0); });
  await p.screenshot({ path: `test/shots/cnn-demo-${w}.png`, fullPage: w < 600 });
  await p.click("#demoList [data-demo='lunar-mixed']");
  await p.waitForFunction(() => window.netglow?.S?.recName?.startsWith("lunar-mixed") && window.netglow.S.viz);
  await p.evaluate(() => { const n = window.netglow; n.stop(); n.S.t = (n.S.viz.T * 0.4) / n.S.hz; n.redraw(0); });
  await p.screenshot({ path: `test/shots/mixed-demo-${w}.png`, fullPage: w < 600 });
}
await b.close();
