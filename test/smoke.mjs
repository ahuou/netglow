// Headless smoke test: load fixtures or a demo, check the match, screenshot. usage: node test/smoke.mjs [mlp|cnn|demo:<id>] [url]
import { chromium } from "@playwright/test";
const which = process.argv[2] ?? "mlp";
const url = process.argv[3] ?? "http://localhost:5181/";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
const logs = [];
page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
await page.goto(url);
if (which.startsWith("demo:")) {
  await page.click(`#demoList [data-demo="${which.slice(5)}"]`);
} else {
  await page.setInputFiles("#fileModel", `test/fixtures/${which}.onnx`);
  await page.setInputFiles("#fileRec", `test/fixtures/${which}.npz`);
}
await page.waitForFunction(() => window.netglow?.S?.viz || document.querySelector("#status.error"), null, { timeout: 60000 });
const st = await page.evaluate(() => ({ status: document.querySelector("#status").textContent, match: document.querySelector("#matchKv").innerText, pill: document.querySelector("#matchPill").textContent }));
console.log(JSON.stringify(st, null, 1));
await page.evaluate(() => { const n = window.netglow; n.S.t = (n.S.viz.T * 0.4) / n.S.hz; n.redraw(0); });
await page.waitForTimeout(300);
const name = which.replace(":", "-");
await page.screenshot({ path: `test/shots/${name}.png` });
console.log(logs.filter((l) => !l.includes("[vite]")).join("\n"));
await browser.close();
