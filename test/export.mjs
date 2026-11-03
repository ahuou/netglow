// Headless export test. usage: node test/export.mjs <mlp|cnn|demo:id> <gif|mp4|webm|webm-alpha> <outfile> [size] [fps] [toSeconds] [extra: theme=..,frame=1]
import { chromium } from "@playwright/test";
const [which, format, outFile, size = "960x540", fps = "30", to = "", extra = ""] = process.argv.slice(2);
const browser = await chromium.launch({ args: ["--enable-features=WebCodecs"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
await page.goto(process.env.URL ?? "http://localhost:5181/");
if (which.startsWith("demo:")) await page.click(`#demoList [data-demo="${which.slice(5)}"]`);
else {
  await page.setInputFiles("#fileModel", `test/fixtures/${which}.onnx`);
  await page.setInputFiles("#fileRec", `test/fixtures/${which}.npz`);
}
await page.waitForFunction(() => window.netglow?.S?.viz, null, { timeout: 60000 });
for (const kv of extra.split(",").filter(Boolean)) {
  const [k, v] = kv.split("=");
  if (k === "theme") await page.click(`[data-theme="${v}"]`);
  if (k === "frame") await page.setChecked("#showFrame", v === "1");
  if (k === "transparent") await page.setChecked("#transparent", v === "1");
  if (k === "from") await page.fill("#from", v);
}
await page.selectOption("#format", format);
await page.selectOption("#size", size);
await page.selectOption("#fps", fps);
if (to) await page.fill("#to", to);
const t0 = Date.now();
const [dl] = await Promise.all([page.waitForEvent("download", { timeout: 300000 }), page.click("#exportBtn")]);
await dl.saveAs(outFile);
console.log(await page.textContent("#exportResult"), `| wall ${((Date.now() - t0) / 1000).toFixed(1)} s ->`, outFile);
await browser.close();
