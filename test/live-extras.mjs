// Live mode with the saliency heatmap on and a second model in compare mode: both must follow the rolling window.
// usage: node test/live-extras.mjs <page url with ?live=…> <second model.onnx>. Prints one JSON line.
import { chromium } from "@playwright/test";
const [url, second] = process.argv.slice(2);
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
await p.addInitScript(() => localStorage.setItem("netglow:legend-seen:v1", "1"));
const errors = [];
p.on("pageerror", (e) => errors.push(e.message));
p.on("console", (m) => m.type() === "error" && errors.push(m.text()));
await p.goto(url);
await p.waitForFunction(() => window.netglow?.S?.viz && window.netglow.live.window >= 10, null, { timeout: 60000 });
const set = (k, v) => p.evaluate(([k, v]) => { const el = document.getElementById(`st-${k}`); if (el.type === "checkbox") el.checked = v === "1"; else el.value = v; el.dispatchEvent(new Event("input", { bubbles: true })); }, [k, v]);
await set("saliency", "1");
await p.waitForTimeout(1200);
const sal = await p.evaluate(() => { const S = window.netglow.S; return { on: !!S.sal, forCurrent: S.salFor === S.viz, T: S.viz.T, newest: window.netglow.live.newest, match: document.getElementById("matchPill").textContent }; });
await p.setInputFiles("#fileCompare", second);
await p.waitForFunction(() => window.netglow.S.cmp?.viz, null, { timeout: 30000 });
await p.waitForTimeout(1500);
const cmp = await p.evaluate(() => { const S = window.netglow.S; return { T: S.viz.T, cmpT: S.cmp?.viz?.T ?? -1, newest: window.netglow.live.newest, match: document.getElementById("matchPill").textContent }; });
await p.screenshot({ path: second.replace(/[^/]+$/, "live-compare.png") });
console.log(JSON.stringify({ sal, cmp, errors }));
await b.close();
