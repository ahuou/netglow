// Screenshot of the settings panel with every group open, and a customised CNN look.
import { chromium } from "@playwright/test";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
await p.goto("http://localhost:5181/?demo=lunar-cnn&paused");
await p.waitForFunction(() => window.netglow?.S?.viz);
await p.evaluate(() => { const n = window.netglow; n.stop(); n.S.t = 2.3; n.redraw(0); });
await p.screenshot({ path: "test/shots/settings-closed.png" });
await p.evaluate(() => document.querySelectorAll("#styleGroups details").forEach((d) => (d.open = true)));
await p.evaluate(() => document.getElementById("styleGroups").scrollIntoView());
await p.screenshot({ path: "test/shots/settings-open.png" });
const set = (k, v) => p.evaluate(([k, v]) => { const el = document.getElementById(`st-${k}`); if (el.type === "checkbox") el.checked = v === "1"; else el.value = v; el.dispatchEvent(new Event("input", { bubbles: true })); }, [k, v]);
await set("aiPos", "right"); await set("recPos", "bl"); await set("recScale", "0.8"); await set("nodeScale", "1.3"); await set("shape", "square");
await set("edgeMax", "4"); await set("pos", "#ff7a1a"); await set("neg", "#36d1c4"); await set("labelSize", "19"); await set("padding", "40");
await p.evaluate(() => { const n = window.netglow; n.S.t = 2.3; n.redraw(0); });
const png = await p.evaluate(() => document.getElementById("view").toDataURL());
(await import("node:fs")).writeFileSync("test/shots/custom-look-canvas.png", Buffer.from(png.split(",")[1], "base64"));
await p.screenshot({ path: "test/shots/custom-look.png" });
await b.close();
