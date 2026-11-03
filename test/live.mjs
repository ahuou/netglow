// Live mode end to end: the page connects to a Python stream, runs the model on each tick, animates, records and
// saves what it streamed. usage: node test/live.mjs <page url with ?live=…> <dir for the saved recording>
// Prints one JSON line. Used by python/tests/test_live.py (which runs the Python side).
import { chromium } from "@playwright/test";
const url = process.argv[2];
const out = process.argv[3];
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
const errors = [];
p.on("pageerror", (e) => errors.push(e.message));
p.on("console", (m) => m.type() === "error" && errors.push(m.text()));
await p.goto(url);
try {
  await p.waitForFunction(() => window.netglow?.S?.viz && window.netglow.live.window >= 10, null, { timeout: 60000 });
} catch (e) {
  const st = await p.evaluate(() => [document.getElementById("liveInfo")?.textContent, document.getElementById("status")?.textContent]);
  console.error("live view never filled:", st, errors);
  throw e;
}
const snap = () =>
  p.evaluate(() => {
    const n = window.netglow;
    const cv = document.getElementById("view");
    const d = cv.getContext("2d").getImageData(0, 0, cv.width, cv.height).data;
    let h = 0;
    for (let i = 0; i < d.length; i += 4 * 97) h = (h * 31 + d[i] + d[i + 1] * 7 + d[i + 2] * 13) >>> 0;
    return { newest: n.live.newest, window: n.live.window, t: n.S.t, T: n.S.viz.T, hash: h, match: document.getElementById("matchPill").textContent, info: document.getElementById("liveInfo").textContent };
  });
const a = await snap();
await p.waitForTimeout(1200);
const z = await snap();
// record ~1 s of the stream, then save it
await p.click("#liveRec");
await p.waitForTimeout(1000);
const [dl] = await Promise.all([p.waitForEvent("download", { timeout: 20000 }), p.click("#liveRec")]);
await dl.saveAs(`${out}/recording.npz`);
const saved = await p.textContent("#liveSaved");
const inputs = await p.evaluate(() => window.netglow.S.plan.inputs.map((i) => `${i.name}:${i.isImage ? "image" : "vector"}`));
await p.screenshot({ path: `${out}/live.png` });
// disconnect: the window stays loaded as an ordinary recording
await p.click("#liveBtn");
await p.waitForFunction(() => /Disconnected/.test(document.getElementById("liveInfo").textContent), null, { timeout: 30000 });
const after = await p.evaluate(() => ({ T: window.netglow.S.viz.T, match: document.getElementById("matchPill").textContent, play: !document.getElementById("play").disabled }));
console.log(JSON.stringify({ first: a, later: z, saved, inputs, after, errors }));
await b.close();
