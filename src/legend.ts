// "What am I looking at": a one-screen key for first-time visitors, drawn in the current theme's colours and
// listing only what the loaded network actually shows (skips, recurrent loops, image parts, the heatmap).

import type { Theme } from "./render";

export interface LegendFlags {
  image: boolean;
  maps: boolean;
  skip: boolean;
  recurrent: boolean;
  saliency: boolean;
  compare: boolean;
}

const svg = (w: number, body: string) => `<svg viewBox="0 0 ${w} 26" width="${w}" height="26" aria-hidden="true">${body}</svg>`;

export function legendHTML(th: Theme, pos: string, neg: string, f: LegendFlags): string {
  const node = (cx: number, c: string, glow = false) =>
    (glow ? `<circle cx="${cx}" cy="13" r="11" fill="${th.on}" opacity=".22"/>` : "") + `<circle cx="${cx}" cy="13" r="6" fill="${c}"/>`;
  const rows: [string, string][] = [
    [svg(44, node(9, th.off) + node(33, th.on, true)), `<b>Nodes</b> are units (neurons); each column is a layer, read left to right: inputs, hidden layers, outputs.`],
    [svg(44, `<rect x="4" y="9" width="36" height="8" rx="4" fill="url(#lg)"/><defs><linearGradient id="lg"><stop offset="0" stop-color="${th.off}"/><stop offset="1" stop-color="${th.on}"/></linearGradient></defs>`),
      `<b>Glow</b> is how active a unit is this tick, scaled per layer over the whole recording, so flicker is real change.`],
    [svg(44, `<path d="M3 7 C20 7 24 19 41 19" stroke="${pos}" stroke-width="3" fill="none" stroke-linecap="round"/><path d="M3 19 C20 19 24 7 41 7" stroke="${neg}" stroke-width="1.5" fill="none" stroke-linecap="round"/>`),
      `<b>Edges</b> are contributions this tick (weight × activation): <span style="color:${pos}">warm excites</span>, <span style="color:${neg}">cool inhibits</span>, thicker is stronger. Each node shows its top few.`],
    [svg(44, `<circle cx="9" cy="13" r="6" fill="${th.on}"/><circle cx="9" cy="13" r="9.5" fill="none" stroke="${th.text}" stroke-width="1.6"/><rect x="21" y="6" width="22" height="14" rx="4" fill="${th.on}"/>`),
      `<b>Outputs</b>: glow is the probability; the ringed, lit one is what the network chooses.`],
  ];
  if (f.image) rows.push([svg(44, `<rect x="10" y="2" width="24" height="22" fill="${th.on}"/><rect x="12.5" y="4.5" width="19" height="17" fill="#000"/><rect x="18" y="12" width="6" height="5" fill="#ddd"/>`),
    `<b>The image node</b> is the frame the network reads; its border lights up when the picture changes.${f.maps ? " Conv channels are drawn as their live <b>feature maps</b>." : ""}`]);
  if (f.skip) rows.push([svg(44, `<path d="M3 22 C8 22 8 5 14 5 L30 5 C36 5 36 22 41 22" stroke="${pos}" stroke-width="2" fill="none"/><circle cx="22" cy="19" r="4" fill="${th.off}"/>`),
    `<b>Skip</b> rails run over a residual block: the block's input added straight to its output.`]);
  if (f.recurrent) rows.push([svg(44, `<circle cx="14" cy="13" r="6" fill="${th.on}"/><path d="M21 9 C36 7 36 19 21 17" stroke="${pos}" stroke-width="1.8" fill="none" stroke-dasharray="3 2.5"/>`),
    `<b>Dashed loops</b> are memory: the recurrent cell's state from the previous tick feeding back in.`]);
  if (f.saliency) rows.push([svg(44, `<rect x="6" y="3" width="32" height="20" fill="#000"/><rect x="12" y="8" width="8" height="8" fill="${pos}" opacity=".85"/><rect x="24" y="10" width="7" height="7" fill="${neg}" opacity=".8"/>`),
    `<b>Heatmap</b>: pixels that push the chosen output up (warm) or down (cool), gradient × input.`]);
  if (f.compare) rows.push([svg(44, `<rect x="3" y="10" width="38" height="6" fill="${th.off}"/><rect x="15" y="10" width="7" height="6" fill="#e5566f"/><rect x="29" y="10" width="4" height="6" fill="#e5566f"/>`),
    `<b>Compare strip</b>: red ticks are where the two models choose differently.`]);
  return rows.map(([g, t]) => `<dt>${g}</dt><dd>${t}</dd>`).join("");
}
