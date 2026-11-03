// Everything a person can tune about the picture, as one JSON-able object: kept per model in localStorage,
// exported / imported as a file, and shareable in the URL hash.
//
// Vocabulary:
//   settings - view + export + style knobs (see DEFAULT_SETTINGS); "style" holds the size and look values
//   diff     - only the values that differ from the defaults; what goes in a share link

import { DEFAULT_STYLE, type Style } from "./render";
import type { ImageSettings } from "./image";
import type { OutputMode } from "./compute";
import type { ExportFormat } from "./export";

export interface Settings {
  nodes: number;
  topK: number;
  theme: string;
  /** null = decide from the model and recording */
  mode: OutputMode | null;
  speed: number;
  showFrame: boolean;
  transparent: boolean;
  showMaps: boolean;
  img: ImageSettings;
  format: ExportFormat;
  fps: number;
  size: string;
  style: Style;
}

export const DEFAULT_SETTINGS: Settings = {
  nodes: 10,
  topK: 3,
  theme: "ember",
  mode: null,
  speed: 1,
  showFrame: false,
  transparent: false,
  showMaps: true,
  img: { range: "auto", layout: "auto", newest: "last" },
  format: "gif",
  fps: 30,
  size: "1280x720",
  style: { ...DEFAULT_STYLE },
};

export const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x));

/** deep-merge known keys of `src` onto a copy of `base`; unknown keys and wrong types are ignored */
export function merge(base: Settings, src: unknown): Settings {
  const out = clone(base) as unknown as Record<string, unknown>;
  if (!src || typeof src !== "object") return out as unknown as Settings;
  const walk = (dst: Record<string, unknown>, s: Record<string, unknown>, ref: Record<string, unknown>) => {
    for (const k of Object.keys(ref)) {
      if (!(k in s)) continue;
      const rv = ref[k];
      const sv = s[k];
      if (rv && typeof rv === "object") {
        if (sv && typeof sv === "object") walk(dst[k] as Record<string, unknown>, sv as Record<string, unknown>, rv as Record<string, unknown>);
      } else if (rv === null ? sv === null || typeof sv === "string" : typeof sv === typeof rv) dst[k] = sv;
    }
  };
  walk(out, src as Record<string, unknown>, DEFAULT_SETTINGS as unknown as Record<string, unknown>);
  return out as unknown as Settings;
}

export function diff(s: Settings, base: Settings = DEFAULT_SETTINGS): Partial<Settings> {
  const walk = (a: Record<string, unknown>, b: Record<string, unknown>) => {
    const o: Record<string, unknown> = {};
    for (const k of Object.keys(a)) {
      if (a[k] && typeof a[k] === "object" && b[k] && typeof b[k] === "object") {
        const d = walk(a[k] as Record<string, unknown>, b[k] as Record<string, unknown>);
        if (Object.keys(d).length) o[k] = d;
      } else if (a[k] !== b[k]) o[k] = a[k];
    }
    return o;
  };
  return walk(s as unknown as Record<string, unknown>, base as unknown as Record<string, unknown>) as Partial<Settings>;
}

const KEY = "netglow:settings:v1:";

export function loadStored(model: string): unknown | null {
  try {
    const raw = localStorage.getItem(KEY + model);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function store(model: string, s: Settings) {
  try {
    localStorage.setItem(KEY + model, JSON.stringify(diff(s)));
  } catch {
    // private mode / storage full: settings simply don't persist
  }
}

export function clearStored(model: string) {
  try {
    localStorage.removeItem(KEY + model);
  } catch {
    /* ignore */
  }
}

const b64url = (s: string) => btoa(unescape(encodeURIComponent(s))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const unb64url = (s: string) => decodeURIComponent(escape(atob(s.replace(/-/g, "+").replace(/_/g, "/"))));

export function toHash(s: Settings): string {
  return `#look=${b64url(JSON.stringify(diff(s)))}`;
}

export function fromHash(hash: string): unknown | null {
  const m = /look=([A-Za-z0-9_-]+)/.exec(hash);
  if (!m) return null;
  try {
    return JSON.parse(unb64url(m[1]));
  } catch {
    return null;
  }
}
