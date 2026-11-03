#!/usr/bin/env bash
# Run every browser test against a dev server on :5181 (npm run dev -- --port 5181). Exits non-zero on the first failure.
set -u
cd "$(dirname "$0")/.."
fail=0
run() {
  local name="$1"; shift
  local out
  if out=$("$@" 2>&1); then
    if grep -qE "^FAIL|\[pageerror\]|AssertionError" <<<"$out"; then echo "FAIL $name"; echo "$out" | tail -20; fail=1; else echo "ok   $name"; fi
  else echo "FAIL $name (exit $?)"; echo "$out" | tail -20; fail=1; fi
}
smoke() { # smoke.mjs only prints; a demo or fixture must load and match
  local out; out=$(node test/smoke.mjs "$1" 2>&1)
  if grep -q '"pill": "Match 100%"' <<<"$out" && ! grep -q "pageerror" <<<"$out"; then echo "ok   smoke $1"; else echo "FAIL smoke $1"; echo "$out" | tail -12; fail=1; fi
}
for f in mlp cnn rgbf mixed resmlp rescnn lstm gru ${EXTRA_SMOKE:-}; do smoke "$f"; done
for t in ${TESTS:-mixed image drag panel settings residual recurrent saliency compare polish}; do run "$t" node "test/$t.mjs"; done
run shots node test/shots.mjs
mkdir -p test/exports
run "export gif" node test/export.mjs mlp gif test/exports/all-mlp.gif 960x540 30 1
run "export mp4" node test/export.mjs cnn mp4 test/exports/all-cnn.mp4 960x540 30 1
run "export webm-alpha" node test/export.mjs mlp webm-alpha test/exports/all-mlp.webm 960x540 30 1
run "canvas" node test/canvas.mjs mlp test/shots/all-canvas.png
exit $fail
