"""Shared fixtures: a running viewer (vite dev server from the repo) and a helper that loads a folder in it
with the node + playwright harness in test/, so every test checks what the real web app makes of the files."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def viewer_url():
    """URL of the viewer: $NETGLOW_URL, else a vite dev server started from the repo for this session."""
    if os.environ.get("NETGLOW_URL"):
        yield os.environ["NETGLOW_URL"].rstrip("/") + "/"
        return
    if not shutil.which("node") or not (REPO / "node_modules" / "@playwright").is_dir():
        pytest.skip("browser tests need node and `npm install` at the repo root")
    port = _free_port()
    proc = subprocess.Popen(["npx", "vite", "--port", str(port), "--strictPort"], cwd=REPO,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://localhost:{port}/"
    for _ in range(100):
        try:
            urllib.request.urlopen(url, timeout=1)
            break
        except Exception:
            time.sleep(0.2)
    else:
        proc.kill()
        pytest.fail("vite did not start")
    yield url
    proc.terminate()
    proc.wait(10)


@pytest.fixture(scope="session")
def load_in_viewer(viewer_url):
    """load_in_viewer(dir) -> what the app shows for that folder (match pill, inputs, labels, ...)."""

    def run(folder) -> dict:
        out = subprocess.run(["node", "test/folder.mjs", str(folder), viewer_url + "?demo=none"], cwd=REPO,
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        r = json.loads(out.stdout.strip().splitlines()[-1])
        assert not r["error"], r["status"]
        assert not r["pageErrors"], r["pageErrors"]
        return r

    return run
