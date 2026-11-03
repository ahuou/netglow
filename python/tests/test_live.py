"""Live mode: the wire protocol, and the browser end to end (connect, run the model per tick, animate, record, save)."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import threading
import time

import numpy as np
import pytest
import torch

import netglow
from netglow._live import LiveServer

from conftest import REPO, _free_port
from test_export import LSTMStep, Mixed, lstm_state


def _decode(msg: bytes):
    n = struct.unpack("<I", msg[:4])[0]
    return json.loads(msg[4:4 + n]), msg[4 + n:]


def test_protocol_model_backlog_ticks():
    from websockets.sync.client import connect

    srv = LiveServer(port=0, backlog=5, quiet=True)
    try:
        srv.set_model(b"fake-onnx", {"title": "x"}, 25)
        for i in range(8):
            srv.push({"obs": {"obs": np.full(3, i, np.float64)}, "output": np.array([i, -i]), "action": 1})
        time.sleep(0.2)
        with connect(srv.url, max_size=None) as ws:
            h, body = _decode(ws.recv(timeout=5))
            assert h["type"] == "model" and h["fps"] == 25 and body == b"fake-onnx"
            ticks = [_decode(ws.recv(timeout=5)) for _ in range(5)]  # backlog: the last 5 ticks
            assert [t[0]["i"] for t in ticks] == [3, 4, 5, 6, 7]
            h, body = ticks[-1]
            assert [a["key"] for a in h["arrays"]] == ["obs_obs", "outputs", "actions"]
            assert h["arrays"][0]["dtype"] == "<f4"  # float64 is sent as float32
            assert np.frombuffer(body[:12], "<f4").tolist() == [7, 7, 7]
            srv.push({"obs": {"obs": np.zeros(3, np.float32)}})
            assert _decode(ws.recv(timeout=5))[0]["i"] == 8
    finally:
        srv.close()


def _stream(rec: netglow.Recorder, stop: threading.Event, kind: str, hz=20):
    torch.set_num_threads(1)  # OpenMP thread count is per thread; the conftest setting doesn't reach this one
    rng = np.random.default_rng(0)
    vec = np.zeros(6, np.float32)
    t = 0
    while not stop.is_set():
        vec = (vec + rng.normal(size=6).astype(np.float32) * 0.1).astype(np.float32)
        if kind == "mixed":
            obs = {"frames": rng.integers(0, 255, (2, 64, 64), dtype=np.uint8), "vector": vec.copy()}
        else:
            obs = vec[:5].copy()
        start = t % 25 == 0  # recurrent: 25-tick episodes, the state resets on each first tick
        out = rec.infer(obs, episode_start=start)
        rec.add(obs, output=out, action=int(out.argmax()), episode_start=start)
        t += 1
        time.sleep(1 / hz)


def _live_run(page_url: str, out_dir, port: int, web=None, kind: str = "mixed"):
    torch.manual_seed(3)
    if kind == "mixed":
        rec = netglow.Recorder(Mixed(), out_dir=None, live=True, port=port, fps=20, quiet=True, web=web,
                               labels={"inputs": ["camera", "a", "b", "c", "d", "e", "f"], "title": "live test"})
    else:
        rec = netglow.Recorder(LSTMStep(), out_dir=None, live=True, port=port, fps=20, quiet=True, web=web,
                               state=lstm_state())
    stop = threading.Event()
    th = threading.Thread(target=_stream, args=(rec, stop, kind), daemon=True)
    th.start()
    if kind != "mixed":
        time.sleep(1.7)  # the viewer joins mid-episode: it must start from the state Python fed into that tick
    try:
        r = subprocess.run(["node", "test/live.mjs", page_url, str(out_dir)], cwd=REPO, capture_output=True, text=True, timeout=180)
    finally:
        stop.set()
        th.join(5)
        rec.close()
    assert r.returncode == 0, r.stderr
    return rec, json.loads(r.stdout.strip().splitlines()[-1])


def _check(res, rec, out_dir, load_in_viewer, kind="mixed"):
    assert not res["errors"], res["errors"]
    a, z = res["first"], res["later"]
    assert z["newest"] > a["newest"] + 10, ("ticks keep arriving", a, z)
    assert z["hash"] != a["hash"], "the canvas animates"
    assert a["match"] == "Match 100%" and z["match"] == "Match 100%"
    assert res["inputs"] == (["frames:image", "vector:vector"] if kind == "mixed" else ["obs:vector"])
    assert "Saved" in res["saved"]
    assert res["after"]["play"] and res["after"]["T"] >= 10
    assert res["after"]["match"] == "Match 100%", "the window replays exactly after Disconnect"
    z = np.load(out_dir / "recording.npz")
    if kind != "mixed":
        # a slice that starts mid-episode carries the state fed into its first tick
        assert z["episode_starts"][0] == 1 and {"init_h0", "init_c0"} <= set(z.files)
        assert z["init_h0"].shape == (16,)  # one state, not one per tick
        (out_dir / "model.onnx").write_bytes(rec.onnx)
        assert load_in_viewer(out_dir)["match"] == "Match 100%"
        return
    assert z["obs_frames"].dtype == np.uint8 and z["obs_frames"].shape[1:] == (2, 64, 64)
    assert z["obs_vector"].dtype == np.float32 and z["obs_vector"].shape[1:] == (6,)
    n = len(z["obs_vector"])
    assert 8 <= n <= 60 and z["outputs"].shape == (n, 4) and z["actions"].shape == (n,)
    assert float(z["fps"]) == pytest.approx(20)
    # the saved stream is a normal recording: with the model it replays with a full match
    (out_dir / "model.onnx").write_bytes(rec.onnx)
    assert load_in_viewer(out_dir)["match"] == "Match 100%"


def test_live_recurrent(tmp_path, viewer_url, load_in_viewer):
    """LSTM step function streamed live: the browser threads h / c, resets at episode starts, joins mid-episode"""
    port = _free_port()
    rec, res = _live_run(f"{viewer_url}?live={port}", tmp_path, port, kind="lstm")
    _check(res, rec, tmp_path, load_in_viewer, kind="lstm")


def test_live_with_saliency_and_compare(tmp_path, viewer_url):
    """the saliency heatmap and a compare model both follow the rolling live window"""
    torch.manual_seed(9)
    second = tmp_path / "b.onnx"
    second.write_bytes(netglow.export_bytes(Mixed(), (torch.zeros(1, 2, 64, 64, dtype=torch.uint8), torch.zeros(1, 6)),
                                            ["frames", "vector"]))
    port = _free_port()
    torch.manual_seed(3)
    rec = netglow.Recorder(Mixed(), out_dir=None, live=True, port=port, fps=20, quiet=True)
    stop = threading.Event()
    th = threading.Thread(target=_stream, args=(rec, stop, "mixed"), daemon=True)
    th.start()
    try:
        r = subprocess.run(["node", "test/live-extras.mjs", f"{viewer_url}?live={port}", str(second)], cwd=REPO,
                           capture_output=True, text=True, timeout=180)
    finally:
        stop.set()
        th.join(5)
        rec.close()
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert not res["errors"], res["errors"]
    assert res["sal"]["on"] and res["sal"]["forCurrent"] and res["sal"]["match"] == "Match 100%"
    assert res["cmp"]["cmpT"] == res["cmp"]["T"] and res["cmp"]["newest"] > res["sal"]["newest"]
    assert res["cmp"]["match"] == "Match 100%"


def test_live_in_browser(tmp_path, viewer_url, load_in_viewer):
    port = _free_port()
    rec, res = _live_run(f"{viewer_url}?live={port}", tmp_path, port)
    _check(res, rec, tmp_path, load_in_viewer)


@pytest.fixture(scope="session")
def dist(tmp_path_factory):
    if not shutil.which("npx"):
        pytest.skip("needs node")
    d = tmp_path_factory.mktemp("dist")
    subprocess.run(["npx", "vite", "build", "--outDir", str(d), "--emptyOutDir"], cwd=REPO, check=True, capture_output=True)
    return d


def test_live_served_from_python(tmp_path, dist, load_in_viewer):
    """The fallback for browsers that block https -> localhost: open http://localhost:PORT/ from the live server."""
    port = _free_port()
    rec, res = _live_run(f"http://localhost:{port}/", tmp_path, port, web=str(dist))
    _check(res, rec, tmp_path, load_in_viewer)
