"""torch modules -> netglow folder -> the web app runs them and agrees with torch (Match 100%)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

import netglow


def _rollout(rec: netglow.Recorder, obs_fn, T=60):
    for t in range(T):
        out = rec.add(obs_fn(t))
        rec.ticks[-1]["action"] = np.asarray(int(np.argmax(out)))  # what a greedy agent would execute
    return rec.save()


def test_mlp(tmp_path, load_in_viewer):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4))
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(size=(60, 8)) * 0.1, 0)
    d = _rollout(netglow.Recorder(net, out_dir=tmp_path, fps=20, title="mlp test"), lambda t: walk[t])
    r = load_in_viewer(d)
    assert r["match"] == "Match 100%"
    assert r["inputs"] == ["obs:vector"] and r["layers"] == 3 and r["T"] == 60
    assert r["title"] == "mlp test"


def test_export_file(tmp_path):
    net = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))
    p = netglow.export(net, torch.zeros(1, 3), tmp_path / "m.onnx")
    import onnx

    m = onnx.load(p)
    assert [i.name for i in m.graph.input] == ["obs"]
    assert {n.op_type for n in m.graph.node} >= {"Gemm"}
    assert m.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "N"  # dynamic batch


class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(2, 8, 8, 4), nn.ReLU(), nn.Conv2d(8, 16, 4, 2), nn.ReLU(), nn.Flatten())
        s.h = nn.Sequential(nn.Linear(16 * 36, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(s, x):
        return s.h(s.c(x.float() / 255.0))


def test_cnn_uint8(tmp_path, load_in_viewer):
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    d = _rollout(netglow.Recorder(CNN(), out_dir=tmp_path, fps=10), lambda t: rng.integers(0, 255, (2, 64, 64), dtype=np.uint8), T=30)
    r = load_in_viewer(d)
    assert r["match"] == "Match 100%"
    assert r["inputs"] == ["obs:image"]
    assert r["labels"]["inputs"] == ["image"]  # a lone image input is always called "image"


class Mixed(nn.Module):
    def __init__(s):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(2, 8, 8, 4), nn.ReLU(), nn.Conv2d(8, 16, 4, 2), nn.ReLU(), nn.Flatten())
        s.h = nn.Sequential(nn.Linear(16 * 36 + 6, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(s, frames, vector):
        return s.h(torch.cat([s.c(frames.float() / 255.0), vector], 1))


def test_mixed_two_inputs(tmp_path, load_in_viewer):
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    rec = netglow.Recorder(Mixed(), out_dir=tmp_path, fps=20,
                           labels={"inputs": ["camera", "a", "b", "c", "d", "e", "f"]})
    d = _rollout(rec, lambda t: {"frames": rng.integers(0, 255, (2, 64, 64), dtype=np.uint8),
                                 "vector": rng.normal(size=6).astype(np.float32)}, T=30)
    assert set(np.load(f"{d}/recording.npz").files) >= {"obs_frames", "obs_vector", "outputs", "actions"}
    r = load_in_viewer(d)
    assert r["match"] == "Match 100%"
    assert r["inputs"] == ["frames:image", "vector:vector"]
    assert r["labels"]["inputs"] == ["camera", "a", "b", "c", "d", "e", "f"]


def test_manual_outputs_and_tuple_obs(tmp_path):
    """Recorder without a model: you pass outputs yourself; tuple obs take input_names."""
    rec = netglow.Recorder(None, out_dir=tmp_path, input_names=["a", "b"], fps=5)
    for t in range(4):
        rec.add((np.zeros(3), np.ones(2)), output=np.arange(2.0), action=1)
    rec.save()
    z = np.load(tmp_path / "recording.npz")
    assert z["obs_a"].shape == (4, 3) and z["obs_a"].dtype == np.float32 and z["obs_b"].shape == (4, 2)
    assert z["actions"].tolist() == [1, 1, 1, 1]
    assert not (tmp_path / "model.onnx").exists()
    assert json.loads((tmp_path / "labels.json").read_text())["inputs"] == ["a[0]", "a[1]", "a[2]", "b[0]", "b[1]"]


def test_example_shape_guess():
    from netglow._export import _batched

    one = torch.zeros(8)
    assert _batched((one,))[0].shape == (1, 8)
    assert _batched((torch.zeros(1, 8),))[0].shape == (1, 8)
    assert _batched((torch.zeros(1, 64, 64),), batched=False)[0].shape == (1, 1, 64, 64)


@pytest.mark.parametrize("bad", ["nope"])
def test_export_needs_example(bad):
    with pytest.raises(ValueError):
        netglow.export_bytes(nn.Linear(2, 2))


H = 16


class LSTMStep(nn.Module):
    """one tick with explicit state: forward(obs, h0, c0) -> (logits, h, c), the form netglow steps"""

    def __init__(s):
        super().__init__()
        s.enc = nn.Linear(5, 16)
        s.rnn = nn.LSTM(16, H)
        s.head = nn.Linear(H, 3)

    def forward(s, obs, h0, c0):
        y, (h, c) = s.rnn(torch.tanh(s.enc(obs)).unsqueeze(0), (h0, c0))
        return s.head(y[0]), h, c


def lstm_state():
    return (torch.zeros(1, 1, H), torch.zeros(1, 1, H))


def test_lstm_recorder_episodes(tmp_path, load_in_viewer):
    """the Recorder threads the state, resets it at episode starts and saves episode_starts; the viewer must agree"""
    torch.manual_seed(4)
    rng = np.random.default_rng(4)
    rec = netglow.Recorder(LSTMStep(), out_dir=tmp_path, fps=20, state=lstm_state())
    for t in range(60):
        out = rec.add(rng.normal(size=5).astype(np.float32), episode_start=t % 20 == 0)
        rec.ticks[-1]["action"] = np.asarray(int(np.argmax(out)))
    rec.save()
    z = np.load(tmp_path / "recording.npz")
    assert z["episode_starts"].tolist() == [1 if t % 20 == 0 else 0 for t in range(60)]
    import onnx

    m = onnx.load(tmp_path / "model.onnx")
    assert [i.name for i in m.graph.input if i.name not in {w.name for w in m.graph.initializer}] == ["obs", "h0", "c0"]
    assert [o.name for o in m.graph.output] == ["logits", "h", "c"]
    r = load_in_viewer(tmp_path)
    assert r["match"] == "Match 100%" and r["inputs"] == ["obs:vector"]
