"""Random-weight fixtures for skip connections and recurrent policies.

    python/.venv/bin/python test/fixtures/make_v2.py test/fixtures

resmlp  - dense residual MLP: an identity skip block and a projection (Linear) skip block
rescnn  - conv residual blocks: an identity skip (3x3 convs, padding 1) and a projection skip (1x1 stride-2 conv)
lstm    - one LSTM cell stepped per tick: inputs obs [N, 6], h0 / c0 [1, N, 32]; outputs logits, h, c
gru     - the same with a GRU cell: inputs obs [N, 6], h0 [1, N, 24]; outputs logits, h
gated   - x * sigmoid(W x): a multiplicative join that netglow still rejects with a clear message

The recurrent recordings hold two episodes (episode_starts marks the first tick of each); the logged outputs
come from stepping the torch module with the state reset to zeros at every episode start, which is exactly
what the browser has to reproduce for the match check to hit 100 %.
"""
import sys

import numpy as np
import torch
import torch.nn as nn

out = sys.argv[1]
torch.manual_seed(1)
rng = np.random.default_rng(1)


def export(m, args, path, inputs, outputs, dyn):
    torch.onnx.export(m, args, path, input_names=inputs, output_names=outputs, dynamic_axes=dyn, opset_version=17,
                      do_constant_folding=True, dynamo=False)


# ---------------------------------------------------------------- residual MLP
class ResMLP(nn.Module):
    def __init__(s):
        super().__init__()
        s.inp = nn.Linear(8, 32)
        s.a1, s.a2 = nn.Linear(32, 32), nn.Linear(32, 32)  # block 1, identity skip
        s.b1, s.b2 = nn.Linear(32, 48), nn.Linear(48, 48)  # block 2, projection skip 32 -> 48
        s.proj = nn.Linear(32, 48)
        s.head = nn.Linear(48, 4)

    def forward(s, x):
        h = torch.relu(s.inp(x))
        h = torch.relu(s.a2(torch.relu(s.a1(h))) + h)
        h = torch.relu(s.b2(torch.relu(s.b1(h))) + s.proj(h))
        return s.head(h)


T = 120
obs = np.cumsum(rng.standard_normal((T, 8)).astype(np.float32) * 0.15, 0)
m = ResMLP().eval()
with torch.no_grad():
    lg = m(torch.from_numpy(obs)).numpy()
export(m, torch.zeros(1, 8), f"{out}/resmlp.onnx", ["obs"], ["logits"], {"obs": {0: "N"}, "logits": {0: "N"}})
np.savez_compressed(f"{out}/resmlp.npz", obs=obs, actions=lg.argmax(1).astype(np.int64), outputs=lg, fps=np.float64(20))


# ---------------------------------------------------------------- residual CNN
class ResCNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.stem = nn.Conv2d(2, 16, 5, 2, padding=2)  # 64 -> 32
        s.c1, s.c2 = nn.Conv2d(16, 16, 3, 1, 1), nn.Conv2d(16, 16, 3, 1, 1)  # identity block
        s.d1, s.d2 = nn.Conv2d(16, 24, 3, 2, 1), nn.Conv2d(24, 24, 3, 1, 1)  # downsampling block 32 -> 16
        s.proj = nn.Conv2d(16, 24, 1, 2)  # projection skip
        s.head = nn.Sequential(nn.Flatten(), nn.Linear(24 * 16 * 16, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(s, x):
        h = torch.relu(s.stem(x.float() / 255.0))
        h = torch.relu(s.c2(torch.relu(s.c1(h))) + h)
        h = torch.relu(s.d2(torch.relu(s.d1(h))) + s.proj(h))
        return s.head(h)


T = 50
yy, xx = np.mgrid[0:64, 0:64]
img = np.stack([np.stack([(np.sin(xx / 5 + t / 4 + k) * np.cos(yy / 7 - t / 6) > 0.3) * 200 + rng.integers(0, 40, (64, 64)) for k in range(2)])
                for t in range(T)]).astype(np.uint8)
m = ResCNN().eval()
with torch.no_grad():
    lg = m(torch.from_numpy(img)).numpy()
export(m, torch.zeros(1, 2, 64, 64, dtype=torch.uint8), f"{out}/rescnn.onnx", ["obs"], ["logits"], {"obs": {0: "N"}, "logits": {0: "N"}})
np.savez_compressed(f"{out}/rescnn.npz", obs=img, actions=lg.argmax(1).astype(np.int64), outputs=lg, fps=np.float64(10))


# ---------------------------------------------------------------- recurrent policies, stepped once per tick
class LSTMPolicy(nn.Module):
    def __init__(s, n_in=6, H=32):
        super().__init__()
        s.enc = nn.Linear(n_in, 32)
        s.rnn = nn.LSTM(32, H)
        s.head = nn.Linear(H, 4)

    def forward(s, obs, h0, c0):
        y, (h, c) = s.rnn(torch.tanh(s.enc(obs)).unsqueeze(0), (h0, c0))
        return s.head(y[0]), h, c


class GRUPolicy(nn.Module):
    def __init__(s, n_in=6, H=24):
        super().__init__()
        s.rnn = nn.GRU(n_in, H)
        s.head = nn.Sequential(nn.Linear(H, 32), nn.Tanh(), nn.Linear(32, 3))

    def forward(s, obs, h0):
        y, h = s.rnn(obs.unsqueeze(0), h0)
        return s.head(y[0]), h


def step_all(m, obs, starts, shapes):
    """run tick by tick, threading the state; reset to zeros where an episode starts"""
    outs, state = [], None
    with torch.no_grad():
        for t in range(len(obs)):
            if starts[t] or state is None:
                state = [torch.zeros(*s) for s in shapes]
            r = m(torch.from_numpy(obs[t:t + 1]), *state)
            outs.append(r[0][0].numpy())
            state = list(r[1:])
    return np.stack(outs).astype(np.float32)


T = 140
obs = np.cumsum(rng.standard_normal((T, 6)).astype(np.float32) * 0.2, 0)
starts = np.zeros(T, np.uint8)
starts[[0, 80]] = 1
m = LSTMPolicy().eval()
lg = step_all(m, obs, starts, [(1, 1, 32), (1, 1, 32)])
export(m, (torch.zeros(1, 6), torch.zeros(1, 1, 32), torch.zeros(1, 1, 32)), f"{out}/lstm.onnx", ["obs", "h0", "c0"], ["logits", "h", "c"],
       {"obs": {0: "N"}, "h0": {1: "N"}, "c0": {1: "N"}, "logits": {0: "N"}, "h": {1: "N"}, "c": {1: "N"}})
np.savez_compressed(f"{out}/lstm.npz", obs=obs, episode_starts=starts, actions=lg.argmax(1).astype(np.int64), outputs=lg, fps=np.float64(20))

m = GRUPolicy().eval()
lg = step_all(m, obs, starts, [(1, 1, 24)])
export(m, (torch.zeros(1, 6), torch.zeros(1, 1, 24)), f"{out}/gru.onnx", ["obs", "h0"], ["logits", "h"],
       {"obs": {0: "N"}, "h0": {1: "N"}, "logits": {0: "N"}, "h": {1: "N"}})
np.savez_compressed(f"{out}/gru.npz", obs=obs, episode_starts=starts, actions=lg.argmax(1).astype(np.int64), outputs=lg, fps=np.float64(20))


# ---------------------------------------------------------------- still unsupported: multiplicative gating
class Gated(nn.Module):
    def __init__(s):
        super().__init__()
        s.a, s.g, s.b = nn.Linear(8, 16), nn.Linear(8, 16), nn.Linear(16, 4)

    def forward(s, x):
        return s.b(s.a(x) * torch.sigmoid(s.g(x)))


export(Gated().eval(), torch.zeros(1, 8), f"{out}/gated.onnx", ["obs"], ["logits"], {"obs": {0: "N"}})
np.savez_compressed(f"{out}/gated.npz", obs=obs[:, :6].repeat(2, 1)[:, :8])
print("ok")
