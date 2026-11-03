"""Build the residual, LSTM and compare LunarLander demos (public/demos/lunar-resmlp, lunar-lstm, lunar-compare).

    python/.venv/bin/python test/fixtures/demos_v2.py resmlp     # residual MLP pilot, BC + DAgger from the PPO expert
    python/.venv/bin/python test/fixtures/demos_v2.py lstm       # LSTM pilot that never sees a velocity
    python/.venv/bin/python test/fixtures/demos_v2.py compare    # early PPO checkpoint vs the trained one

It reuses python/train_demos.py (expert, episode picking, evaluation helpers) and python/export_recording.py
without changing them; it needs the PPO expert at python/runs/ppo_lunar.zip (python train_demos.py ppo).

Vocabulary
----------
expert          the trained PPO MLP from the lunar-mlp demo; it sees the true 8-number state
student         the net being taught to imitate the expert (behaviour cloning + DAgger, as for the CNN demo)
partial state   x, y, angle and the two leg contacts: no speeds and no spin. A feed-forward net cannot tell
                which way it is moving from that; the LSTM has to remember earlier ticks to land
episode_starts  1 on the first tick of each episode in a recording; netglow resets the LSTM state there
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "python"))
import train_demos as td  # noqa: E402
from export_recording import export_onnx, save_recording, verify_onnx, write_labels  # noqa: E402

RUNS = os.path.join(HERE, "runs")
torch.set_num_threads(2)  # the students run one tick at a time; more threads only add overhead
DEMOS = os.path.join(ROOT, "public", "demos")
log = td.log


def landed(env) -> bool:
    return td.landed(env)


class StateObs(gym.Wrapper):
    """Plain LunarLander state, exposed as ``self.state`` too so td.collect can ask the expert."""

    def reset(self, **kw):
        self.state, info = self.env.reset(**kw)
        return self.state.copy(), info

    def step(self, a):
        self.state, r, te, tr, info = self.env.step(a)
        return self.state.copy(), r, te, tr, info


def fresh_eval(env, act_fn, seeds, reset_fn=None):
    """closed-loop landing rate and mean return on seeds never used in training"""
    rets, lands = [], []
    for s in seeds:
        if reset_fn:
            reset_fn()
        obs, _ = env.reset(seed=int(s))
        total, te = 0.0, False
        for _ in range(600):
            a, _ = act_fn(obs)
            obs, r, te, tr, _ = env.step(a)
            total += r
            if te or tr:
                break
        rets.append(total)
        lands.append(bool(te and landed(env)))
    return float(np.mean(rets)), float(np.mean(lands))


# ------------------------------------------------------------------ residual MLP


class ResPilot(nn.Module):
    """8 numbers -> dense 64 -> residual block (dense 64, ReLU, dense 64, + skip) -> ReLU -> 4 logits."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(8, 64)
        self.b1, self.b2 = nn.Linear(64, 64), nn.Linear(64, 64)
        self.head = nn.Linear(64, 4)

    def forward(self, x):
        h = torch.relu(self.inp(x))
        h = torch.relu(self.b2(torch.relu(self.b1(h))) + h)
        return self.head(h)


def resmlp() -> None:
    torch.manual_seed(0)
    _, expert = td.load_expert()
    env = StateObs(gym.make(td.ENV_ID, render_mode="rgb_array"))
    X, L, rets = td.collect(env, expert, episodes=60, seed0=10_000)
    log(f"[resmlp] BC on {len(X)} expert ticks (expert return {np.mean(rets):.1f})")
    net = ResPilot()
    for r in range(4):
        acc = td.fit(net, X.astype(np.float32), L, "cpu", epochs=20)
        student = td.make_student_fn(net, "cpu")
        ret, land = fresh_eval(env, student, range(50_000, 50_010))
        log(f"[resmlp] round {r}: train/val acc {acc[0]:.3f}/{acc[1]:.3f}, quick eval return {ret:.1f}, landed {land:.0%}")
        Xn, Ln, _ = td.collect(env, expert, student, episodes=30, seed0=20_000 + 1000 * r, beta=0.5 ** (r + 1))
        X, L = np.concatenate([X, Xn]), np.concatenate([L, Ln])
        net = ResPilot()
    acc = td.fit(net, X.astype(np.float32), L, "cpu", epochs=30)
    student = td.make_student_fn(net, "cpu")
    seeds = range(100, 150)
    ret, land = fresh_eval(env, student, seeds)
    log(f"[resmlp] final: {len(X)} ticks, train/val acc {acc[0]:.3f}/{acc[1]:.3f}; {len(seeds)} fresh seeds: return {ret:.1f}, landed {land:.0%}")
    out = os.path.join(DEMOS, "lunar-resmlp")
    rec = td.pick_episode(env, student, render=True, seeds=range(60))
    obs = rec["obs"].astype(np.float32)
    os.makedirs(out, exist_ok=True)
    export_onnx(net, torch.as_tensor(obs[:1]), os.path.join(out, "model.onnx"))
    v = verify_onnx(os.path.join(out, "model.onnx"), net, obs)
    inputs = ["x", "y", "x speed", "y speed", "angle", "spin", "left leg", "right leg"]
    save_recording(os.path.join(out, "recording.npz"), obs, rec["actions"], rec["outputs"], frames=rec["frames"], fps=td.FPS,
                   input_names=inputs, output_names=td.OUTPUT_NAMES)
    write_labels(os.path.join(out, "labels.json"), "LunarLander · residual MLP", inputs, td.OUTPUT_NAMES, icons=td.ICON_PATHS,
                 description=f"A residual MLP (dense 64, then a block of two dense 64 layers with an identity skip around it) taught "
                             f"to imitate the PPO expert (behaviour cloning + DAgger). Landed {land:.0%} of {len(seeds)} fresh seeds "
                             f"(mean return {ret:.0f}).",
                 input_units=["", "", "", "", "rad", "", "0/1", "0/1"])
    td.copy_icons(out)
    stats = {"fresh_seeds": [seeds.start, seeds.stop], "return": ret, "landed": land, "episode_seed": rec["seed"],
             "episode_T": len(obs), "episode_return": rec["episode_return"], "episode_landed": rec["landed"], "verify": v}
    os.makedirs(RUNS, exist_ok=True)
    json.dump(stats, open(os.path.join(RUNS, "resmlp_stats.json"), "w"), indent=2)
    log(f"[resmlp] demo: seed {rec['seed']} T={len(obs)} return {rec['episode_return']:.1f} landed={rec['landed']}")


# ------------------------------------------------------------------ LSTM pilot on the partial state

PART = [0, 1, 4, 6, 7]  # x, y, angle, left leg, right leg
PART_NAMES = ["x", "y", "angle", "left leg", "right leg"]
H = 64


class LSTMPilot(nn.Module):
    """partial state (5 numbers) -> dense 64 tanh -> LSTM 64 -> 4 logits. forward() is one tick with explicit state,
    the form netglow steps; seq() runs whole episodes for training with the same weights."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(len(PART), 64)
        self.rnn = nn.LSTM(64, H)
        self.head = nn.Linear(H, 4)

    def forward(self, obs, h0, c0):
        y, (h, c) = self.rnn(torch.tanh(self.enc(obs)).unsqueeze(0), (h0, c0))
        return self.head(y[0]), h, c

    def seq(self, x):  # x [T, B, 5]
        y, _ = self.rnn(torch.tanh(self.enc(x)))
        return self.head(y)


class Stepper:
    """the student flying tick by tick: keeps its own LSTM state"""

    def __init__(self, net):
        self.net = net
        self.reset()

    def reset(self):
        self.h = torch.zeros(1, 1, H)
        self.c = torch.zeros(1, 1, H)

    def __call__(self, obs):
        with torch.no_grad():
            lg, self.h, self.c = self.net(torch.as_tensor(obs[PART], dtype=torch.float32)[None], self.h, self.c)
        lg = lg[0].numpy()
        return int(lg.argmax()), lg


def collect_seq(env, expert, student=None, episodes=30, seed0=0, beta=1.0):
    eps = []
    rng = np.random.default_rng(seed0)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        if student:
            student.reset()
        X, L = [], []
        for _ in range(600):
            a_exp, lg = expert(env.state)
            X.append(obs[PART])
            L.append(lg)
            a = a_exp
            if student is not None:
                a_s, _ = student(obs)  # always step the student so its memory follows the episode
                if rng.random() >= beta:
                    a = a_s
            obs, r, te, tr, _ = env.step(a)
            if te or tr:
                break
        eps.append((np.asarray(X, np.float32), np.asarray(L, np.float32)))
    return eps


def fit_seq(net, eps, epochs=40, bs=16, lr=2e-3):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        idx = rng.permutation(len(eps))
        for i in range(0, len(idx), bs):
            batch = [eps[j] for j in idx[i:i + bs]]
            T = max(len(x) for x, _ in batch)
            X = torch.zeros(T, len(batch), len(PART))
            P = torch.zeros(T, len(batch), 4)
            M = torch.zeros(T, len(batch))
            for b, (x, lg) in enumerate(batch):
                X[: len(x), b] = torch.as_tensor(x)
                P[: len(x), b] = torch.softmax(torch.as_tensor(lg), -1)
                M[: len(x), b] = 1
            loss = -((P * torch.log_softmax(net.seq(X), -1)).sum(-1) * M).sum() / M.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
    with torch.no_grad():
        agree = [float((net.seq(torch.as_tensor(x)[:, None])[:, 0].argmax(-1).numpy() == lg.argmax(-1)).mean()) for x, lg in eps[-20:]]
    return float(np.mean(agree))


def lstm(rounds: int = 6) -> None:
    torch.manual_seed(0)
    _, expert = td.load_expert()
    env = StateObs(gym.make(td.ENV_ID, render_mode="rgb_array"))
    t0 = time.time()
    eps = collect_seq(env, expert, episodes=80, seed0=10_000)
    net = LSTMPilot()
    best = None
    for r in range(rounds + 1):
        acc = fit_seq(net, eps)
        st = Stepper(net)
        ret, land = fresh_eval(env, st, range(50_000, 50_020), reset_fn=st.reset)
        log(f"[lstm] round {r}: {len(eps)} episodes, agreement {acc:.3f}; eval return {ret:.1f}, landed {land:.0%} ({time.time() - t0:.0f}s)")
        if best is None or ret > best[0]:
            best = (ret, {k: v.clone() for k, v in net.state_dict().items()})
        if r < rounds:
            eps += collect_seq(env, expert, Stepper(net), episodes=40, seed0=20_000 + 1000 * r, beta=0.5 ** (r + 1))
            net = LSTMPilot()
    net.load_state_dict(best[1])
    net.eval()
    os.makedirs(RUNS, exist_ok=True)
    torch.save(best[1], os.path.join(RUNS, "lstm_pilot.pt"))
    export_lstm(net, env)


def export_lstm(net=None, env=None) -> None:
    if net is None:
        net = LSTMPilot()
        net.load_state_dict(torch.load(os.path.join(RUNS, "lstm_pilot.pt")))
        net.eval()
        env = StateObs(gym.make(td.ENV_ID, render_mode="rgb_array"))
    st = Stepper(net)
    seeds = range(100, 150)
    ret, land = fresh_eval(env, st, seeds, reset_fn=st.reset)
    log(f"[lstm] {len(seeds)} fresh seeds: return {ret:.1f}, landed {land:.0%}")
    # two consecutive episodes in one recording: the browser has to reset the state where the second one starts
    picked = []
    for s in range(0, 80):
        if len(picked) == 2:
            break
        st.reset()
        rec = td.record_episode(env, st, seed=s, max_steps=600, render=True)
        ok = rec["terminated"] and landed(env)
        log(f"  seed {s}: T={len(rec['obs'])} return={rec['episode_return']:.1f} landed={ok}")
        if ok and 150 <= len(rec["obs"]) <= 420:
            rec["seed"] = s
            picked.append(rec)
    obs = np.concatenate([r["obs"][:, PART] for r in picked]).astype(np.float32)
    starts = np.zeros(len(obs), np.uint8)
    starts[0] = starts[len(picked[0]["obs"])] = 1
    # outputs logged while flying; check they are what stepping the exported module with resets gives
    outputs = np.concatenate([r["outputs"] for r in picked]).astype(np.float32)
    actions = np.concatenate([r["actions"] for r in picked])
    frames = np.concatenate([r["frames"] for r in picked])
    out = os.path.join(DEMOS, "lunar-lstm")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "model.onnx")
    export_onnx_state(net, path)
    check_stepped(path, obs, starts, outputs)
    save_recording(os.path.join(out, "recording.npz"), obs, actions, outputs, frames=frames, fps=td.FPS,
                   input_names=PART_NAMES, output_names=td.OUTPUT_NAMES)
    with np.load(os.path.join(out, "recording.npz")) as z:
        arrays = dict(z)
    arrays["episode_starts"] = starts
    np.savez_compressed(os.path.join(out, "recording.npz"), **arrays)
    write_labels(os.path.join(out, "labels.json"), "LunarLander · LSTM, no speeds", PART_NAMES, td.OUTPUT_NAMES, icons=td.ICON_PATHS,
                 description=f"An LSTM pilot that sees only position, tilt and leg contacts: no speeds, no spin, so it has to "
                             f"remember earlier ticks to know how it is moving. Taught to imitate the PPO expert (behaviour cloning + "
                             f"DAgger on whole episodes). Landed {land:.0%} of {len(seeds)} fresh seeds (mean return {ret:.0f}). "
                             f"The recording holds two episodes; the state resets where the second one starts.",
                 input_units=["", "", "rad", "0/1", "0/1"])
    td.copy_icons(out)
    stats = {"fresh_seeds": [seeds.start, seeds.stop], "return": ret, "landed": land, "episodes": [
        {"seed": r["seed"], "T": len(r["obs"]), "return": r["episode_return"]} for r in picked]}
    json.dump(stats, open(os.path.join(RUNS, "lstm_stats.json"), "w"), indent=2)
    log(f"[lstm] demo: {stats['episodes']}")


def export_onnx_state(net, path):
    with torch.no_grad():
        torch.onnx.export(net, (torch.zeros(1, len(PART)), torch.zeros(1, 1, H), torch.zeros(1, 1, H)), path,
                          input_names=["obs", "h0", "c0"], output_names=["logits", "h", "c"],
                          dynamic_axes={"obs": {0: "N"}, "h0": {1: "N"}, "c0": {1: "N"}, "logits": {0: "N"}, "h": {1: "N"}, "c": {1: "N"}},
                          opset_version=17, do_constant_folding=True, dynamo=False)


def check_stepped(path, obs, starts, outputs):
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    h = c = None
    got = []
    for t in range(len(obs)):
        if starts[t]:
            h = np.zeros((1, 1, H), np.float32)
            c = np.zeros((1, 1, H), np.float32)
        lg, h, c = sess.run(None, {"obs": obs[t:t + 1], "h0": h, "c0": c})
        got.append(lg[0])
    got = np.stack(got)
    d = float(np.abs(got - outputs).max())
    agree = float((got.argmax(1) == outputs.argmax(1)).mean())
    log(f"[lstm] stepped onnxruntime vs logged outputs: max diff {d:.2g}, argmax agreement {agree:.2%}")
    assert agree == 1.0


# ------------------------------------------------------------------ compare: early vs trained PPO


def compare(steps: int = 60_000) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    from export_recording import sb3_actor

    torch.set_num_threads(4)
    venv = make_vec_env(td.ENV_ID, n_envs=16, seed=0)
    model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=64, gae_lambda=0.98, gamma=0.999, n_epochs=4, ent_coef=0.01,
                verbose=0, seed=0, device="cpu")
    model.learn(total_timesteps=steps)
    venv.close()
    early = sb3_actor(model).eval()

    def act(o):
        with torch.no_grad():
            lg = early(torch.as_tensor(o, dtype=torch.float32)[None])[0].numpy()
        return int(lg.argmax()), lg

    env = StateObs(gym.make(td.ENV_ID))
    seeds = range(100, 150)
    ret_e, land_e = fresh_eval(env, act, seeds)
    _, expert = td.load_expert()
    ret_t, land_t = fresh_eval(env, expert, seeds)
    log(f"[compare] early ({steps} steps): return {ret_e:.1f}, landed {land_e:.0%}; trained (1M steps): return {ret_t:.1f}, landed {land_t:.0%}")
    out = os.path.join(DEMOS, "lunar-compare")
    os.makedirs(out, exist_ok=True)
    src = os.path.join(DEMOS, "lunar-mlp")
    for f in ("model.onnx", "recording.npz"):
        shutil.copy(os.path.join(src, f), os.path.join(out, f))
    with np.load(os.path.join(src, "recording.npz")) as z:
        obs = z["obs"]
    export_onnx(early, torch.as_tensor(obs[:1]), os.path.join(out, "model-b.onnx"))
    with torch.no_grad():
        agree = float((early(torch.as_tensor(obs)).argmax(-1).numpy() == z_actions(src)).mean())
    labels = json.load(open(os.path.join(src, "labels.json")))
    labels["title"] = "LunarLander · trained vs early PPO"
    labels["description"] = (f"The same recording (the trained PPO policy landing) run through two checkpoints of that policy: "
                             f"A = trained for 1M steps (landed {land_t:.0%} of {len(seeds)} fresh seeds, mean return {ret_t:.0f}), "
                             f"B = the same run after {steps // 1000}k steps (landed {land_e:.0%}, mean return {ret_e:.0f}). "
                             f"They pick the same action on {agree:.0%} of the ticks.")
    labels["compare"] = {"model": "model-b.onnx", "a": "trained, 1M steps", "b": f"early, {steps // 1000}k steps"}
    json.dump(labels, open(os.path.join(out, "labels.json"), "w"), indent=2, ensure_ascii=False)
    td.copy_icons(out)
    json.dump({"steps": steps, "early": {"return": ret_e, "landed": land_e}, "trained": {"return": ret_t, "landed": land_t}, "agree": agree},
              open(os.path.join(RUNS, "compare_stats.json"), "w"), indent=2)


def z_actions(src):
    with np.load(os.path.join(src, "recording.npz")) as z:
        return z["actions"]


if __name__ == "__main__":
    stage = sys.argv[1]
    {"resmlp": resmlp, "lstm": lstm, "lstm-export": export_lstm, "compare": compare}[stage]()
