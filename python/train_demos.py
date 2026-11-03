"""Reproduce netglow's bundled demos end to end.

    python train_demos.py all            # everything below, in order
    python train_demos.py ppo            # train the PPO MLP expert   -> runs/ppo_lunar.zip
    python train_demos.py mlp            # export demo A              -> public/demos/lunar-mlp/
    python train_demos.py cnn            # BC + DAgger pixel pilot    -> runs/cnn_pilot.pt
    python train_demos.py cnn-export     # export demo B              -> public/demos/lunar-cnn/

Demo A: Stable-Baselines3 PPO on the 8-number LunarLander-v3 state.
Demo B: a small CNN that flies from pixels only, taught by imitating demo A
(behaviour cloning = supervised learning on the expert's actions; DAgger =
let the student fly, ask the expert what it *would* have done in the states the
student reaches, add those labels, retrain).

Vocabulary
----------
expert       the PPO policy from demo A; it sees the true state.
student      the CNN; it sees only two stacked 64x64 grayscale frames.
frame stack  the last 2 frames as 2 channels, so motion (velocity) is visible.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from export_recording import export_onnx, record_episode, save_recording, sb3_actor, verify_onnx, write_labels

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(HERE, "runs")
DEMOS = os.path.join(ROOT, "public", "demos")
ICONS = os.path.join(HERE, "icons")
ENV_ID = "LunarLander-v3"
FPS = 50.0  # LunarLander decides at its physics rate, 50 Hz
OUTPUT_NAMES = ["nothing", "left engine", "main engine", "right engine"]
ICON_PATHS = ["icons/noop.svg", "icons/left.svg", "icons/main.svg", "icons/right.svg"]
PPO_PATH = os.path.join(RUNS, "ppo_lunar.zip")
CNN_PATH = os.path.join(RUNS, "cnn_pilot.pt")


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ demo A: PPO


def train_ppo(timesteps: int) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import SubprocVecEnv

    torch.set_num_threads(4)
    # rl-zoo3 hyperparameters for LunarLander.
    venv = make_vec_env(ENV_ID, n_envs=16, seed=0, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=64, gae_lambda=0.98, gamma=0.999,
                n_epochs=4, ent_coef=0.01, verbose=1, seed=0, device="cpu")
    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    os.makedirs(RUNS, exist_ok=True)
    model.save(PPO_PATH)
    venv.close()
    mean, std = evaluate_policy(model, gym.make(ENV_ID), n_eval_episodes=50, deterministic=True)
    log(f"[ppo] {timesteps} steps in {time.time() - t0:.0f}s; eval (50 eps, deterministic) return {mean:.1f} +- {std:.1f}")


def load_expert():
    from stable_baselines3 import PPO

    actor = sb3_actor(PPO.load(PPO_PATH, device="cpu"))

    def act(state: np.ndarray) -> tuple[int, np.ndarray]:
        with torch.no_grad():
            logits = actor(torch.as_tensor(state, dtype=torch.float32)[None])[0].numpy()
        return int(logits.argmax()), logits

    return actor, act


def landed(env) -> bool:
    """True if the episode ended at rest (LunarLander's +100 'landed' outcome)."""
    u = env.unwrapped
    return (not u.game_over) and abs(u.lander.position.x - 10.0) < 20 and not u.lander.awake


def pick_episode(env, policy_fn, render: bool, seeds, lo=200, hi=500):
    """Try several seeds; prefer a landing inside [lo, hi] ticks, else the best return."""
    best = None
    for s in seeds:
        rec = record_episode(env, policy_fn, seed=int(s), max_steps=600, render=False)
        rec["landed"] = rec["terminated"] and landed(env)
        rec["seed"] = int(s)
        T = len(rec["obs"])
        score = rec["episode_return"] + (1000 if rec["landed"] and lo <= T <= hi else 0)
        log(f"  seed {s}: T={T} return={rec['episode_return']:.1f} landed={rec['landed']}")
        if best is None or score > best[0]:
            best = (score, rec)
        if rec["landed"] and lo <= T <= hi and rec["episode_return"] > 200:
            break
    rec = best[1]
    if render:  # deterministic env + policy: replaying the seed reproduces it, now with frames
        again = record_episode(env, policy_fn, seed=rec["seed"], max_steps=600, render=True)
        assert len(again["obs"]) == len(rec["obs"]) and np.allclose(again["obs"], rec["obs"])
        again.update(landed=rec["landed"], seed=rec["seed"])
        rec = again
    return rec


def copy_icons(out: str) -> None:
    dst = os.path.join(out, "icons")
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(ICONS):
        shutil.copy(os.path.join(ICONS, name), dst)


def export_mlp() -> None:
    out = os.path.join(DEMOS, "lunar-mlp")
    actor, act = load_expert()
    env = gym.make(ENV_ID, render_mode="rgb_array")
    log("[mlp] picking an episode")
    rec = pick_episode(env, act, render=True, seeds=range(100))
    obs = rec["obs"].astype(np.float32)
    os.makedirs(out, exist_ok=True)
    export_onnx(actor, torch.as_tensor(obs[:1]), os.path.join(out, "model.onnx"))
    v = verify_onnx(os.path.join(out, "model.onnx"), actor, obs)
    inputs = ["x", "y", "x speed", "y speed", "angle", "spin", "left leg", "right leg"]
    save_recording(os.path.join(out, "recording.npz"), obs, rec["actions"], rec["outputs"],
                   frames=rec["frames"], fps=FPS, input_names=inputs, output_names=OUTPUT_NAMES)
    write_labels(os.path.join(out, "labels.json"), "LunarLander · PPO MLP",
                 inputs, OUTPUT_NAMES, icons=ICON_PATHS,
                 description="A PPO agent (Stable-Baselines3, 2x64 tanh MLP) landing from the 8-number state: "
                             "position, velocity, tilt, spin and leg contacts. Outputs are logits for its 4 discrete actions.",
                 input_units=["", "", "", "", "rad", "", "0/1", "0/1"])
    copy_icons(out)
    log(f"[mlp] seed {rec['seed']}: T={len(obs)} return={rec['episode_return']:.1f} landed={rec['landed']} "
        f"verify={v}")


# -------------------------------------------------------------- demo B: pixels


CROP = 320  # side of the square crop (px of the 600x400 render), top-aligned, centred in x


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """600x400 RGB render -> 64x64 uint8 grayscale of a CROPxCROP square (top, centred)."""
    from PIL import Image

    w = rgb.shape[1]
    x0 = (w - CROP) // 2
    img = Image.fromarray(np.ascontiguousarray(rgb[:CROP, x0:x0 + CROP])).convert("L").resize((64, 64), Image.BOX)
    return np.asarray(img, dtype=np.uint8)


class PixelObs(gym.Wrapper):
    """Replace the state with the last 2 preprocessed frames, shape (2, 64, 64) uint8,
    and hold each action for ``repeat`` physics ticks (Atari-style frame skip).

    Why frame skip: at 50 Hz the lander moves far less than one 64x64 pixel per
    tick, and the PPO expert dithers between actions tick to tick, so (t-1, t)
    pairs carry almost no velocity signal and the labels look random. Deciding
    every ``repeat`` ticks makes motion visible between the two frames; the
    expert still lands 20/20 with repeat=4.

    The true state stays available as ``self.state`` so the expert can label it.
    ``render()`` returns the cached full-resolution frame (no second draw).
    """

    def __init__(self, env, repeat: int = 4):
        super().__init__(env)
        self.repeat = repeat
        self.observation_space = gym.spaces.Box(0, 255, (2, 64, 64), np.uint8)

    def _frame(self):
        self.rgb = self.env.render()
        return preprocess(self.rgb)

    def reset(self, **kw):
        self.state, info = self.env.reset(**kw)
        f = self._frame()
        self.stack = np.stack([f, f])
        return self.stack.copy(), info

    def step(self, action):
        total = 0.0
        for _ in range(self.repeat):
            self.state, r, te, tr, info = self.env.step(action)
            total += r
            if te or tr:
                break
        self.stack = np.stack([self.stack[1], self._frame()])
        return self.stack.copy(), total, te, tr, info

    def render(self):
        return self.rgb


class CNNPilot(nn.Module):
    """uint8 (N, 2, 64, 64) frames -> 4 action logits. Scaling to [0, 1] is inside the graph."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 8, stride=4), nn.ReLU(),   # -> 16 x 15 x 15
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),  # -> 32 x 6 x 6
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(32 * 6 * 6, 128), nn.ReLU(), nn.Linear(128, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x.float() / 255.0))


def collect(env, expert, student=None, episodes=30, seed0=0, beta=1.0, max_steps=600):
    """Roll out episodes, labelling every visited state with the expert's logits.

    Who flies: the expert with probability ``beta`` per tick, else the student
    (DAgger mixing). Returns pixel obs, expert logits and per-episode returns.
    """
    X, L, rets = [], [], []
    rng = np.random.default_rng(seed0)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        total = 0.0
        for _ in range(max_steps):
            a_exp, logits = expert(env.state)
            X.append(obs)
            L.append(logits)
            a = a_exp
            if student is not None and rng.random() >= beta:
                a = student(obs)[0]
            obs, r, te, tr, _ = env.step(a)
            total += r
            if te or tr:
                break
        rets.append(total)
    return np.stack(X), np.stack(L).astype(np.float32), rets


def fit(net, X, L, device, epochs=30, bs=256, lr=1e-3):
    """Distil the expert: cross-entropy against softmax(expert logits) (soft labels
    carry how sure the expert was, which helps where it dithers between actions).
    Returns (train, held-out) accuracy of argmax vs the expert's argmax."""
    n = len(X)
    idx = np.random.default_rng(0).permutation(n)
    va, tr = idx[: n // 10], idx[n // 10:]
    Xt = torch.as_tensor(X).to(device)
    Pt = torch.softmax(torch.as_tensor(L), -1).to(device)
    Yt = Pt.argmax(-1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        net.train()
        perm = torch.as_tensor(np.random.permutation(tr)).to(device)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = -(Pt[b] * torch.log_softmax(net(Xt[b]), -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        def acc(ix):
            ix = torch.as_tensor(ix[:20000]).to(device)
            return (net(Xt[ix]).argmax(-1) == Yt[ix]).float().mean().item()
        return acc(tr), acc(va)


def make_student_fn(net, device):
    def act(obs):
        with torch.no_grad():
            logits = net(torch.as_tensor(obs[None]).to(device))[0].cpu().numpy()
        return int(logits.argmax()), logits

    return act


def evaluate(env, policy_fn, seeds):
    rets, lands = [], []
    for s in seeds:
        rec = record_episode(env, policy_fn, seed=int(s), max_steps=600)
        rets.append(rec["episode_return"])
        lands.append(rec["terminated"] and landed(env))
    return float(np.mean(rets)), float(np.mean(lands))


def train_cnn(bc_episodes: int, dagger_rounds: int, dagger_episodes: int, repeat: int, epochs: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    torch.set_num_threads(8)
    _, expert = load_expert()
    env = PixelObs(gym.make(ENV_ID, render_mode="rgb_array"), repeat=repeat)
    t0 = time.time()
    X, L, rets = collect(env, expert, episodes=bc_episodes, seed0=10_000)
    log(f"[cnn] repeat={repeat}: {bc_episodes} expert episodes, {len(X)} frames, expert return {np.mean(rets):.1f} ({time.time() - t0:.0f}s)")
    net = CNNPilot().to(device)
    acc = fit(net, X, L, device, epochs)
    student = make_student_fn(net, device)
    stats = {"repeat": repeat, "rounds": []}
    eval_seeds = range(50_000, 50_020)
    ret, land = evaluate(env, student, eval_seeds)
    log(f"[cnn] BC: train acc {acc[0]:.3f}, val acc {acc[1]:.3f}; closed loop return {ret:.1f}, landed {land:.0%}")
    stats["rounds"].append({"round": 0, "frames": len(X), "train_acc": acc[0], "val_acc": acc[1], "return": ret, "landed": land})
    best = (ret, {k: v.detach().clone() for k, v in net.state_dict().items()})
    for r in range(1, dagger_rounds + 1):
        Xn, Ln, srets = collect(env, expert, student, episodes=dagger_episodes, seed0=20_000 + 1000 * r, beta=0.5 ** r)
        X, L = np.concatenate([X, Xn]), np.concatenate([L, Ln])
        net = CNNPilot().to(device)
        acc = fit(net, X, L, device, epochs)
        student = make_student_fn(net, device)
        ret, land = evaluate(env, student, eval_seeds)
        log(f"[cnn] DAgger {r}: +{len(Xn)} frames (student return while collecting {np.mean(srets):.1f}); "
            f"train acc {acc[0]:.3f}, val acc {acc[1]:.3f}; closed loop return {ret:.1f}, landed {land:.0%}")
        stats["rounds"].append({"round": r, "frames": len(X), "train_acc": acc[0], "val_acc": acc[1], "return": ret, "landed": land})
        if ret > best[0]:
            best = (ret, {k: v.detach().clone() for k, v in net.state_dict().items()})
    os.makedirs(RUNS, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in best[1].items()}, "repeat": repeat, "crop": CROP}, CNN_PATH)
    stats["best_eval_return"] = best[0]
    stats["wall_s"] = time.time() - t0
    with open(os.path.join(RUNS, "cnn_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    log(f"[cnn] kept best model (eval return {best[0]:.1f}); {time.time() - t0:.0f}s total")


def export_cnn() -> None:
    out = os.path.join(DEMOS, "lunar-cnn")
    net = CNNPilot()
    ckpt = torch.load(CNN_PATH)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    repeat = ckpt["repeat"]
    global CROP
    CROP = ckpt["crop"]
    act = make_student_fn(net, "cpu")
    env = PixelObs(gym.make(ENV_ID, render_mode="rgb_array"), repeat=repeat)
    log("[cnn-export] picking an episode flown by the CNN")
    rec = pick_episode(env, act, render=True, seeds=range(60), lo=200 // repeat, hi=500 // repeat)
    obs = rec["obs"].astype(np.uint8)
    os.makedirs(out, exist_ok=True)
    export_onnx(net, torch.as_tensor(obs[:1]), os.path.join(out, "model.onnx"))
    v = verify_onnx(os.path.join(out, "model.onnx"), net, obs)
    inputs = ["frame t-1", "frame t"]
    assert (rec["outputs"].argmax(-1) == rec["actions"]).all()
    save_recording(os.path.join(out, "recording.npz"), obs, rec["actions"], rec["outputs"],
                   frames=rec["frames"], fps=FPS / repeat, input_names=inputs, output_names=OUTPUT_NAMES)
    write_labels(os.path.join(out, "labels.json"), "LunarLander · CNN from pixels",
                 inputs, OUTPUT_NAMES, icons=ICON_PATHS,
                 description="A small CNN that flies from pixels only: the last two 64x64 grayscale frames, deciding every 4 physics ticks (12.5 Hz). "
                             "Taught by imitating the PPO expert (behaviour cloning + DAgger); every action in this "
                             "episode is the CNN's own choice.",
                 input_units=None)
    copy_icons(out)
    log(f"[cnn-export] seed {rec['seed']}: T={len(obs)} return={rec['episode_return']:.1f} landed={rec['landed']} verify={v}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["all", "ppo", "mlp", "cnn", "cnn-export"])
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--bc-episodes", type=int, default=150)
    ap.add_argument("--dagger-rounds", type=int, default=5)
    ap.add_argument("--dagger-episodes", type=int, default=60)
    ap.add_argument("--repeat", type=int, default=4, help="CNN decides every N physics ticks")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    if args.stage in ("all", "ppo"):
        train_ppo(args.timesteps)
    if args.stage in ("all", "mlp"):
        export_mlp()
    if args.stage in ("all", "cnn"):
        train_cnn(args.bc_episodes, args.dagger_rounds, args.dagger_episodes, args.repeat, args.epochs)
    if args.stage in ("all", "cnn-export"):
        export_cnn()


if __name__ == "__main__":
    sys.exit(main())
