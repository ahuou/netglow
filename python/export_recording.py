"""Export a policy to ONNX and record what it sees, for the netglow viewer.

netglow (the browser app) needs two files per demo:

* ``model.onnx``    -- the policy network: observation in, raw outputs (logits) out.
* ``recording.npz`` -- one episode, tick by tick: the observations fed to the
  network, plus optionally the actions taken, the outputs computed at record
  time, timestamps and rendered frames.

The browser replays ``obs`` through ``model.onnx`` itself, reading every hidden
activation, so the recording only has to contain the *inputs*. ``outputs`` is
stored so the viewer can check that its own forward pass matches PyTorch's.

Vocabulary
----------
tick         one decision step of the policy (LunarLander: 50 per second).
obs          the network input at one tick (state vector, or stacked pixels).
logits       raw, un-normalised scores per discrete action; argmax = action.
opset        ONNX operator-set version; 17 is supported by onnxruntime-web.
Gemm         ONNX "general matrix multiply" node; what a torch Linear becomes
             with the legacy (TorchScript) exporter, which the viewer expects.

Use as a library::

    from export_recording import export_onnx, record_episode, save_recording
    export_onnx(net, torch.zeros(1, 8), "out/model.onnx")
    rec = record_episode(env, policy_fn, render=True)
    save_recording("out/recording.npz", **rec, fps=50)

or from the command line for a Stable-Baselines3 checkpoint::

    python export_recording.py --sb3 ppo.zip --env LunarLander-v3 --out DIR --frames
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- ONNX


def export_onnx(
    module: nn.Module,
    example_input: torch.Tensor,
    path: str,
    input_name: str = "obs",
    output_name: str = "logits",
    opset: int = 17,
) -> str:
    """Export ``module`` (obs -> raw outputs) to a single self-contained ONNX file.

    Uses the legacy TorchScript exporter (``dynamo=False``) so nn.Linear maps to a
    Gemm node, with a dynamic batch axis 0 and all weights embedded in the file.
    """
    module = module.eval().cpu()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module,
            (example_input.cpu(),),
            path,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes={input_name: {0: "N"}, output_name: {0: "N"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    # The legacy exporter never writes external data for small models, but make
    # sure: reload and re-save with everything inline.
    import onnx

    model = onnx.load(path)
    onnx.checker.check_model(model)
    onnx.save_model(model, path, save_as_external_data=False)
    return path


class _SB3Actor(nn.Module):
    """obs -> action logits (Discrete) or mean action (Box) for an SB3 ActorCriticPolicy."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        p = self.policy
        from stable_baselines3.common.torch_layers import FlattenExtractor

        if isinstance(p.pi_features_extractor, FlattenExtractor) and obs.dtype == torch.float32 and obs.dim() == 2:
            # Flat float vector: SB3's preprocessing is a no-op cast + flatten; skip
            # it so the ONNX graph starts directly at the first Gemm.
            features = obs
        else:
            features = p.extract_features(obs, p.pi_features_extractor)
        latent_pi = p.mlp_extractor.forward_actor(features)
        return p.action_net(latent_pi)


def sb3_actor(model) -> nn.Module:
    """Wrap a Stable-Baselines3 on-policy model (PPO/A2C) as obs -> raw action outputs.

    Discrete actions: features_extractor -> mlp_extractor.policy_net -> action_net
    gives the logits. Box actions: the same path gives the mean action (what
    ``deterministic=True`` uses, before clipping). Observation preprocessing
    (e.g. uint8 image /255) is the features extractor's and lands in the graph.
    """
    return _SB3Actor(model.policy).eval().cpu()


def verify_onnx(path: str, module: nn.Module, obs: np.ndarray, input_name: str = "obs") -> dict:
    """Run ``obs`` through onnxruntime (CPU) and torch; print and return the agreement."""
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {input_name: obs})[0]
    with torch.no_grad():
        torch_out = module.eval().cpu()(torch.as_tensor(obs)).numpy()
    diff = float(np.abs(onnx_out - torch_out).max())
    agree = float((onnx_out.argmax(-1) == torch_out.argmax(-1)).mean())
    print(f"[verify] {os.path.basename(path)}: max|onnx-torch| = {diff:.3g}, argmax agreement = {agree * 100:.2f}% over {len(obs)} ticks")
    return {"max_abs_diff": diff, "argmax_agreement": agree}


# ---------------------------------------------------------------------- recording


def _resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    from PIL import Image

    h, w = frame.shape[:2]
    height = max(1, round(h * width / w))
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.BILINEAR), dtype=np.uint8)


def record_episode(
    env,
    policy_fn: Callable[[np.ndarray], tuple],
    seed: int | None = None,
    max_steps: int = 1000,
    render: bool = False,
    frame_width: int = 240,
) -> dict:
    """Roll out one episode and return arrays ready for :func:`save_recording`.

    ``policy_fn(obs)`` returns ``(action, outputs)`` where ``outputs`` is the raw
    network output for that obs (or None). With ``render=True`` the env must use
    ``render_mode="rgb_array"``; frames are downscaled to ``frame_width`` px.
    Extra keys ``episode_return`` and ``terminated`` are returned for reporting
    (save_recording ignores them).
    """
    obs, _ = env.reset(seed=seed)
    rec = {"obs": [], "actions": [], "outputs": [], "frames": []}
    total, terminated = 0.0, False
    for _ in range(max_steps):
        action, outputs = policy_fn(obs)
        rec["obs"].append(np.array(obs, copy=True))
        rec["actions"].append(action)
        rec["outputs"].append(outputs)
        if render:
            rec["frames"].append(_resize_frame(env.render(), frame_width))
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    out = {
        "obs": np.stack(rec["obs"]),
        "actions": np.asarray(rec["actions"]),
        "outputs": None if rec["outputs"][0] is None else np.stack(rec["outputs"]).astype(np.float32),
        "frames": np.stack(rec["frames"]) if render else None,
        "episode_return": total,
        "terminated": bool(terminated),
    }
    return out


def save_recording(
    path: str,
    obs: np.ndarray,
    actions: np.ndarray | None = None,
    outputs: np.ndarray | None = None,
    t: np.ndarray | None = None,
    frames: np.ndarray | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    fps: float | None = None,
    **_ignored,
) -> str:
    """Write the netglow recording format (np.savez_compressed, no pickled objects).

    obs      (T, *input_shape) in the ONNX input dtype -- required
    actions  int64 (T,) discrete, or float32 (T, A) continuous
    outputs  float32 (T, n_out) raw model outputs at record time
    t        float64 (T,) seconds since episode start (defaults to tick / fps)
    fps      float64 scalar decision rate in Hz
    frames   uint8 (T, H, W, 3) renders for context
    input_names / output_names  unicode string arrays
    """
    obs = np.asarray(obs)
    T = len(obs)
    arrays = {"obs": obs}
    if actions is not None:
        a = np.asarray(actions)
        arrays["actions"] = a.astype(np.int64) if np.issubdtype(a.dtype, np.integer) else a.astype(np.float32)
    if outputs is not None:
        arrays["outputs"] = np.asarray(outputs, dtype=np.float32).reshape(T, -1)
    if t is None and fps:
        t = np.arange(T) / float(fps)
    if t is not None:
        arrays["t"] = np.asarray(t, dtype=np.float64)
    if fps is not None:
        arrays["fps"] = np.float64(fps)
    if frames is not None:
        arrays["frames"] = np.asarray(frames, dtype=np.uint8)
    if input_names is not None:
        arrays["input_names"] = np.asarray(list(input_names), dtype=str)
    if output_names is not None:
        arrays["output_names"] = np.asarray(list(output_names), dtype=str)
    for k, v in arrays.items():
        assert v.dtype != object, k
        if k not in ("fps", "input_names", "output_names"):
            assert len(v) == T, f"{k} has length {len(v)}, expected {T}"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def write_labels(path: str, title: str, inputs, outputs, description: str = "", icons=None,
                 output_mode: str = "categorical", input_units=None) -> str:
    """Write labels.json describing the input/output names for the viewer."""
    data = {"title": title, "description": description, "inputs": list(inputs), "outputs": list(outputs),
            "icons": list(icons) if icons else None, "output_mode": output_mode, "input_units": input_units}
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(description="Export an SB3 checkpoint + recorded episode(s) for netglow.")
    ap.add_argument("--sb3", required=True, help="path to an SB3 PPO/A2C .zip")
    ap.add_argument("--algo", default="PPO", choices=["PPO", "A2C"])
    ap.add_argument("--env", required=True, help="gymnasium env id, e.g. LunarLander-v3")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--episodes", type=int, default=1, help="episodes to concatenate into the recording")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=600, help="cap on recorded ticks per episode")
    ap.add_argument("--frames", action="store_true", help="store rgb renders (~240 px wide)")
    ap.add_argument("--fps", type=float, default=None, help="decision rate (default: env metadata render_fps)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    import gymnasium as gym
    import stable_baselines3 as sb3

    model = getattr(sb3, args.algo).load(args.sb3, device="cpu")
    actor = sb3_actor(model)
    env = gym.make(args.env, render_mode="rgb_array" if args.frames else None)
    fps = args.fps or float(env.metadata.get("render_fps") or 0) or None

    def policy_fn(obs):
        with torch.no_grad():
            out = actor(torch.as_tensor(obs[None]))[0].numpy()
        if isinstance(env.action_space, gym.spaces.Discrete):
            return int(out.argmax()), out
        return np.clip(out, env.action_space.low, env.action_space.high), out

    eps = [record_episode(env, policy_fn, seed=args.seed + i, max_steps=args.max_steps, render=args.frames)
           for i in range(args.episodes)]
    for i, e in enumerate(eps):
        print(f"[record] episode {i}: {len(e['obs'])} ticks, return {e['episode_return']:.1f}")
    cat = lambda k: None if eps[0][k] is None else np.concatenate([e[k] for e in eps])
    obs = cat("obs").astype(model.observation_space.dtype)

    os.makedirs(args.out, exist_ok=True)
    onnx_path = os.path.join(args.out, "model.onnx")
    export_onnx(actor, torch.as_tensor(obs[:1]), onnx_path)
    verify_onnx(onnx_path, actor, obs)
    save_recording(os.path.join(args.out, "recording.npz"), obs, cat("actions"), cat("outputs"),
                   frames=cat("frames"), fps=fps)
    n_in = int(np.prod(obs.shape[1:]))
    n_out = eps[0]["outputs"].shape[-1]
    write_labels(os.path.join(args.out, "labels.json"), args.title or f"{args.env} · {args.algo}",
                 [f"obs[{i}]" for i in range(n_in)] if obs.ndim == 2 else [f"channel {i}" for i in range(obs.shape[1])],
                 [f"out[{i}]" for i in range(n_out)],
                 output_mode="categorical" if isinstance(env.action_space, gym.spaces.Discrete) else "continuous")
    print(f"[done] wrote {args.out}/model.onnx, recording.npz, labels.json")


if __name__ == "__main__":
    main()
