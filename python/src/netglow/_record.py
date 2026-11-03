"""Recording what a policy sees, tick by tick, in the format the netglow viewer replays.

Vocabulary
----------
tick       one decision step of the policy: one observation in, one output out.
obs        the network input at one tick; with several ONNX inputs, one array per input,
           stored as ``obs_<input name>``.
outputs    the raw network output at record time; the viewer re-runs the ONNX model on ``obs``
           and compares (the "Match" pill), so you know the picture shows the network that acted.
frames     the env's own render (rgb), shown beside the network; optional.
"""

from __future__ import annotations

import collections
import json
import os
import time
from typing import Any, Callable, Sequence

import numpy as np

# Input / output names for a few well-known Gymnasium envs, used for labels.json.
KNOWN_ENVS: dict[str, dict] = {
    "LunarLander": {
        "inputs": ["x", "y", "x speed", "y speed", "angle", "spin", "left leg", "right leg"],
        "outputs": ["nothing", "left engine", "main engine", "right engine"],
        "input_units": ["", "", "", "", "rad", "", "0/1", "0/1"],
    },
    "LunarLanderContinuous": {
        "inputs": ["x", "y", "x speed", "y speed", "angle", "spin", "left leg", "right leg"],
        "outputs": ["main engine", "side engines"],
    },
    "CartPole": {"inputs": ["cart position", "cart speed", "pole angle", "pole spin"], "outputs": ["push left", "push right"]},
    "Acrobot": {"inputs": ["cos θ1", "sin θ1", "cos θ2", "sin θ2", "θ1 speed", "θ2 speed"], "outputs": ["torque −1", "torque 0", "torque +1"]},
    "MountainCar": {"inputs": ["position", "velocity"], "outputs": ["push left", "no push", "push right"]},
    "MountainCarContinuous": {"inputs": ["position", "velocity"], "outputs": ["force"]},
    "Pendulum": {"inputs": ["cos θ", "sin θ", "spin"], "outputs": ["torque"]},
}


def known_labels(env_id: str | None) -> dict:
    if not env_id:
        return {}
    base = env_id.split("/")[-1].rsplit("-v", 1)[0]
    return dict(KNOWN_ENVS.get(base, {}))


# ------------------------------------------------------------------------------ files


def save_recording(
    path: str,
    obs: np.ndarray | dict,
    actions: np.ndarray | None = None,
    outputs: np.ndarray | None = None,
    t: np.ndarray | None = None,
    frames: np.ndarray | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    fps: float | None = None,
    episode_starts: np.ndarray | None = None,
    init: dict | None = None,
    **_ignored,
) -> str:
    """Write the netglow recording format (np.savez_compressed, no pickled objects).

    obs      (T, *input_shape) in the ONNX input dtype. A model with several inputs takes a dict
             {ONNX input name: (T, *shape)}, stored as ``obs_<name>``
    actions  int64 (T,) discrete, or float32 (T, A) continuous
    outputs  float32 (T, n_out) raw model outputs at record time
    t        float64 (T,) seconds (defaults to tick / fps)
    fps      decision rate in Hz
    frames   uint8 (T, H, W, 3) renders for context
    episode_starts  uint8 (T,), 1 on the first tick of each episode (recurrent models reset their state there)
    init     {state input name: array}: the recurrent state fed into tick 0, stored as ``init_<name>`` (default zeros)
    """
    if isinstance(obs, dict):
        arrays = {f"obs_{k}": np.asarray(v) for k, v in obs.items()}
        T = len(next(iter(arrays.values())))
    else:
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
    if episode_starts is not None:
        arrays["episode_starts"] = np.asarray(episode_starts, dtype=np.uint8)
    for k, v in (init or {}).items():
        arrays[f"init_{k}"] = np.asarray(v, dtype=np.float32)
    if input_names is not None:
        arrays["input_names"] = np.asarray(list(input_names), dtype=str)
    if output_names is not None:
        arrays["output_names"] = np.asarray(list(output_names), dtype=str)
    for k, v in arrays.items():
        assert v.dtype != object, k
        if k not in ("fps", "input_names", "output_names") and not k.startswith("init_"):
            assert len(v) == T, f"{k} has length {len(v)}, expected {T}"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def labels_dict(title: str, inputs, outputs, description: str = "", icons=None,
                output_mode: str = "categorical", input_units=None) -> dict:
    return {"title": title, "description": description, "inputs": list(inputs), "outputs": list(outputs),
            "icons": list(icons) if icons else None, "output_mode": output_mode, "input_units": input_units}


def write_labels(path: str, title: str, inputs, outputs, description: str = "", icons=None,
                 output_mode: str = "categorical", input_units=None) -> str:
    """Write labels.json (input / output names, icons) for the viewer."""
    with open(path, "w") as f:
        json.dump(labels_dict(title, inputs, outputs, description, icons, output_mode, input_units), f, indent=2, ensure_ascii=False)
    return path


def resize_frame(frame: np.ndarray, width: int = 240) -> np.ndarray:
    """Downscale an rgb render to ``width`` px (PIL if installed, else block averaging)."""
    frame = np.asarray(frame)
    h, w = frame.shape[:2]
    if w <= width:
        return frame.astype(np.uint8)
    height = max(1, round(h * width / w))
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(frame.astype(np.uint8)).resize((width, height), Image.BILINEAR), dtype=np.uint8)
    except ImportError:
        ys = (np.arange(height) * h / height).astype(int)
        xs = (np.arange(width) * w / width).astype(int)
        return frame[ys][:, xs].astype(np.uint8)


# --------------------------------------------------------------------------- recorder


class Recorder:
    """Collect ticks by hand (any loop, no Gymnasium needed) and write a netglow folder.

    ::

        rec = netglow.Recorder(net, out_dir="run", fps=50)    # live=True also streams each tick
        for ...:
            out = net(obs)                    # or let the recorder run the model: rec.add(obs)
            rec.add(obs, output=out, action=a)
        rec.save()                            # run/model.onnx, run/recording.npz, run/labels.json

    model         torch module, SB3 model / ``.zip``, or None (then only the recording is written).
                  Exported on the first ``add`` using that observation as the example input.
    obs per tick  an array, a tuple of arrays (forward's positional order) or a dict {name: array}.
    output        raw model output for that tick; computed with the model when omitted.
    live          stream every tick to the viewer over a localhost WebSocket (see netglow.live).
    keep          ticks kept in memory for save(): all by default, the last 2000 when live with no out_dir.
    state         recurrent models: the initial state tensors (e.g. LSTM ``(zeros(1, 1, H), zeros(1, 1, H))``) for a
                  step function ``forward(obs..., h0, c0) -> (outputs, h, c)``. The state is threaded here and in
                  the viewer and reset at episode starts (``episode_start=True`` on add / infer).
    """

    def __init__(
        self,
        model=None,
        out_dir: str | None = "netglow_run",
        example_inputs: Any = None,
        input_names: Sequence[str] | None = None,
        labels: dict | None = None,
        title: str | None = None,
        fps: float | None = None,
        output_mode: str | None = None,
        live: bool = False,
        port: int = 8765,
        host: str = "localhost",
        frame_width: int = 240,
        quiet: bool = False,
        web: str | None = None,
        keep: int | None = None,
        state: Sequence[Any] | None = None,
        state_names: Sequence[str] | None = None,
    ):
        from ._sb3 import from_sb3, is_sb3

        self.model = from_sb3(model) if is_sb3(model) else model
        self.out_dir = out_dir
        self.example_inputs = example_inputs
        self.input_names = list(input_names) if input_names else getattr(self.model, "input_names", None)
        self.labels = dict(labels or {})
        self.title = title
        self.fps = fps
        self.output_mode = output_mode or getattr(self.model, "output_mode", None)
        self.frame_width = frame_width
        self.quiet = quiet
        # every tick when writing files; a live-only session keeps the last 2000 so memory stays flat
        self.ticks: collections.deque[dict] = collections.deque(maxlen=keep if keep else (None if out_dir or not live else 2000))
        self.onnx: bytes | None = None
        # recurrent step function: forward(obs..., *state) -> (outputs, *new_state); state threaded here and in the viewer
        self.init_state = None if state is None else [np.asarray(v.detach().cpu() if hasattr(v, "detach") else v, dtype=np.float32) for v in state]
        self.state_names = list(state_names or ["h0", "c0", "s2", "s3"][: len(self.init_state or [])])
        self.state = None
        self._start: bool | None = None  # an episode start given to infer(), for the add() that follows
        self.server = None
        if live:
            from ._live import LiveServer

            self.server = LiveServer(host=host, port=port, quiet=quiet, web=web)

    # ---- per tick
    def _split(self, obs) -> dict[str, np.ndarray]:
        if isinstance(obs, dict):
            names = self.input_names or list(obs.keys())
            arrs = [obs[k] for k in names]
        elif isinstance(obs, (tuple, list)):
            arrs = list(obs)
            names = self.input_names or (["obs"] if len(arrs) == 1 else [f"in{i}" for i in range(len(arrs))])
        else:
            arrs = [obs]
            names = self.input_names or ["obs"]
        if self.input_names is None:
            self.input_names = list(names)
        out = {}
        for n, a in zip(names, arrs):
            if hasattr(a, "detach"):
                a = a.detach().cpu().numpy()
            a = np.asarray(a)
            out[n] = a.astype(np.float32) if a.dtype == np.float64 else a
        return out

    def _ensure_model(self, x: dict[str, np.ndarray]) -> None:
        if self.onnx is not None or self.model is None:
            return
        from ._export import export_bytes

        st = dict(state=self.init_state, state_names=self.state_names) if self.init_state is not None else {}
        if self.example_inputs is not None:
            self.onnx = export_bytes(self.model, self.example_inputs, self.input_names, **st)
        else:
            self.onnx = export_bytes(self.model, tuple(x.values()), self.input_names, batched=False, **st)

    def _run(self, x: dict[str, np.ndarray], start: bool) -> np.ndarray:
        import torch

        args = [torch.as_tensor(v)[None] for v in x.values()]
        if self.init_state is not None:
            if start or self.state is None:
                self.state = [v.copy() for v in self.init_state]
            self.state_in = [v.copy() for v in self.state]
            args += [torch.as_tensor(v) for v in self.state]
        with torch.no_grad():
            y = self.model.eval()(*args)
        if self.init_state is not None:
            self.state = [np.asarray(v.cpu(), dtype=np.float32) for v in y[1:1 + len(self.init_state)]]
        if isinstance(y, (tuple, list)):
            y = y[0]
        return y[0].cpu().numpy()

    def _is_start(self, episode_start: bool | None) -> bool:
        return bool(episode_start) or not self.ticks and self._start is None and self.state is None

    def infer(self, obs, episode_start: bool | None = None) -> np.ndarray | None:
        """The model's raw output for one observation (exports the model on first use). Recurrent models step
        their state; ``episode_start=True`` resets it first. Pass the result to ``add(obs, output=...)``."""
        x = self._split(obs)
        self._ensure_model(x)
        if self.model is None:
            return None
        start = self._is_start(episode_start)
        self._start = start
        return np.asarray(self._run(x, start), dtype=np.float32).reshape(-1)

    def add(self, obs, output=None, action=None, frame=None, episode_start: bool | None = None) -> np.ndarray | None:
        """Record one tick. Returns the output (computed with the model if not given).
        ``episode_start``: this tick begins an episode (recurrent models reset their state here)."""
        x = self._split(obs)
        self._ensure_model(x)
        if output is None and self.model is not None:
            start = self._is_start(episode_start)
            output = self._run(x, start)
        else:
            start = bool(episode_start) or bool(self._start) or not self.ticks
        self._start = None
        if output is not None:
            if hasattr(output, "detach"):
                output = output.detach().cpu().numpy()
            output = np.asarray(output, dtype=np.float32).reshape(-1)
        if frame is not None:
            frame = resize_frame(frame, self.frame_width)
        tick = {"obs": x, "output": output, "action": None if action is None else np.asarray(action), "frame": frame,
                "start": start}
        if self.init_state is not None and getattr(self, "state_in", None) is not None:
            tick["state_in"] = dict(zip(self.state_names, self.state_in))
        self.ticks.append(tick)
        if self.server is not None:
            if not self.server.has_model and self.onnx is not None:
                self.server.set_model(self.onnx, self._labels_for(), self.fps)
            self.server.push(tick)
        return output

    # ---- output
    def _labels_for(self, x: dict[str, np.ndarray] | None = None) -> dict:
        x = x or (self.ticks[0]["obs"] if self.ticks else {})
        names = []
        for n, a in x.items():
            names += [n] if a.ndim >= 2 else [f"{n}[{i}]" if len(x) > 1 else f"obs[{i}]" for i in range(a.size)]
        n_out = 0
        if self.ticks and self.ticks[0]["output"] is not None:
            n_out = self.ticks[0]["output"].size
        L = labels_dict(self.title or "netglow recording", names, [f"out[{i}]" for i in range(n_out)],
                        output_mode=self.output_mode or "categorical")
        given = {k: v for k, v in self.labels.items() if v is not None}
        # names that don't fit the model (e.g. a known env's state names on a pixel policy) are dropped
        for k, n in (("inputs", len(names)), ("input_units", len(names)), ("outputs", n_out)):
            if k in given and len(given[k]) != n:
                given.pop(k)
        L.update(given)
        if not self.output_mode and "output_mode" not in self.labels:
            L.pop("output_mode")  # let the viewer decide from the model / actions
        return L

    def arrays(self) -> dict:
        """The recording as the dict :func:`save_recording` takes."""
        if not self.ticks:
            raise ValueError("nothing recorded")
        names = list(self.ticks[0]["obs"].keys())
        obs = {n: np.stack([t["obs"][n] for t in self.ticks]) for n in names}
        has = lambda k: all(t[k] is not None for t in self.ticks)
        starts = np.array([1] + [int(t.get("start", False)) for t in list(self.ticks)[1:]], np.uint8)
        return {
            "obs": obs[names[0]] if len(names) == 1 else obs,
            "outputs": np.stack([t["output"] for t in self.ticks]) if has("output") else None,
            "actions": np.stack([t["action"] for t in self.ticks]) if has("action") else None,
            "frames": np.stack([t["frame"] for t in self.ticks]) if has("frame") else None,
            "fps": self.fps,
            "episode_starts": starts if self.init_state is not None or starts[1:].any() else None,
            # the state fed into the first kept tick, so a replay that starts mid-episode is still exact
            "init": self.ticks[0].get("state_in") or {},
        }

    def save(self, out_dir: str | None = None) -> str:
        """Write model.onnx (if there is a model), recording.npz and labels.json; returns the folder."""
        d = out_dir or self.out_dir or "netglow_run"
        os.makedirs(d, exist_ok=True)
        a = self.arrays()
        if isinstance(a["obs"], dict) or self.input_names and self.input_names != ["obs"]:
            # several inputs, or one with a custom name: obs_<name> matches the ONNX input
            a["obs"] = a["obs"] if isinstance(a["obs"], dict) else {self.input_names[0]: a["obs"]}
        if self.onnx is not None:
            with open(os.path.join(d, "model.onnx"), "wb") as f:
                f.write(self.onnx)
        save_recording(os.path.join(d, "recording.npz"), **a)
        with open(os.path.join(d, "labels.json"), "w") as f:
            json.dump(self._labels_for(), f, indent=2, ensure_ascii=False)
        if not self.quiet:
            print(f"[netglow] wrote {len(self.ticks)} ticks to {d}/ (model.onnx, recording.npz, labels.json)."
                  f" Open https://ahuou.github.io/netglow/ and drop the files in.")
        return d

    def close(self) -> None:
        if self.server is not None:
            self.server.close()
            self.server = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            if self.out_dir and self.ticks and exc[0] is None:
                self.save()
        finally:
            self.close()


# ----------------------------------------------------------------------------- record


def _action_from(output: np.ndarray, action_space):
    from gymnasium import spaces

    if isinstance(action_space, spaces.Discrete):
        return int(np.argmax(output)) + int(getattr(action_space, "start", 0))
    if isinstance(action_space, spaces.Box):
        return np.clip(output.reshape(action_space.shape), action_space.low, action_space.high).astype(action_space.dtype)
    raise TypeError(f"can't turn outputs into a {action_space}; pass act=lambda obs, out: action")


def record(
    model,
    env,
    steps: int = 1000,
    out_dir: str | None = "netglow_run",
    seed: int | None = 0,
    frames: bool = True,
    act: Callable | None = None,
    live: bool = False,
    port: int = 8765,
    realtime: bool | None = None,
    fps: float | None = None,
    title: str | None = None,
    algo: str | None = None,
    quiet: bool = False,
    web: str | None = None,
    state=None,
) -> Recorder:
    """Run a policy in a Gymnasium env and write ``out_dir/{model.onnx, recording.npz, labels.json}``.

    model     SB3 model or ``.zip`` (actions from SB3's deterministic ``predict``), or a torch
              module obs -> outputs (Discrete: argmax, Box: clipped outputs, or your ``act``).
    env       a gymnasium.Env, or an env id (made with render_mode="rgb_array" when frames=True).
    steps     ticks to record; episodes are reset and concatenated.
    frames    store the env's rgb render (~240 px wide) beside the network.
    live      also stream every tick to the browser (realtime pacing at the env fps by default).
    state     recurrent torch step function forward(obs, h0[, c0]) -> (outputs, h[, c]): the initial state
              tensors; reset at every episode start (see :class:`Recorder`).
    Returns the :class:`Recorder` (``.save()`` already called when out_dir is set).
    """
    import gymnasium as gym
    from ._sb3 import from_sb3, is_sb3

    env_id = env if isinstance(env, str) else getattr(getattr(env, "spec", None), "id", None)
    if isinstance(env, str):
        env = gym.make(env, render_mode="rgb_array" if frames else None)
    frames = frames and getattr(env, "render_mode", None) == "rgb_array"
    policy = from_sb3(model, algo) if is_sb3(model) else model
    fps = fps or float(env.metadata.get("render_fps") or 0) or None
    lab = known_labels(env_id)
    space = env.observation_space
    names = getattr(policy, "input_names", None)
    if names is None and isinstance(space, gym.spaces.Dict):
        names = list(space.spaces.keys())
    rec = Recorder(policy, out_dir=out_dir, input_names=names, title=title or (f"{env_id}" if env_id else None), fps=fps,
                   labels={k: v for k, v in lab.items()}, live=live, port=port, quiet=quiet, web=web, state=state)
    realtime = live if realtime is None else realtime
    obs, _ = env.reset(seed=seed)
    ret, episodes, period = 0.0, 1, 1.0 / fps if fps else 0.0
    t_next = time.perf_counter()
    start = True
    try:
        for _ in range(steps):
            frame = env.render() if frames else None
            x = {k: obs[k] for k in names} if isinstance(obs, dict) else obs
            out = rec.infer(x, episode_start=start)
            if act is not None:
                a = act(obs, out)
            elif hasattr(policy, "predict"):
                a = policy.predict(obs)
            else:
                a = _action_from(out, env.action_space)
            rec.add(x, output=out, action=a, frame=frame, episode_start=start)
            start = False
            obs, r, term, trunc, _ = env.step(a)
            ret += float(r)
            if term or trunc:
                if not quiet:
                    print(f"[netglow] episode {episodes}: return {ret:.1f}")
                obs, _ = env.reset()
                ret, episodes, start = 0.0, episodes + 1, True
            if realtime and period:
                t_next += period
                dt = t_next - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                else:
                    t_next = time.perf_counter()
    except KeyboardInterrupt:
        if not quiet:
            print(f"[netglow] stopped after {len(rec.ticks)} ticks")
    if out_dir and rec.ticks:
        rec.save()
    return rec
