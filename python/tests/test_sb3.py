"""Stable-Baselines3 policies (tiny, trained for seconds) -> netglow.record -> the web app agrees with SB3."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3")
gym = pytest.importorskip("gymnasium")

import netglow  # noqa: E402


@pytest.fixture(scope="module")
def ppo_zip(tmp_path_factory):
    m = sb3.PPO("MlpPolicy", "CartPole-v1", n_steps=256, batch_size=64, n_epochs=2, seed=0, device="cpu",
                policy_kwargs={"net_arch": [32, 32]})
    m.learn(512)
    p = tmp_path_factory.mktemp("sb3") / "ppo_cartpole.zip"
    m.save(p)
    return p


def test_ppo_record(ppo_zip, tmp_path, load_in_viewer):
    rec = netglow.record(str(ppo_zip), "CartPole-v1", steps=120, out_dir=tmp_path, quiet=True)
    z = np.load(tmp_path / "recording.npz")
    assert z["obs"].shape == (120, 4) and z["frames"].shape[0] == 120 and z["frames"].shape[2] == 240
    lab = json.loads((tmp_path / "labels.json").read_text())
    assert lab["outputs"] == ["push left", "push right"] and lab["output_mode"] == "categorical"
    # SB3's own deterministic action equals argmax of the extracted logits
    assert (z["actions"] == z["outputs"].argmax(1)).all()
    r = load_in_viewer(tmp_path)
    assert r["match"] == "Match 100%"
    assert r["labels"]["inputs"] == ["cart position", "cart speed", "pole angle", "pole spin"]
    assert rec.fps == 50


def test_dqn(tmp_path, load_in_viewer):
    m = sb3.DQN("MlpPolicy", "CartPole-v1", learning_starts=50, seed=0, device="cpu", policy_kwargs={"net_arch": [16]})
    m.learn(100)
    netglow.record(m, gym.make("CartPole-v1"), steps=60, out_dir=tmp_path, quiet=True)
    r = load_in_viewer(tmp_path)
    assert r["match"] == "Match 100%" and r["layers"] == 2


def test_sac_continuous(tmp_path, load_in_viewer):
    m = sb3.SAC("MlpPolicy", "Pendulum-v1", learning_starts=50, seed=0, device="cpu", policy_kwargs={"net_arch": [16, 16]})
    m.learn(80)
    netglow.record(m, "Pendulum-v1", steps=50, out_dir=tmp_path, frames=False, quiet=True)
    z = np.load(tmp_path / "recording.npz")
    # deterministic SAC action = tanh(mu) rescaled to the bounds; Pendulum's bounds are +-2
    assert np.allclose(z["actions"].reshape(-1), 2 * z["outputs"].reshape(-1), atol=1e-5)
    r = load_in_viewer(tmp_path)
    assert r["match"] in ("Outputs match", "Match 100%")
    assert json.loads((tmp_path / "labels.json").read_text())["output_mode"] == "continuous"


def test_dict_obs_two_lanes(tmp_path, load_in_viewer):
    """MultiInputPolicy on a Dict(image, vector) space -> two ONNX inputs, two lanes in the viewer."""
    from gymnasium import spaces

    class DictEnv(gym.Env):
        observation_space = spaces.Dict({"img": spaces.Box(0, 255, (1, 36, 36), np.uint8),
                                         "vec": spaces.Box(-1, 1, (3,), np.float32)})
        action_space = spaces.Discrete(3)
        metadata = {"render_fps": 10}

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            return self.observation_space.sample(), {}

        def step(self, a):
            self.t += 1
            return self.observation_space.sample(), 0.0, False, self.t >= 20, {}

    env = DictEnv()
    m = sb3.PPO("MultiInputPolicy", env, n_steps=32, batch_size=32, n_epochs=1, seed=0, device="cpu",
                policy_kwargs={"net_arch": [16], "normalize_images": True})
    netglow.record(m, env, steps=30, out_dir=tmp_path, quiet=True)
    z = np.load(tmp_path / "recording.npz")
    assert set(z.files) >= {"obs_img", "obs_vec"}
    r = load_in_viewer(tmp_path)
    assert r["inputs"] == ["img:image", "vec:vector"]
    assert r["match"] == "Match 100%"


def test_cli_record_and_export(ppo_zip, tmp_path, load_in_viewer):
    out = tmp_path / "cli"
    subprocess.run([sys.executable, "-m", "netglow.cli", "record", "--sb3", str(ppo_zip), "--env", "CartPole-v1",
                    "--steps", "40", "--out", str(out), "--no-frames"], check=True, capture_output=True)
    assert load_in_viewer(out)["match"] == "Match 100%"
    subprocess.run([sys.executable, "-m", "netglow.cli", "export", "--sb3", str(ppo_zip), "-o", str(tmp_path / "x.onnx")],
                   check=True, capture_output=True)
    assert (tmp_path / "x.onnx").stat().st_size > 1000
