"""Stable-Baselines3 policy -> the deterministic policy network as a plain torch module.

What gets extracted (observation -> raw outputs, all preprocessing such as uint8 /255 in-graph):

  PPO, A2C   features -> mlp_extractor.policy_net -> action_net: logits (Discrete) or the mean
             action (Box, what ``deterministic=True`` uses, before clipping)
  DQN        q_net: Q-values per action (argmax = the greedy action)
  SAC        actor: tanh(mu), the deterministic action in [-1, 1] before rescaling to the bounds
  TD3, DDPG  actor.mu: the deterministic action in [-1, 1]

Dict observation spaces (MultiInputPolicy) become one ONNX input per key, which the viewer shows
as separate lanes (an image plus plain numbers, say), joined by the extractor's Concat.
"""

from __future__ import annotations

import json
import os
import zipfile

import numpy as np

ALGOS = ("PPO", "A2C", "DQN", "SAC", "TD3", "DDPG")


def is_sb3(obj) -> bool:
    if isinstance(obj, (str, os.PathLike)):
        return str(obj).endswith(".zip")
    try:
        from stable_baselines3.common.base_class import BaseAlgorithm
    except ImportError:
        return False
    return isinstance(obj, BaseAlgorithm)


def _guess_algo(path: str) -> list[str]:
    """Read the policy class module out of the zip's 'data' JSON: sb3.dqn.policies -> DQN, ..."""
    try:
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("data"))
        mod = str(data.get("policy_class", {}).get("__module__", ""))
    except Exception:
        mod = ""
    for a in ("dqn", "sac", "td3", "ddpg"):
        if f".{a}." in mod:
            return [a.upper()]
    return ["PPO", "A2C", "DQN", "SAC", "TD3"]


def load_sb3(path: str, algo: str | None = None):
    import stable_baselines3 as sb3

    errors = []
    for a in [algo.upper()] if algo else _guess_algo(str(path)):
        try:
            return getattr(sb3, a).load(str(path), device="cpu")
        except Exception as e:  # wrong algorithm class for this zip: try the next one
            errors.append(f"{a}: {e}")
    raise ValueError(f"could not load {path} with any SB3 algorithm ({'; '.join(errors)}); pass algo=")


def from_sb3(model, algo: str | None = None):
    """SB3 model (or path of its ``.zip``) -> ``SB3Policy`` torch module, deterministic outputs.

    The returned module also carries ``input_names``, ``example_inputs()``, ``output_mode``
    ("categorical" for discrete actions, else "continuous") and ``predict(obs)`` (SB3's own
    deterministic action), which :func:`netglow.record` uses.
    """
    if _is_policy(model):
        return model
    if isinstance(model, (str, os.PathLike)):
        model = load_sb3(str(model), algo)
    return SB3Policy(model).eval().cpu()


def _make_module_class():
    import torch
    from torch import nn

    class _SB3Policy(nn.Module):
        def __init__(self, algo_model):
            super().__init__()
            from gymnasium import spaces

            self.sb3 = [algo_model]  # list: keep it out of the module tree (and out of the export)
            self.policy = algo_model.policy.to("cpu").eval()
            self.observation_space = algo_model.observation_space
            self.action_space = algo_model.action_space
            obs = self.observation_space
            self.is_dict = isinstance(obs, spaces.Dict)
            self.input_names = list(obs.spaces.keys()) if self.is_dict else ["obs"]
            self.output_mode = "categorical" if isinstance(self.action_space, spaces.Discrete) else "continuous"
            p = self.policy
            kind = type(p).__module__
            if hasattr(p, "q_net") and not hasattr(p, "actor"):
                self.kind = "dqn"
            elif ".sac." in kind:
                self.kind = "sac"
            elif hasattr(p, "actor") and hasattr(p.actor, "mu"):
                self.kind = "td3"
            elif hasattr(p, "mlp_extractor") and hasattr(p, "action_net"):
                self.kind = "ac"
            else:
                raise TypeError(f"unsupported SB3 policy {type(p).__name__}")

        def _flat_float(self, extractor) -> bool:
            """A flat float Box through a FlattenExtractor: SB3's preprocessing is a no-op cast; skip it
            so the ONNX graph starts directly at the first Gemm."""
            from gymnasium import spaces
            from stable_baselines3.common.torch_layers import FlattenExtractor

            o = self.observation_space
            return (isinstance(extractor, FlattenExtractor) and isinstance(o, spaces.Box)
                    and len(o.shape) == 1 and np.issubdtype(o.dtype, np.floating))

        def _features(self, owner, extractor, obs):
            if not self.is_dict and self._flat_float(extractor):
                return obs.float() if obs.dtype != torch.float32 else obs
            return owner.extract_features(obs, extractor)

        def forward(self, *xs):
            obs = dict(zip(self.input_names, xs)) if self.is_dict else xs[0]
            p = self.policy
            if self.kind == "ac":
                f = self._features(p, p.pi_features_extractor, obs)
                return p.action_net(p.mlp_extractor.forward_actor(f))
            if self.kind == "dqn":
                q = p.q_net
                return q.q_net(self._features(q, q.features_extractor, obs))
            a = p.actor
            f = self._features(a, a.features_extractor, obs)
            if self.kind == "sac":
                return torch.tanh(a.mu(a.latent_pi(f)))
            return a.mu(f)

        def example_inputs(self):
            """One zero observation per input, batch axis 1, in the space's dtype (float64 -> float32)."""
            def one(space):
                dt = np.float32 if np.issubdtype(space.dtype, np.floating) else space.dtype
                return torch.as_tensor(np.zeros((1, *space.shape), dtype=dt))

            o = self.observation_space
            return tuple(one(o.spaces[k]) for k in self.input_names) if self.is_dict else (one(o),)

        def predict(self, obs):
            """SB3's own deterministic action for one observation (what the env executes)."""
            return self.sb3[0].predict(obs, deterministic=True)[0]

    return _SB3Policy


_cls = None


def SB3Policy(algo_model):  # noqa: N802  (a factory that looks like a class, so torch imports lazily)
    global _cls
    if _cls is None:
        _cls = _make_module_class()
    return _cls(algo_model)


def _is_policy(obj) -> bool:
    return _cls is not None and isinstance(obj, _cls)
