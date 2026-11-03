"""torch module -> one self-contained ONNX file in the shape the netglow viewer reads.

Vocabulary
----------
opset    ONNX operator-set version; 17 is supported by onnxruntime-web.
Gemm     ONNX "general matrix multiply"; what nn.Linear becomes with the legacy (TorchScript)
         exporter. The viewer reads weights from Gemm / MatMul / Conv nodes.
legacy   torch.onnx.export(..., dynamo=False): traces the module once and keeps one ONNX node
exporter per torch op, so the graph stays the plain chain the viewer can attribute through.
"""

from __future__ import annotations

import io
import os
from typing import Any, Sequence

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("netglow.export needs PyTorch: pip install torch (CPU wheel is enough)") from e
    return torch


def as_tensor(x: Any):
    """ndarray / list / tensor -> CPU tensor. float64 becomes float32 (policies run in float32)."""
    torch = _torch()
    t = x.detach().cpu() if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
    return t.float() if t.dtype == torch.float64 else t


def normalize_inputs(example_inputs: Any, input_names: Sequence[str] | None = None) -> tuple[list[str], tuple]:
    """Example inputs as (names, tensors). Accepts one tensor, a tuple / list (forward's positional
    order) or a dict {name: tensor} (insertion order = forward's order)."""
    if isinstance(example_inputs, dict):
        names = list(input_names or example_inputs.keys())
        xs = tuple(as_tensor(v) for v in example_inputs.values())
    elif isinstance(example_inputs, (tuple, list)):
        xs = tuple(as_tensor(v) for v in example_inputs)
        names = list(input_names) if input_names else (["obs"] if len(xs) == 1 else [f"in{i}" for i in range(len(xs))])
    else:
        xs = (as_tensor(example_inputs),)
        names = list(input_names) if input_names else ["obs"]
    if len(names) != len(xs):
        raise ValueError(f"{len(names)} input names for {len(xs)} example inputs")
    return names, xs


def _batched(xs: tuple, batched: bool | None = None) -> tuple:
    """Give every example a leading batch axis. batched=None guesses: a leading 1 on a tensor of 2+ dims
    is taken as the batch axis (pass batched=False for e.g. a single-channel [1, H, W] observation)."""
    if batched is None:
        return tuple(x if x.dim() > 1 and x.shape[0] == 1 else x.unsqueeze(0) for x in xs)
    return xs if batched else tuple(x.unsqueeze(0) for x in xs)


def export_bytes(
    model,
    example_inputs: Any = None,
    input_names: Sequence[str] | None = None,
    output_name: str = "logits",
    opset: int = 17,
    batched: bool | None = None,
) -> bytes:
    """Like :func:`export` but returns the ONNX file as bytes."""
    torch = _torch()
    from ._sb3 import is_sb3, from_sb3

    if is_sb3(model):
        model = from_sb3(model)
    if example_inputs is None:
        ex = getattr(model, "example_inputs", None)
        if ex is None:
            raise ValueError("example_inputs is required for a plain torch module (one observation, or a batch of 1)")
        example_inputs, batched = ex(), True
    if input_names is None:
        input_names = getattr(model, "input_names", None)
    names, xs = normalize_inputs(example_inputs, input_names)
    xs = _batched(xs, batched)
    module = model.eval().cpu()
    buf = io.BytesIO()
    kw = dict(
        input_names=names,
        output_names=[output_name],
        dynamic_axes={**{n: {0: "N"} for n in names}, output_name: {0: "N"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    with torch.no_grad():
        try:
            torch.onnx.export(module, xs, buf, dynamo=False, **kw)
        except TypeError:  # torch < 2.5 has no dynamo kwarg; the legacy exporter is the default there
            torch.onnx.export(module, xs, buf, **kw)
    import onnx

    m = onnx.load_from_string(buf.getvalue())
    onnx.checker.check_model(m)
    return m.SerializeToString()  # weights inline: the legacy exporter never splits small models


def export(
    model,
    example_inputs: Any = None,
    path: str = "model.onnx",
    input_names: Sequence[str] | None = None,
    output_name: str = "logits",
    opset: int = 17,
    verify: bool = True,
    batched: bool | None = None,
) -> str:
    """Export a policy network to an ONNX file the netglow viewer loads.

    model           a torch.nn.Module (observation(s) -> raw outputs, e.g. logits), a
                    Stable-Baselines3 model, or the path of an SB3 ``.zip``.
    example_inputs  one observation (with or without the batch axis of 1), a tuple of them for a
                    multi-input ``forward(frames, vector)``, or a dict {input name: observation}.
                    Optional for SB3 models (taken from the observation space).
    input_names     ONNX input names; the recording stores them as ``obs_<name>``.
    batched         whether example_inputs already have the batch axis (default: guess; a leading
                    1 on a 2+-dim tensor counts as the batch axis).
    verify          if onnxruntime is installed, check ONNX and torch agree on the example.
    """
    from ._sb3 import is_sb3, from_sb3

    if is_sb3(model):
        model = from_sb3(model)
    data = export_bytes(model, example_inputs, input_names, output_name, opset, batched)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    if verify:
        if example_inputs is None:
            example_inputs, batched = model.example_inputs(), True
        names, xs = normalize_inputs(example_inputs, input_names or getattr(model, "input_names", None))
        try:
            verify_onnx(path, model, {n: x.numpy() for n, x in zip(names, _batched(xs, batched))}, quiet=True)
        except ImportError:
            pass
    return path


def verify_onnx(path: str, module, obs, input_name: str = "obs", quiet: bool = False, tol: float = 1e-3) -> dict:
    """Run ``obs`` through onnxruntime (CPU) and torch and compare the outputs.

    ``obs`` is one batched array, or {input name: batched array} in ``forward``'s order.
    Raises if they disagree by more than ``tol``.
    """
    import onnxruntime as ort

    torch = _torch()
    feeds = {k: np.asarray(v) for k, v in (obs.items() if isinstance(obs, dict) else [(input_name, obs)])}
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, feeds)[0]
    with torch.no_grad():
        torch_out = module.eval().cpu()(*[as_tensor(v) for v in feeds.values()])
        if isinstance(torch_out, (tuple, list)):
            torch_out = torch_out[0]
        torch_out = torch_out.numpy()
    diff = float(np.abs(onnx_out - torch_out).max()) if onnx_out.size else 0.0
    agree = float((onnx_out.reshape(len(onnx_out), -1).argmax(-1) == torch_out.reshape(len(torch_out), -1).argmax(-1)).mean())
    T = len(next(iter(feeds.values())))
    if not quiet:
        print(f"[netglow] {os.path.basename(path)}: max|onnx-torch| = {diff:.3g}, argmax agreement {agree * 100:.2f}% over {T} ticks")
    if diff > tol * max(1.0, float(np.abs(torch_out).max())):
        raise AssertionError(f"ONNX and torch disagree on {os.path.basename(path)}: max diff {diff:.3g}")
    return {"max_abs_diff": diff, "argmax_agreement": agree}
