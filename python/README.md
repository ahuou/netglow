# netglow (Python)

Export a PyTorch or Stable-Baselines3 policy and what it sees for [netglow](https://ahuou.github.io/netglow/), the
in-browser "glowing network" viewer, or stream it into the page live.

```bash
pip install "git+https://github.com/ahuou/netglow#subdirectory=python"   # install torch first (CPU wheel is enough)
```

```python
import netglow
netglow.export(net, torch.zeros(8), "run/model.onnx")                  # torch module -> ONNX the viewer reads
netglow.record("ppo.zip", "LunarLander-v3", steps=600, out_dir="run")   # SB3 + Gymnasium -> run/{model.onnx, recording.npz, labels.json}
rec = netglow.Recorder(net, out_dir="run"); rec.add(obs); rec.save()    # any loop
rec = netglow.live(net); rec.add(obs)                                   # stream to the browser, tick by tick
```

CLI: `netglow record --sb3 ppo.zip --env LunarLander-v3 [--live]`, `netglow export --sb3 ppo.zip`, `netglow serve`.
See the main README's "From Python" and "Live mode" sections. Tests: `pip install -e ".[test]" && pytest`
(needs `npm install` at the repo root; every folder the tests write is loaded in the real web app).

This folder also builds the bundled demos (`train_demos.py`) with the older standalone `export_recording.py`.

## Setup

```bash
cd python
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # CPU wheel first
uv pip install -r requirements.txt
```

`swig` comes from PyPI as a wheel, so `box2d` builds without a system swig.

## Reproduce the demos

```bash
python train_demos.py all        # ~10 min on a laptop CPU
```

or stage by stage:

| stage        | what it does                                                       | output                       |
|--------------|--------------------------------------------------------------------|------------------------------|
| `ppo`        | PPO MlpPolicy on LunarLander-v3, rl-zoo hparams, 1M steps          | `runs/ppo_lunar.zip`         |
| `mlp`        | pick a landing episode, export ONNX + recording + labels           | `public/demos/lunar-mlp/`    |
| `cnn`        | pixel CNN (2x64x64 frames, 12.5 Hz) taught by the PPO expert: BC + 5 DAgger rounds | `runs/cnn_pilot.pt`          |
| `cnn-export` | episode flown by the CNN itself, export ONNX + recording + labels  | `public/demos/lunar-cnn/`    |

## Export your own policy

Stable-Baselines3 checkpoint (PPO/A2C):

```bash
python export_recording.py --sb3 model.zip --env LunarLander-v3 --out mydemo --episodes 1 --frames
```

Any torch module (obs -> raw outputs):

```python
from export_recording import export_onnx, record_episode, save_recording, verify_onnx

export_onnx(net, torch.zeros(1, 8), "mydemo/model.onnx")            # opset 17, Gemm, dynamic batch
rec = record_episode(env, lambda o: policy_and_logits(o), render=True)  # -> (action, logits)
verify_onnx("mydemo/model.onnx", net, rec["obs"])                    # onnxruntime vs torch
save_recording("mydemo/recording.npz", **rec, fps=50)
```

Then write `labels.json` (input/output names, icons) — see `write_labels()`.

## Recording format

`recording.npz` (np.savez_compressed, no object arrays):

| key            | dtype / shape                         | required |
|----------------|---------------------------------------|----------|
| `obs`          | ONNX input dtype, `(T, *input_shape)` | yes      |
| `actions`      | int64 `(T,)` or float32 `(T, A)`      | no       |
| `outputs`      | float32 `(T, n_out)` torch outputs    | no       |
| `t`            | float64 `(T,)` seconds                | no       |
| `fps`          | float64 scalar, Hz                    | no       |
| `frames`       | uint8 `(T, H, W, 3)`                  | no       |
| `input_names`, `output_names` | unicode arrays         | no       |
