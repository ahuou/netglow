# netglow / python

Builds the bundled demos and exports your own policies for the netglow viewer.

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
