"""Random-weight fixtures for the browser tests: an MLP (float input) and a small CNN (uint8 image input)."""
import sys, numpy as np, torch, torch.nn as nn
out = sys.argv[1]
torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(8, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4))
class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(2, 16, 8, 4), nn.ReLU(), nn.Conv2d(16, 32, 4, 2), nn.ReLU(), nn.Flatten())
        s.h = nn.Sequential(nn.Linear(32 * 36, 128), nn.ReLU(), nn.Linear(128, 4))
    def forward(s, x): return s.h(s.c(x.float() / 255.0))
cnn = CNN()
T = 120
obs = np.cumsum(np.random.randn(T, 8).astype(np.float32) * 0.1, 0)
with torch.no_grad(): lg = mlp(torch.from_numpy(obs)).numpy()
torch.onnx.export(mlp, torch.zeros(1, 8), f"{out}/mlp.onnx", input_names=["obs"], output_names=["logits"], dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
np.savez_compressed(f"{out}/mlp.npz", obs=obs, actions=lg.argmax(1).astype(np.int64), outputs=lg, fps=np.float64(20))
img = (np.random.rand(T, 2, 64, 64) * 255).astype(np.uint8)
with torch.no_grad(): lg2 = cnn(torch.from_numpy(img)).numpy()
torch.onnx.export(cnn, torch.zeros(1, 2, 64, 64, dtype=torch.uint8), f"{out}/cnn.onnx", input_names=["obs"], output_names=["logits"], dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
np.savez_compressed(f"{out}/cnn.npz", obs=img, actions=lg2.argmax(1).astype(np.int64), outputs=lg2, fps=np.float64(20))
print("ok", lg[:2], lg2[:2])

# unsupported on purpose: a residual (skip) connection, to check the error message
class Res(nn.Module):
    def __init__(s):
        super().__init__(); s.a = nn.Linear(8, 8); s.b = nn.Linear(8, 4)
    def forward(s, x): return s.b(torch.relu(s.a(x)) + x)
torch.onnx.export(Res(), torch.zeros(1, 8), f"{out}/residual.onnx", input_names=["obs"], output_names=["logits"], dynamic_axes={"obs": {0: "batch"}}, opset_version=17, dynamo=False)
np.savez_compressed(f"{out}/residual.npz", obs=obs)

# RGB float input, mean/std normalised (values outside -1..1): exercises RGB layout + "stretch" display
class RGBNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(3, 8, 5, 2, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, 1, padding=1), nn.ReLU(), nn.Flatten())
        s.h = nn.Sequential(nn.Linear(16 * 8 * 8, 32), nn.ReLU(), nn.Linear(32, 3))
    def forward(s, x): return s.h(s.c(x))
rgb = RGBNet()
yy, xx = np.mgrid[0:32, 0:32]
imgs = np.stack([np.stack([np.sin(xx / 4 + t / 5), np.cos(yy / 5 - t / 7), np.sin((xx + yy) / 6 + t / 3)]) for t in range(60)]).astype(np.float32)
imgs = (imgs * 0.5 + 0.5 - 0.45) / 0.22
with torch.no_grad(): lg3 = rgb(torch.from_numpy(imgs)).numpy()
torch.onnx.export(rgb, torch.zeros(1, 3, 32, 32), f"{out}/rgbf.onnx", input_names=["obs"], output_names=["logits"], dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
np.savez_compressed(f"{out}/rgbf.npz", obs=imgs, actions=lg3.argmax(1).astype(np.int64), outputs=lg3, fps=np.float64(10))

# two inputs joined by a Concat: uint8 frames through a conv stack + a float vector straight in, then a dense head
class Mixed(nn.Module):
    def __init__(s):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(2, 16, 8, 4), nn.ReLU(), nn.Conv2d(16, 32, 4, 2), nn.ReLU(), nn.Flatten())
        s.h = nn.Sequential(nn.Linear(32 * 36 + 6, 128), nn.ReLU(), nn.Linear(128, 4))
    def forward(s, frames, vector): return s.h(torch.cat([s.c(frames.float() / 255.0), vector], 1))
mixed = Mixed()
fr = (np.random.rand(40, 2, 64, 64) * 255).astype(np.uint8)
vec = np.cumsum(np.random.randn(40, 6).astype(np.float32) * 0.1, 0)
with torch.no_grad(): lg4 = mixed(torch.from_numpy(fr), torch.from_numpy(vec)).numpy()
torch.onnx.export(mixed, (torch.zeros(1, 2, 64, 64, dtype=torch.uint8), torch.zeros(1, 6)), f"{out}/mixed.onnx", input_names=["frames", "vector"], output_names=["logits"], dynamic_axes={"frames": {0: "batch"}, "vector": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
np.savez_compressed(f"{out}/mixed.npz", obs_frames=fr, obs_vector=vec, actions=lg4.argmax(1).astype(np.int64), outputs=lg4, fps=np.float64(20))
