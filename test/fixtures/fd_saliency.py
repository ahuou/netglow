"""Finite-difference check of netglow's saliency: d logit[target] / d x by central differences in onnxruntime,
times x, summed over channels; compared with the heat netglow computed in the browser.

    python/.venv/bin/python test/fixtures/fd_saliency.py model.onnx rec.npz dump.json
dump.json: {"tick": t, "target": j, "heat": [H*W values]}
"""
import json, sys
import numpy as np
import onnxruntime as ort

model, rec, dump = sys.argv[1:4]
d = json.load(open(dump))
x = np.load(rec)["obs"][d["tick"]].astype(np.float32)
sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name
n = x.size
eps = 1e-3
X = np.repeat(x[None], 2 * n, 0).reshape(2 * n, -1)
idx = np.arange(n)
X[idx, idx] += eps
X[n + idx, idx] -= eps
out = sess.run(None, {name: X.reshape(2 * n, *x.shape)})[0][:, d["target"]]
g = ((out[:n] - out[n:]) / (2 * eps)).reshape(x.shape)
heat = (g * x).sum(0).ravel()
mine = np.asarray(d["heat"], np.float32)
err = np.abs(heat - mine).max() / (np.abs(heat).max() + 1e-12)
print(json.dumps({"max_rel_err": float(err), "corr": float(np.corrcoef(heat, mine)[0, 1])}))
