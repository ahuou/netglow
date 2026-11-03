"""netglow command line: record, export, serve.

    netglow record --sb3 ppo.zip --env LunarLander-v3 [--steps 600] [--out DIR] [--live]
    netglow export --sb3 ppo.zip -o model.onnx
    netglow export --torch net.pt --input-shape 1,8 -o model.onnx     # torch.save(module) file
    netglow serve [--port 8765] [--dist ../dist]
"""

from __future__ import annotations

import argparse
import sys


def _record(a) -> None:
    from ._record import record

    rec = record(a.sb3 or a.torch_module(), a.env, steps=a.steps, out_dir=a.out, seed=a.seed, frames=not a.no_frames,
                 live=a.live, port=a.port, title=a.title, algo=a.algo, web=a.dist)
    if a.live:
        print("[netglow] done; the viewer keeps the stream. Ctrl+C to stop serving.")
        try:
            rec.server._thread.join()
        except KeyboardInterrupt:
            pass
        rec.close()


def _export(a) -> None:
    from ._export import export

    if a.sb3:
        from ._sb3 import from_sb3

        print(export(from_sb3(a.sb3, a.algo), path=a.out))
        return
    import torch

    shape = [int(s) for s in a.input_shape.split(",")]
    ex = torch.zeros(*shape, dtype=getattr(torch, a.input_dtype))
    print(export(a.torch_module(), ex, a.out, batched=True))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="netglow", description="Export policies and recordings for the netglow viewer.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_args(p):
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--sb3", help="Stable-Baselines3 checkpoint .zip (PPO, A2C, DQN, SAC, TD3)")
        g.add_argument("--torch", help="a whole module saved with torch.save(module, path)")
        p.add_argument("--algo", default=None, help="SB3 algorithm, if the zip doesn't say (PPO, A2C, DQN, SAC, TD3)")

    r = sub.add_parser("record", help="run a policy in a Gymnasium env and write model.onnx + recording.npz + labels.json")
    model_args(r)
    r.add_argument("--env", required=True, help="Gymnasium env id, e.g. LunarLander-v3")
    r.add_argument("--steps", type=int, default=1000)
    r.add_argument("--out", default="netglow_run")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--no-frames", action="store_true", help="don't store the env's render")
    r.add_argument("--title", default=None)
    r.add_argument("--live", action="store_true", help="also stream to the browser while recording (real-time pace)")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--dist", default=None, help="serve this local viewer build on the live port")
    r.set_defaults(fn=_record)

    e = sub.add_parser("export", help="export a policy to model.onnx")
    model_args(e)
    e.add_argument("-o", "--out", default="model.onnx")
    e.add_argument("--input-shape", default=None, help="torch: example input shape with batch, e.g. 1,8 or 1,4,84,84")
    e.add_argument("--input-dtype", default="float32", help="torch: example input dtype (float32, uint8, ...)")
    e.set_defaults(fn=_export)

    s = sub.add_parser("serve", help="serve the viewer at http://localhost:PORT/ (fallback when a browser blocks https -> localhost)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--host", default="localhost")
    s.add_argument("--dist", default=None, help="a local `npm run build` output; default: cached copy of the hosted page")
    s.set_defaults(fn=lambda a: __import__("netglow._live", fromlist=["serve"]).serve(a.port, a.host, a.dist))

    a = ap.parse_args(argv)

    def torch_module():
        import torch

        if a.torch is None:
            return None
        return torch.load(a.torch, weights_only=False, map_location="cpu")

    a.torch_module = torch_module
    if a.cmd == "export" and a.torch and not a.input_shape:
        ap.error("--torch needs --input-shape")
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
