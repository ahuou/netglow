"""netglow: export a policy network and what it sees for the netglow viewer, or stream it live.

    import netglow
    netglow.export(net, torch.zeros(1, 8), "run/model.onnx")       # any torch module
    netglow.record("ppo.zip", "LunarLander-v3", steps=600)           # SB3 + Gymnasium -> netglow_run/
    rec = netglow.Recorder(net, out_dir="run"); rec.add(obs); rec.save()
    rec = netglow.live(net)                                          # stream ticks to the browser

Open https://ahuou.github.io/netglow/ and drop model.onnx, recording.npz and labels.json on it.
"""

from ._export import export, export_bytes, verify_onnx
from ._live import LiveServer, live, serve
from ._record import Recorder, record, save_recording, write_labels
from ._sb3 import from_sb3

__all__ = ["export", "export_bytes", "verify_onnx", "from_sb3", "record", "Recorder", "save_recording", "write_labels",
           "live", "serve", "LiveServer"]
__version__ = "0.1.0"
