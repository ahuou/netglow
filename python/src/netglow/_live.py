"""Live mode: stream each tick's inputs (and the model, once) to the netglow viewer over a localhost WebSocket.

The browser runs the ONNX model itself on the streamed inputs, so only inputs travel; the raw outputs
you computed are sent too, for the live match check.

Wire format (every message is binary)::

    u32 little-endian  header length
    header             UTF-8 JSON
    payload            raw bytes of the arrays listed in the header, back to back

    {"type": "model", "fps": 50, "labels": {...}, "inputs": ["obs"], "size": n}   payload: model.onnx
    {"type": "tick", "i": 17, "arrays": [{"key": "obs_obs", "dtype": "<f4", "shape": [8], "size": 32}, ...]}

Tick array keys are the recording keys: ``obs_<input name>``, ``outputs``, ``actions``, ``frames``.

The same port also answers plain HTTP with the viewer itself (from a local build, or a cached copy of
the hosted page), so ``http://localhost:PORT/`` works in any browser without the https -> localhost
permission question. See :mod:`netglow._web`.
"""

from __future__ import annotations

import asyncio
import collections
import json
import struct
import threading
from typing import Any

import numpy as np

HOSTED = "https://ahuou.github.io/netglow/"


def encode(header: dict, blobs: list[bytes]) -> bytes:
    h = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<I", len(h)) + h + b"".join(blobs)


def encode_tick(i: int, tick: dict) -> bytes:
    arrays, blobs = [], []

    def add(key: str, a: Any):
        a = np.asarray(a, order="C")  # not ascontiguousarray: that turns a scalar action into shape (1,)
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        if a.dtype == np.bool_:
            a = a.astype(np.uint8)
        b = a.astype(a.dtype.newbyteorder("<")).tobytes()
        arrays.append({"key": key, "dtype": a.dtype.newbyteorder("<").str, "shape": list(a.shape), "size": len(b)})
        blobs.append(b)

    for name, a in tick["obs"].items():
        add(f"obs_{name}", a)
    if tick.get("output") is not None:
        add("outputs", tick["output"])
    if tick.get("action") is not None:
        a = np.asarray(tick["action"])
        add("actions", a.astype(np.int64) if np.issubdtype(a.dtype, np.integer) else a.astype(np.float32))
    if tick.get("frame") is not None:
        add("frames", np.asarray(tick["frame"], dtype=np.uint8))
    return encode({"type": "tick", "i": i, "arrays": arrays}, blobs)


class LiveServer:
    """A WebSocket + HTTP server on localhost, running in a background thread.

    ``set_model(onnx_bytes, labels, fps)`` once, then ``push(tick)`` per tick; browsers that connect
    later get the model and the last ``backlog`` ticks first, so the view fills at once.
    """

    def __init__(self, host: str = "localhost", port: int = 8765, backlog: int = 600, web: str | None = None,
                 quiet: bool = False, serve_only: bool = False):
        self.host, self.port, self.quiet, self.serve_only = host, port, quiet, serve_only
        self.model_msg: bytes | None = None
        self.backlog: collections.deque[bytes] = collections.deque(maxlen=backlog)
        self.n = 0
        self.clients: set = set()
        self._web = web
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Future | None = None
        self._thread = threading.Thread(target=self._run, name="netglow-live", daemon=True)
        self._thread.start()
        self._ready.wait(10)
        if self._error:
            raise self._error
        if serve_only:
            if not quiet:
                print(f"[netglow] viewer at http://{host}:{self.port}/")
        elif not quiet:
            print(f"[netglow] live on ws://{host}:{self.port}\n"
                  f"          open {HOSTED}?live={self.port}  (Chrome asks once to allow local access)\n"
                  f"          or   http://{host}:{self.port}/  (the viewer served from here, any browser)")

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    @property
    def has_model(self) -> bool:
        return self.model_msg is not None

    # ---- called from the user's thread
    def set_model(self, onnx: bytes, labels: dict | None = None, fps: float | None = None) -> None:
        names = []
        try:
            import onnx as _onnx

            m = _onnx.load_from_string(onnx)
            init = {t.name for t in m.graph.initializer}
            names = [i.name for i in m.graph.input if i.name not in init]
        except Exception:
            pass
        msg = encode({"type": "model", "fps": fps, "labels": labels or {}, "inputs": names, "size": len(onnx)}, [onnx])
        self._call(self._set_model, msg)

    def push(self, tick: dict) -> None:
        msg = encode_tick(self.n, tick)
        self.n += 1
        self._call(self._broadcast, msg)

    def close(self) -> None:
        if self._loop and self._stop and not self._stop.done():
            self._loop.call_soon_threadsafe(self._stop.set_result, None)
            self._thread.join(5)

    def _call(self, fn, *a) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(fn, *a)

    # ---- event loop side
    def _set_model(self, msg: bytes) -> None:
        self.model_msg = msg
        self.backlog.clear()
        for q in self.clients:
            q.put_nowait(msg)

    def _broadcast(self, msg: bytes) -> None:
        self.backlog.append(msg)
        for q in self.clients:
            if q.qsize() < 4000:  # a browser that can't keep up skips ticks instead of growing memory
                q.put_nowait(msg)

    async def _handler(self, ws) -> None:
        q: asyncio.Queue = asyncio.Queue()
        if self.model_msg:
            q.put_nowait(self.model_msg)
            for m in self.backlog:
                q.put_nowait(m)
        self.clients.add(q)
        if not self.quiet:
            print(f"[netglow] viewer connected ({len(self.clients)} open)")

        async def drain():
            async for _ in ws:  # the viewer sends nothing we need; read so close frames are handled
                pass

        reader = asyncio.ensure_future(drain())
        try:
            while not reader.done():
                getter = asyncio.ensure_future(q.get())
                done, _ = await asyncio.wait({getter, reader}, return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    await ws.send(getter.result())
                else:
                    getter.cancel()
        except Exception:
            pass
        finally:
            reader.cancel()
            self.clients.discard(q)

    async def _process_request(self, connection, request):
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        from ._web import respond

        return await respond(request.path, self.port, self._web, live=not self.serve_only)

    def _run(self) -> None:
        from websockets.asyncio.server import serve

        async def main():
            self._loop = asyncio.get_running_loop()
            self._stop = self._loop.create_future()
            try:
                server = await serve(self._handler, self.host, self.port, process_request=self._process_request,
                                     max_size=None, compression=None, ping_interval=None)
            except BaseException as e:
                self._error = e
                self._ready.set()
                return
            if self.port == 0:
                self.port = next(iter(server.sockets)).getsockname()[1]
            self._ready.set()
            async with server:
                await self._stop

        asyncio.run(main())


def live(model=None, port: int = 8765, host: str = "localhost", out_dir: str | None = None, **kw):
    """Start streaming to the viewer; returns a :class:`netglow.Recorder` with ``live=True``.

    ::

        rec = netglow.live(policy, port=8765)      # prints the link to open
        for ...:
            rec.add(obs)                            # the browser animates this tick
        rec.save("run")                             # optional: also write the files

    model may be a torch module or an SB3 model / ``.zip``; it is exported on the first tick.
    """
    from ._record import Recorder

    return Recorder(model, out_dir=out_dir, live=True, port=port, host=host, **kw)


def serve(port: int = 8765, host: str = "localhost", web: str | None = None, block: bool = True) -> LiveServer:
    """Serve the viewer at http://HOST:PORT/ (see :mod:`netglow._web` for where the files come from)."""
    srv = LiveServer(host=host, port=port, web=web, serve_only=True)
    if block:
        try:
            srv._thread.join()
        except KeyboardInterrupt:
            srv.close()
    return srv
