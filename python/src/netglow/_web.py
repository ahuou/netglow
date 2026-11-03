"""Serving the viewer itself from localhost (``netglow serve``, and the HTTP side of live mode).

Why: a page on https://ahuou.github.io talking to ws://localhost is a "local network access"
from a public site. Chrome asks the user once, Firefox allows it, Safari may refuse. A viewer
served from http://localhost:PORT talks to the same machine, so no browser asks anything.

Where the files come from, first match wins:

1. ``web=`` / ``--dist`` / ``$NETGLOW_WEB``: a local ``npm run build`` output (``dist/``).
2. a ``web/`` folder bundled inside the installed package (present if you copied a build there).
3. the hosted page, fetched once and cached in ``~/.cache/netglow/web/`` (needs internet the first time).

Everything is served under ``/netglow/`` (the hosted build's base path); ``/`` redirects there.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import posixpath
import urllib.request
from pathlib import Path

HOSTED = "https://ahuou.github.io/netglow/"
TYPES = {".wasm": "application/wasm", ".mjs": "text/javascript", ".js": "text/javascript", ".onnx": "application/octet-stream",
         ".npz": "application/octet-stream", ".svg": "image/svg+xml", ".json": "application/json", ".css": "text/css",
         ".html": "text/html; charset=utf-8", ".woff2": "font/woff2"}


def cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "netglow" / "web"


def local_dist(web: str | None) -> Path | None:
    for cand in (web, os.environ.get("NETGLOW_WEB"), Path(__file__).parent / "web"):
        if cand and (Path(cand) / "index.html").is_file():
            return Path(cand)
    return None


def _fetch(url: str, timeout: float = 15) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "netglow-python"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def read_file(rel: str, web: str | None) -> bytes | None:
    """The viewer file at ``rel`` (relative to the app root), or None if there is none."""
    dist = local_dist(web)
    if dist is not None:
        p = dist / rel
        return p.read_bytes() if p.is_file() else None
    c = cache_dir() / rel
    if rel != "index.html" and c.is_file():
        return c.read_bytes()
    try:
        data = _fetch(HOSTED + rel)
    except Exception:
        return c.read_bytes() if c.is_file() else None
    c.parent.mkdir(parents=True, exist_ok=True)
    c.write_bytes(data)
    return data


def _response(status: int, reason: str, body: bytes, ctype: str = "text/plain; charset=utf-8", extra=()):
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    h = Headers([("Content-Type", ctype), ("Content-Length", str(len(body))), ("Cache-Control", "no-cache"), *extra])
    return Response(status, reason, h, body)


async def respond(path: str, port: int, web: str | None = None, live: bool = True):
    """HTTP response for a non-WebSocket request on the live / serve port."""
    path = path.split("?", 1)[0].split("#", 1)[0]
    if path in ("", "/", "/netglow"):
        loc = "/netglow/" + ("?live=here" if live else "")
        return _response(302, "Found", b"", extra=[("Location", loc)])
    if not path.startswith("/netglow/"):
        return _response(404, "Not Found", b"not found")
    rel = posixpath.normpath(path[len("/netglow/"):] or "index.html")
    if rel.startswith("..") or rel.startswith("/"):
        return _response(403, "Forbidden", b"forbidden")
    if rel == ".":
        rel = "index.html"
    data = await asyncio.to_thread(read_file, rel, web)
    if data is None:
        return _response(404, "Not Found", f"{rel} not found".encode())
    ext = posixpath.splitext(rel)[1].lower()
    return _response(200, "OK", data, TYPES.get(ext) or mimetypes.guess_type(rel)[0] or "application/octet-stream")
