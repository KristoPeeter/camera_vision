"""Detections sent as a separate metadata stream next to the video.

Sender: one small JSON datagram per frame over UDP, keyed by the frame's capture time,
the same value that is burned into the frame's pixels (camvision/timestamp.py).
Receiver: a background thread collects the messages. For each decoded frame, the
stamp read from its pixels is used to look up that frame's detections.

Why separate: the video stays clean, operators can toggle overlays, and detections
can be logged or fused with other sensors without decoding video. Every frame gets a
message, including frames with no detections, so "no objects" and "metadata
missing" are distinguishable.
"""
from __future__ import annotations

import json
import socket
import threading
from collections import OrderedDict

from .timestamp import MASK32, now_ms


class MetaSender:
    def __init__(self, host: str, port: int):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, ts_ms: int, boxes, **extra) -> None:
        msg = {"ts": ts_ms & MASK32, "boxes": [list(map(int, b)) for b in boxes], **extra}
        self.sock.sendto(json.dumps(msg, separators=(",", ":")).encode(), self.addr)

    def close(self) -> None:
        self.sock.close()


class MetaReceiver:
    """Collects metadata in the background; look it up by frame timestamp."""

    def __init__(self, port: int, keep: int = 600):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(0.2)
        self.keep = keep
        self.store: OrderedDict[int, tuple[int, dict]] = OrderedDict()   # ts -> (arrival ms, msg)
        self.lock = threading.Lock()
        self.received = 0
        self.latest_ts: int | None = None      # newest message, for the "naive" comparison
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self.sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            with self.lock:
                self.store[msg["ts"]] = (now_ms(), msg)
                self.latest_ts = msg["ts"]
                self.received += 1
                while len(self.store) > self.keep:     # bounded: drop the oldest
                    self.store.popitem(last=False)

    def lookup(self, ts_ms: int) -> tuple[dict, int] | None:
        """Return (message, arrival time in ms) for this frame, or None if it hasn't arrived."""
        with self.lock:
            hit = self.store.get(ts_ms & MASK32)
        return None if hit is None else (hit[1], hit[0])

    def close(self) -> None:
        self._stop.set()
        self.sock.close()
        self.thread.join(timeout=1)
