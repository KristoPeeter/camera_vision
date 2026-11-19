"""Receive the RTP/H.264 stream and measure end-to-end latency from the burned-in timestamps.

Run:  python -m camvision.receive --display
      python -m camvision --source webcam --output rtp        (in another terminal)
"""
from __future__ import annotations

import argparse
import csv
import json
import signal

import cv2
import numpy as np

from . import timestamp
from .metrics import estimate_drops
from .pipeline import Capture, Output


def summarize(rows: list[dict], unreadable: int, warmup: int = 10) -> dict:
    # Skip warm-up: the first frames wait for the first keyframe and pipeline startup.
    total = len(rows)
    rows = rows[warmup:]
    if not rows:
        return {"frames": 0, "frames_total": total, "unreadable_stamps": unreadable}
    e2e = np.array([r["e2e_ms"] for r in rows])
    dropped, interval = estimate_drops([r["stamp_ms"] for r in rows])
    return {
        "frames": len(rows),
        "frames_total": total,
        "unreadable_stamps": unreadable,
        "source_interval_ms": round(interval, 1),
        "lost": dropped,
        "loss_%": round(100 * dropped / (dropped + len(rows)), 2),
        "e2e_ms_p50": round(float(np.percentile(e2e, 50)), 1),
        "e2e_ms_p95": round(float(np.percentile(e2e, 95)), 1),
        "e2e_ms_max": round(float(e2e.max()), 1),
    }


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="RTP/H.264 receiver with end-to-end latency metrics")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--jitter-ms", type=int, default=50, help="rtpjitterbuffer latency")
    p.add_argument("--frames", type=int, default=-1, help="stop after N frames (-1 = until Ctrl+C)")
    p.add_argument("--display", action="store_true", help="show the received video")
    p.add_argument("--idle-timeout", type=float, default=3.0,
                   help="seconds without frames before assuming the stream ended (after loss the "
                        "decoder can go silent until the next keyframe)")
    p.add_argument("--warmup", type=int, default=10, help="frames excluded from the summary")
    p.add_argument("--csv", default="receive.csv")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args(argv)

    cap = Capture("rtp", port=args.port, jitter_ms=args.jitter_ms,
                  width=args.width, height=args.height)
    out = Output("display", args.width, args.height, 30) if args.display else None
    rows: list[dict] = []
    unreadable = 0
    stop = False

    def _on_sigint(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_sigint)
    print(f"pipeline: {cap.description}\nwaiting for stream on UDP port {args.port}...")
    cap.start()
    started = False
    idle = 0.0
    try:
        while not stop and (args.frames < 0 or len(rows) < args.frames):
            frame = cap.read(timeout_s=0.5)
            if frame is None:
                idle += 0.5
                if started and idle >= args.idle_timeout:
                    break         # stream stopped after it had started
                continue          # sender not running yet, or a gap after packet loss
            started, idle = True, 0.0
            stamp = timestamp.read(frame.image)
            if stamp is None:
                unreadable += 1
                continue
            e2e = timestamp.age_ms(stamp)
            rows.append({"stamp_ms": stamp, "e2e_ms": e2e})
            if out:
                img = frame.image
                cv2.rectangle(img, (0, 26), (img.shape[1], 50), (0, 0, 0), -1)
                cv2.putText(img, f"RECEIVED | end-to-end {e2e:4d} ms", (6, 43),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                out.push(img)
    finally:
        cap.stop()
        if out:
            out.close()

    if rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    s = summarize(rows, unreadable, args.warmup)
    print(json.dumps(s, indent=2))
    return s


if __name__ == "__main__":
    main()
