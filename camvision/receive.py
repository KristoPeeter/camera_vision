"""Receive the RTP/H.264 stream and measure end-to-end latency from the burned-in timestamps.

Run:  python -m camvision.receive --display
      python -m camvision --source webcam --output rtp        (in another terminal)
"""
from __future__ import annotations

import argparse
import csv
import json
import signal

import numpy as np

from . import timestamp
from .detector import annotate
from .meta import MetaReceiver
from .metrics import estimate_drops
from .pipeline import Capture, Output


def summarize(rows: list[dict], unreadable: int, warmup: int = 10,
              meta: MetaReceiver | None = None) -> dict:
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
        **(_meta_summary(rows, meta) if meta else {}),
    }


def _meta_summary(rows: list[dict], meta: MetaReceiver) -> dict:
    matched = [r for r in rows if r["meta_lead_ms"] is not None]
    # Unmatched frames: did the metadata arrive later (late) or never (missing)?
    late = sum(1 for r in rows if r["meta_lead_ms"] is None and meta.lookup(r["stamp_ms"]))
    lead = np.array([r["meta_lead_ms"] for r in matched]) if matched else np.array([0])
    return {
        "meta_matched_%": round(100 * len(matched) / len(rows), 1),
        "meta_late": late,
        "meta_missing": len(rows) - len(matched) - late,
        # How long the detections waited for their video frame
        "meta_lead_ms_p50": round(float(np.percentile(lead, 50)), 1),
        "meta_lead_ms_min": round(float(lead.min()), 1),
        # % of frames where "use the latest detections" would draw another frame's boxes
        "naive_latest_wrong_%": round(100 * np.mean([r["naive_wrong"] for r in rows]), 1),
    }


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="RTP/H.264 receiver with end-to-end latency metrics")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--jitter-ms", type=int, default=50, help="rtpjitterbuffer latency")
    p.add_argument("--frames", type=int, default=-1, help="stop after N frames (-1 = until Ctrl+C)")
    p.add_argument("--display", action="store_true", help="show the received video")
    p.add_argument("--record", metavar="PATH", help="save the received video with overlays (.webm)")
    p.add_argument("--idle-timeout", type=float, default=3.0,
                   help="seconds without frames before assuming the stream ended (after loss the "
                        "decoder can go silent until the next keyframe)")
    p.add_argument("--meta-port", type=int, default=None,
                   help="UDP port for detection metadata (default: --port + 1)")
    p.add_argument("--no-meta", action="store_true", help="ignore detection metadata")
    p.add_argument("--no-overlay", action="store_true", help="don't draw detections on the video")
    p.add_argument("--warmup", type=int, default=10, help="frames excluded from the summary")
    p.add_argument("--csv", default="receive.csv")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args(argv)

    cap = Capture("rtp", port=args.port, jitter_ms=args.jitter_ms,
                  width=args.width, height=args.height)
    out = None
    if args.display:
        out = Output("display", args.width, args.height, 30)
    elif args.record:
        out = Output("file", args.width, args.height, 30, path=args.record)
    meta = None if args.no_meta else MetaReceiver(args.meta_port or args.port + 1, keep=3000)
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
            hit = meta.lookup(stamp) if meta else None
            lead = None if hit is None else timestamp.now_ms() - hit[1]
            # Naive alternative: draw whatever detections arrived last. How often is that
            # a different frame than the one on screen?
            naive_wrong = bool(meta and meta.latest_ts is not None and meta.latest_ts != stamp)
            rows.append({"stamp_ms": stamp, "e2e_ms": e2e, "meta_lead_ms": lead,
                         "naive_wrong": naive_wrong})
            if out:
                boxes = hit[0]["boxes"] if hit and not args.no_overlay else []
                if not meta:
                    info = "no metadata"
                elif hit is None:
                    info = "detections: not arrived"
                else:
                    info = f"objs {len(hit[0]['boxes'])} | proc {hit[0].get('proc_ms', 0):.1f} ms"
                out.push(annotate(frame.image, boxes, f"RECEIVED | e2e {e2e:4d} ms | {info}"))
    finally:
        cap.stop()
        if out:
            out.close()

    if rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    s = summarize(rows, unreadable, args.warmup, meta)
    if meta:
        meta.close()
    print(json.dumps(s, indent=2))
    return s


if __name__ == "__main__":
    main()
