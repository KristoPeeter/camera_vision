"""Run: python -m camvision --source webcam --output display"""
from __future__ import annotations

import argparse
import json
import signal
import time

from .detector import MotionDetector, annotate, auto_gain
from .metrics import Metrics
from .pipeline import Capture, Output


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="GStreamer + OpenCV motion detection with latency metrics")
    p.add_argument("--source", choices=["webcam", "test", "file"], default="test")
    p.add_argument("--device", default="/dev/video0", help="camera device or video file path")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frames", type=int, default=300, help="stop after N frames (-1 = forever)")
    p.add_argument("--output", choices=["none", "display", "file"], default="none")
    p.add_argument("--out-path", default="out.webm")
    p.add_argument("--csv", default="metrics.csv")
    p.add_argument("--exposure", type=int, default=None,
                   help="webcam manual exposure in 100us units (e.g. 150 = 15 ms); default auto")
    p.add_argument("--gain", action="store_true",
                   help="auto digital gain in OpenCV (pair with a short --exposure in low light)")
    p.add_argument("--min-area", type=int, default=500)
    args = p.parse_args(argv)

    cap = Capture(args.source, args.device, args.width, args.height, args.fps,
                  num_buffers=args.frames if args.source == "test" else -1, exposure=args.exposure)
    det = MotionDetector(min_area=args.min_area)
    metrics = Metrics()
    out = None if args.output == "none" else Output(args.output, args.width, args.height,
                                                     args.fps, args.out_path)
    stop = False

    def _on_sigint(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_sigint)

    print(f"pipeline: {cap.description}")
    cap.start()
    try:
        while not stop and (args.frames < 0 or len(metrics.rows) < args.frames):
            frame = cap.read()
            if frame is None:
                break
            t = time.perf_counter()
            gain = 1.0
            if args.gain:
                frame.image, gain = auto_gain(frame.image)
            boxes = det.detect(frame.image)
            proc = time.perf_counter() - t
            metrics.record(frame.pts_ns, frame.latency_ns, proc, len(boxes))
            if out:
                hud = (f"{metrics.live_fps():5.1f} fps | lat {frame.latency_ns / 1e6:5.1f} ms | "
                       f"proc {proc * 1e3:4.1f} ms | gain x{gain:.1f} | objs {len(boxes)}")
                out.push(annotate(frame.image, boxes, hud))
    finally:
        cap.stop()
        if out:
            out.close()

    metrics.to_csv(args.csv)
    s = metrics.summary()
    print(json.dumps(s, indent=2))
    return s


if __name__ == "__main__":
    main()
