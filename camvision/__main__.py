"""Run: python -m camvision --source webcam --output display"""
from __future__ import annotations

import argparse
import json
import signal
import time

from .detector import MotionDetector, annotate, auto_gain
from . import timestamp
from .meta import MetaSender
from .metrics import Metrics
from .pipeline import Capture, Output, RtspOutput


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="GStreamer + OpenCV motion detection with latency metrics")
    p.add_argument("--source", choices=["webcam", "test", "file"], default="test")
    p.add_argument("--device", default="/dev/video0", help="camera device or video file path")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frames", type=int, default=300, help="stop after N frames (-1 = forever)")
    p.add_argument("--output", choices=["none", "display", "file", "rtp", "rtsp"], default="none")
    p.add_argument("--out-path", default="out.webm")
    p.add_argument("--host", default="127.0.0.1", help="rtp: receiver address")
    p.add_argument("--port", type=int, default=5000, help="rtp: receiver UDP port")
    p.add_argument("--meta-port", type=int, default=None,
                   help="rtp: UDP port for detection metadata (default: --port + 1)")
    p.add_argument("--rtsp-address", default="127.0.0.1",
                   help="rtsp: address to listen on. 127.0.0.1 = this machine only; "
                        "0.0.0.0 = anyone on your network can watch (unencrypted, no password)")
    p.add_argument("--rtsp-port", type=int, default=8554)
    p.add_argument("--rtsp-tcp", action="store_true",
                   help="rtsp: send video over the RTSP TCP connection only (firewall/phone friendly)")
    p.add_argument("--burn-boxes", action="store_true",
                   help="rtp/rtsp: also draw boxes into the video, for players that can't read "
                        "the metadata (VLC, phones)")
    p.add_argument("--encoder", choices=["x264", "nvenc"], default="x264",
                   help="rtp: H.264 encoder (x264 = CPU, nvenc = NVIDIA GPU)")
    p.add_argument("--net-loss", type=float, default=0, help="rtp: simulated packet loss %%")
    p.add_argument("--net-jitter", type=int, default=0, help="rtp: simulated jitter, max ms")
    p.add_argument("--pattern", default="pattern=ball",
                   help='test source: videotestsrc properties, e.g. "pattern=smpte horizontal-speed=4"')
    p.add_argument("--bitrate", type=int, default=2000, help="rtp: kbit/s")
    p.add_argument("--key-int", type=int, default=30, help="rtp: frames between keyframes")
    p.add_argument("--csv", default="metrics.csv")
    p.add_argument("--exposure", type=int, default=None,
                   help="webcam manual exposure in 100us units (e.g. 150 = 15 ms); default auto")
    p.add_argument("--gain", action="store_true",
                   help="auto digital gain in OpenCV (pair with a short --exposure in low light)")
    p.add_argument("--no-detect", action="store_true",
                   help="skip motion detection (e.g. to measure encoder CPU cost alone)")
    p.add_argument("--min-area", type=int, default=500)
    args = p.parse_args(argv)

    cap = Capture(args.source, args.device, args.width, args.height, args.fps,
                  num_buffers=args.frames if args.source == "test" else -1, exposure=args.exposure,
                  pattern=args.pattern)
    det = MotionDetector(min_area=args.min_area)
    metrics = Metrics()
    if args.output == "none":
        out = None
    elif args.output == "rtsp":
        out = RtspOutput(args.width, args.height, args.fps, encoder=args.encoder,
                         bitrate=args.bitrate, key_int=args.key_int,
                         address=args.rtsp_address, port=args.rtsp_port, tcp_only=args.rtsp_tcp)
        print(f"serving {_rtsp_urls(args.rtsp_address, args.rtsp_port)}", flush=True)
    else:
        out = Output(args.output, args.width, args.height, args.fps, args.out_path,
                     host=args.host, port=args.port, encoder=args.encoder, bitrate=args.bitrate,
                     key_int=args.key_int, loss_pct=args.net_loss, jitter_ms=args.net_jitter)
    # With rtp, video goes out clean and detections travel separately as metadata.
    meta = None
    if args.output in ("rtp", "rtsp"):
        meta = MetaSender(args.host, args.meta_port or args.port + 1)
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
            boxes = [] if args.no_detect else det.detect(frame.image)
            proc = time.perf_counter() - t
            metrics.record(frame.pts_ns, frame.latency_ns, proc, len(boxes))
            if meta:
                meta.send(frame.capture_wall_ms, boxes, proc_ms=round(proc * 1e3, 1),
                          gain=round(gain, 2))
                img = annotate(frame.image, boxes, None) if args.burn_boxes else frame.image
                out.push(timestamp.stamp(img, frame.capture_wall_ms))
            elif out:
                hud = (f"{metrics.live_fps():5.1f} fps | lat {frame.latency_ns / 1e6:5.1f} ms | "
                       f"proc {proc * 1e3:4.1f} ms | gain x{gain:.1f} | objs {len(boxes)}")
                img = annotate(frame.image, boxes, hud)
                out.push(timestamp.stamp(img, frame.capture_wall_ms))
    finally:
        cap.stop()
        if out:
            out.close()
        if meta:
            meta.close()

    metrics.to_csv(args.csv)
    s = metrics.summary()
    print(json.dumps(s, indent=2))
    return s


def _rtsp_urls(address: str, port: int) -> str:
    if address != "0.0.0.0":
        return f"rtsp://{address}:{port}/cam"
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))     # no packet is sent; picks the LAN interface
        lan = s.getsockname()[0]
    except OSError:
        lan = "<this-machine's-ip>"
    finally:
        s.close()
    return f"rtsp://{lan}:{port}/cam  (visible to your whole network)"


if __name__ == "__main__":
    main()
