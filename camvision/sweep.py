"""Run sender + receiver for many settings and collect the results in one CSV.

Run:  python -m camvision.sweep                 (about 5 minutes)
      python -m camvision.sweep --quick         (one run per experiment, as a smoke test)

Each run starts the receiver, then streams a fixed number of frames from the synthetic
source, so runs are repeatable. CPU time is measured per process (user + system).
Motion detection is off in the sender, so its CPU % is capture + encoding only
(detection alone costs about two cores and would hide the encoder difference).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

FRAMES = 300
FPS = 30
# Moving colour bars: more realistic for an encoder than a ball on black.
PATTERN = "pattern=smpte horizontal-speed=4"

BASE = {"encoder": "x264", "res": "640x480", "bitrate": 2000, "key_int": 30, "jitter_buffer": 50,
        "net_loss": 0.0, "net_jitter": 0}

EXPERIMENTS = {
    # Which encoder is faster and cheaper, and do bitrate and resolution change latency?
    "encoder": [dict(encoder=e, res=r, bitrate=b) for e in ("x264", "nvenc")
                for r, b in (("640x480", 500), ("640x480", 2000), ("640x480", 8000),
                             ("1920x1080", 8000))],
    # Jitter buffer size vs network jitter: latency vs frames ruined by late packets.
    "jitter": [dict(encoder="nvenc", jitter_buffer=jb, net_jitter=nj)
               for nj in (0, 30) for jb in (0, 50, 200)],
    # Packet loss: does a shorter keyframe interval limit the damage?
    "loss": [dict(encoder="nvenc", net_loss=l, key_int=k) for l in (0.5, 2.0) for k in (10, 30, 90)],
}


def _cpu_children() -> float:
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def _json_from(stdout: str) -> dict:
    start = stdout.find("{\n")
    return json.loads(stdout[start:]) if start >= 0 else {}


def run_one(cfg: dict, port: int) -> dict:
    py = [sys.executable, "-m"]
    w, h = cfg["res"].split("x")
    rx_cmd = py + ["camvision.receive", "--port", str(port), "--jitter-ms", str(cfg["jitter_buffer"]),
                   "--csv", os.devnull]
    tx_cmd = py + ["camvision", "--source", "test", "--pattern", PATTERN, "--frames", str(FRAMES),
                   "--fps", str(FPS), "--width", w, "--height", h, "--no-detect",
                   "--output", "rtp", "--port", str(port),
                   "--encoder", cfg["encoder"], "--bitrate", str(cfg["bitrate"]),
                   "--key-int", str(cfg["key_int"]), "--net-loss", str(cfg["net_loss"]),
                   "--net-jitter", str(cfg["net_jitter"]), "--csv", os.devnull]

    rx = subprocess.Popen(rx_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(1.0)                                    # let the receiver bind its port

    cpu0 = _cpu_children()                             # children's CPU counts once reaped
    tx = subprocess.run(tx_cmd, capture_output=True, text=True, timeout=60)
    cpu_tx = _cpu_children() - cpu0
    try:
        rx_out, _ = rx.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        rx.kill()
        rx_out, _ = rx.communicate()
    cpu_rx = _cpu_children() - cpu0 - cpu_tx

    t, r = _json_from(tx.stdout), _json_from(rx_out)
    if not t:
        raise RuntimeError(f"sender failed:\n{tx.stdout}\n{tx.stderr}")
    duration = t.get("duration_s") or FRAMES / FPS
    return {
        **cfg,
        "e2e_ms_p50": r.get("e2e_ms_p50"),
        "e2e_ms_p95": r.get("e2e_ms_p95"),
        # Frames that arrived with a readable stamp: they were decoded without visible damage
        # to that corner. Corrupted frames (bad stamp) and frames never output both count against it.
        "intact_%": round(100 * r.get("frames_total", 0) / FRAMES, 1),
        "corrupted_frames": r.get("unreadable_stamps", 0),
        "sender_cpu_%": round(100 * cpu_tx / duration, 1),
        "receiver_cpu_%": round(100 * cpu_rx / duration, 1),
    }


def to_markdown(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main(argv=None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", choices=list(EXPERIMENTS), nargs="*", help="run only these experiments")
    p.add_argument("--quick", action="store_true", help="first config of each experiment only")
    p.add_argument("--out", default="results/sweep.csv")
    p.add_argument("--port", type=int, default=5100)
    args = p.parse_args(argv)

    rows = []
    for name, configs in EXPERIMENTS.items():
        if args.only and name not in args.only:
            continue
        for cfg in configs[:1] if args.quick else configs:
            cfg = {"experiment": name, **BASE, **cfg}
            print(f"[{name}] {cfg}", flush=True)
            row = run_one(cfg, args.port)
            print(f"   -> p50 {row['e2e_ms_p50']} ms, p95 {row['e2e_ms_p95']} ms, "
                  f"intact {row['intact_%']}%, tx cpu {row['sender_cpu_%']}%", flush=True)
            rows.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")
    return rows


if __name__ == "__main__":
    main()
