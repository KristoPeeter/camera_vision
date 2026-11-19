"""Per-frame metrics: throughput, capture latency, processing time, frame gaps."""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Metrics:
    rows: list[dict] = field(default_factory=list)
    _t0: float | None = None

    def record(self, pts_ns: int, capture_latency_ns: int, process_s: float, detections: int):
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        self.rows.append({
            "t_s": now - self._t0,
            "pts_ms": pts_ns / 1e6,
            "capture_latency_ms": capture_latency_ns / 1e6,
            "process_ms": process_s * 1e3,
            "detections": detections,
        })

    def live_fps(self, window: int = 30) -> float:
        if len(self.rows) < 2:
            return 0.0
        r = self.rows[-window:]
        dt = r[-1]["t_s"] - r[0]["t_s"]
        return (len(r) - 1) / dt if dt > 0 else 0.0

    def summary(self) -> dict:
        if not self.rows:
            return {}
        lat = np.array([r["capture_latency_ms"] for r in self.rows])
        proc = np.array([r["process_ms"] for r in self.rows])
        dur = self.rows[-1]["t_s"]
        dropped, interval_ms = estimate_drops([r["pts_ms"] for r in self.rows])
        return {
            "frames": len(self.rows),
            "duration_s": round(dur, 2),
            "source_interval_ms": round(interval_ms, 2),
            "fps": round((len(self.rows) - 1) / dur, 1) if dur > 0 else 0.0,
            "dropped": dropped,
            "drop_rate_%": round(100 * dropped / (dropped + len(self.rows)), 2),
            "capture_latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
            "capture_latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
            "process_ms_p50": round(float(np.percentile(proc, 50)), 2),
            "process_ms_p95": round(float(np.percentile(proc, 95)), 2),
            "frames_with_motion_%": round(100 * np.mean([r["detections"] > 0 for r in self.rows]), 1),
        }

    def to_csv(self, path: str) -> None:
        if not self.rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)


def estimate_drops(pts_ms: list[float]) -> tuple[int, float]:
    """Count frames lost between source and our loop (e.g. appsink drop=true).

    The real source interval is the median PTS gap: this does not trust the requested
    fps, which a camera may silently miss (auto-exposure in low light lowers fps).
    A gap of k intervals means k-1 frames were dropped.
    """
    if len(pts_ms) < 3:
        return 0, 0.0
    gaps = np.diff(pts_ms)
    interval = float(np.median(gaps))
    if interval <= 0:
        return 0, interval
    return int(np.sum(np.maximum(np.round(gaps / interval) - 1, 0))), interval
