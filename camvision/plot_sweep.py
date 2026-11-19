"""Charts for the README from results/sweep.csv.  Run: python -m camvision.plot_sweep"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]          # validated categorical slots 1-2 (blue, orange)


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)


def encoder_cpu(rows: list[dict], out: Path) -> None:
    cases = [("640x480", "2000"), ("1920x1080", "8000")]
    labels = ["640x480, 2 Mbit/s", "1920x1080, 8 Mbit/s"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    width = 0.36
    for i, enc in enumerate(["x264", "nvenc"]):
        vals = [float(next(r for r in rows if r["experiment"] == "encoder" and r["encoder"] == enc
                           and r["res"] == res and r["bitrate"] == br)["sender_cpu_%"])
                for res, br in cases]
        xs = [x + (i - 0.5) * (width + 0.02) for x in range(len(cases))]   # 2% gap between bars
        bars = ax.bar(xs, vals, width, color=SERIES[i],
                      label="x264 (CPU)" if enc == "x264" else "NVENC (GPU)")
        ax.bar_label(bars, fmt="%.0f%%", color=INK, fontsize=9, padding=3)
    ax.set_xticks(range(len(cases)), labels)
    _style(ax, "Sender CPU: capture + encode", "% of one CPU core")
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def loss_keyint(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    for i, loss in enumerate(["0.5", "2.0"]):
        pts = sorted((int(r["key_int"]), float(r["intact_%"])) for r in rows
                     if r["experiment"] == "loss" and r["net_loss"] == loss)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{float(loss):g}% packet loss", (xs[-1], ys[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_xticks([10, 30, 90], ["every 10", "every 30", "every 90"])
    ax.set_xlabel("keyframe interval (frames)", color=INK2)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 125)
    _style(ax, "Frames arriving intact under packet loss (NVENC, 1 run each)", "% of frames intact")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(path: str = "results/sweep.csv") -> None:
    rows = list(csv.DictReader(open(path)))
    out = Path(path).parent
    encoder_cpu(rows, out / "encoder_cpu.png")
    loss_keyint(rows, out / "loss_keyint.png")
    print(f"wrote {out / 'encoder_cpu.png'} and {out / 'loss_keyint.png'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
