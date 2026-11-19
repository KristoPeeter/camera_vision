"""Burn a capture timestamp into the image as black/white blocks, and read it back.

This measures latency end to end, including encoding, the network and decoding,
because the timestamp travels inside the picture itself.

Layout: 2 rows x 20 blocks of 16x16 px in the bottom-left corner (320x32 px).
Blocks are aligned to H.264's 16x16 macroblocks and use only pure black or white,
so they survive lossy compression even at low bitrates.

40 bits = 32-bit timestamp (wall-clock ms, wraps every ~49 days) + 8-bit checksum,
so a frame whose stamp got damaged is rejected instead of giving a wrong latency.
"""
from __future__ import annotations

import time

import numpy as np

BLOCK = 16
COLS, ROWS = 20, 2
BITS = COLS * ROWS            # 40
MASK32 = 0xFFFFFFFF


def now_ms() -> int:
    # Wall clock, so sender and receiver agree when on the same machine
    # (or on machines with synced clocks).
    return time.time_ns() // 1_000_000


def _checksum(value: int) -> int:
    b = value.to_bytes(4, "big")
    return (b[0] ^ b[1] ^ b[2] ^ b[3] ^ 0xA5) & 0xFF   # 0xA5: an all-black area doesn't pass


def _origin(img: np.ndarray) -> tuple[int, int]:
    return img.shape[0] - ROWS * BLOCK, 0          # (y, x) of top-left corner


def stamp(img: np.ndarray, ms: int) -> np.ndarray:
    value = ms & MASK32
    word = (value << 8) | _checksum(value)
    y0, x0 = _origin(img)
    for i in range(BITS):
        bit = (word >> (BITS - 1 - i)) & 1
        r, c = divmod(i, COLS)
        y, x = y0 + r * BLOCK, x0 + c * BLOCK
        img[y:y + BLOCK, x:x + BLOCK] = 255 if bit else 0
    return img


def read(img: np.ndarray) -> int | None:
    """Return the 32-bit ms timestamp, or None if the checksum fails."""
    y0, x0 = _origin(img)
    gray = img.mean(axis=2) if img.ndim == 3 else img
    word = 0
    q = BLOCK // 4                                  # sample the block centre; edges blur
    for i in range(BITS):
        r, c = divmod(i, COLS)
        y, x = y0 + r * BLOCK, x0 + c * BLOCK
        word = (word << 1) | int(gray[y + q:y + BLOCK - q, x + q:x + BLOCK - q].mean() > 128)
    value, check = word >> 8, word & 0xFF
    return value if _checksum(value) == check else None


def age_ms(stamped_ms: int, now: int | None = None) -> int:
    """now - stamped, handling the 32-bit wraparound."""
    now = now_ms() if now is None else now
    return ((now & MASK32) - stamped_ms) & MASK32
