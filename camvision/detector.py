"""Motion detection with OpenCV background subtraction (MOG2)."""
from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    def __init__(self, min_area: int = 500, history: int = 200, var_threshold: float = 32):
        self.min_area = min_area
        self.bg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold,
                                                     detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return bounding boxes (x, y, w, h) of moving regions."""
        mask = self.bg.apply(img)
        # MOG2 marks shadows as 127 - drop them, keep confident foreground (255)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= self.min_area]


def annotate(img: np.ndarray, boxes, hud: str | None) -> np.ndarray:
    for x, y, w, h in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    if hud is None:
        return img
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, hud, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def auto_gain(img: np.ndarray, target: float = 200, max_gain: float = 8.0) -> tuple[np.ndarray, float]:
    """Digital gain: scale so the 99th-percentile pixel lands near `target`.

    Lets a short exposure (high fps, little motion blur) look as bright as a long one.
    It also amplifies noise, which is why real cameras prefer analog sensor gain.
    """
    g = float(np.clip(target / max(1.0, float(np.percentile(img[::4, ::4], 99))), 1.0, max_gain))
    return (cv2.convertScaleAbs(img, alpha=g) if g > 1.0 else img), g
