import numpy as np
import pytest

from camvision.detector import MotionDetector
from camvision.metrics import Metrics, estimate_drops


def _frame(x=None, size=(240, 320)):
    img = np.zeros((*size, 3), np.uint8)
    if x is not None:
        img[100:140, x:x + 40] = 255
    return img


def test_static_scene_has_no_motion():
    det = MotionDetector(min_area=100)
    for _ in range(30):
        boxes = det.detect(_frame())
    assert boxes == []


def test_moving_square_is_detected_near_its_position():
    det = MotionDetector(min_area=100)
    for _ in range(30):
        det.detect(_frame())             # learn empty background
    boxes = det.detect(_frame(x=150))
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert x <= 150 <= x + w and y <= 100 <= y + h


@pytest.mark.parametrize("pts,expected", [
    ([0, 33, 66, 100, 133], 0),          # steady 30 fps
    ([0, 33, 66, 133, 166], 1),          # one frame missing
    ([0, 133, 266, 400, 533], 0),        # camera slowed to 7.5 fps: not "drops"
    ([0, 33, 66, 200, 233, 266], 3),     # three missing
])
def test_estimate_drops(pts, expected):
    dropped, _ = estimate_drops(pts)
    assert dropped == expected


def test_summary_percentiles():
    m = Metrics()
    for i in range(100):
        m.record(pts_ns=i * 33_333_333, capture_latency_ns=1_000_000 * i, process_s=0.005, detections=i % 2)
    s = m.summary()
    assert s["frames"] == 100 and s["dropped"] == 0
    assert s["capture_latency_ms_p50"] == pytest.approx(49.5)
    assert s["frames_with_motion_%"] == 50.0


def test_end_to_end_with_gstreamer_test_source(tmp_path):
    from camvision.__main__ import main
    s = main(["--source", "test", "--frames", "60", "--csv", str(tmp_path / "m.csv")])
    assert s["frames"] == 60
    assert s["frames_with_motion_%"] > 50      # the bouncing ball keeps moving
    assert (tmp_path / "m.csv").read_text().count("\n") == 61


def test_auto_gain_brightens_dark_frame_and_leaves_bright_alone():
    from camvision.detector import auto_gain
    dark = np.full((100, 100, 3), 25, np.uint8)
    out, g = auto_gain(dark)
    assert g == pytest.approx(8.0) and out.mean() == pytest.approx(200)
    bright = np.full((100, 100, 3), 230, np.uint8)
    out, g = auto_gain(bright)
    assert g == 1.0 and out is bright
