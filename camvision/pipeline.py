"""GStreamer capture (appsink) and output (appsrc) wrappers that speak numpy.

Capture:  <source> ! videoconvert ! BGR ! appsink   -> numpy frames + PTS
Output:   appsrc (BGR) ! videoconvert ! <sink>      <- annotated numpy frames
"""
from __future__ import annotations

from dataclasses import dataclass

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

SOURCES = {
    # Webcam via Video4Linux2. Pin the camera's own output caps: left to negotiate freely, UVC cams
    # often pick raw YUY2 at max resolution, which USB2 bandwidth limits to ~10 fps. MJPEG is
    # compressed on the camera, so 640x480@30 fits easily; we decode it on the CPU.
    "webcam": "v4l2src device={device} {controls} ! image/jpeg,width={width},height={height},framerate={fps}/1 "
              "! jpegdec",
    # Synthetic moving ball - handy without a camera, and deterministic for tests.
    "test": "videotestsrc pattern=ball is-live=true num-buffers={num_buffers} "
            "! video/x-raw,width={width},height={height},framerate={fps}/1",
    "file": "filesrc location={device} ! decodebin ! videoconvert ! videoscale "
            "! video/x-raw,width={width},height={height}",
}


@dataclass
class Frame:
    image: np.ndarray       # HxWx3 BGR
    pts_ns: int             # buffer presentation timestamp (pipeline running time)
    latency_ns: int         # pipeline clock "now" minus PTS when we received it


class Capture:
    def __init__(self, source: str = "test", device: str = "/dev/video0",
                 width: int = 640, height: int = 480, fps: int = 30, num_buffers: int = -1,
                 exposure: int | None = None):
        # exposure: manual exposure in 100us units (V4L2 exposure_time_absolute). None = auto.
        # In low light, auto-exposure lengthens exposure and silently lowers the frame rate:
        # frame period >= exposure, so 30 fps allows at most ~330 (33 ms).
        # V4L2 controls are device state and persist after we close it, so always set the
        # image controls explicitly instead of inheriting whatever the last program left.
        ae = "auto_exposure=3" if exposure is None else \
            f"auto_exposure=1,exposure_time_absolute={exposure}"
        controls = f'extra-controls="c,{ae},brightness=0,contrast=50,gamma=300"'
        src = SOURCES[source].format(device=device, width=width, height=height, fps=fps,
                                     num_buffers=num_buffers, controls=controls)
        # max-buffers=1 drop=true: always hand us the newest frame (low latency over completeness)
        desc = (f"{src} ! videoconvert ! video/x-raw,format=BGR "
                f"! appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false")
        self.description = desc
        self.pipeline = Gst.parse_launch(desc)
        self.sink = self.pipeline.get_by_name("sink")
        self.live = source != "file"

    def start(self) -> None:
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"Failed to start pipeline:\n  {self.description}")

    def read(self, timeout_s: float = 2.0) -> Frame | None:
        sample = self.sink.emit("try-pull-sample", int(timeout_s * Gst.SECOND))
        if sample is None:
            self._raise_on_error()
            return None  # EOS or timeout
        buf = sample.get_buffer()
        s = sample.get_caps().get_structure(0)
        w, h = s.get_value("width"), s.get_value("height")
        ok, info = buf.map(Gst.MapFlags.READ)
        try:
            # Rows may be padded; derive stride from buffer size.
            stride = info.size // h
            arr = np.frombuffer(info.data, dtype=np.uint8).reshape(h, stride)[:, : w * 3]
            img = arr.reshape(h, w, 3).copy()
        finally:
            buf.unmap(info)

        latency = 0
        clock = self.pipeline.get_clock()
        if clock is not None and buf.pts != Gst.CLOCK_TIME_NONE:
            running_time = clock.get_time() - self.pipeline.get_base_time()
            latency = max(0, running_time - buf.pts)
        return Frame(img, buf.pts, latency)

    def _raise_on_error(self) -> None:
        msg = self.pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
        if msg:
            err, dbg = msg.parse_error()
            raise RuntimeError(f"GStreamer error: {err.message}\n{dbg}")

    def stop(self) -> None:
        self.pipeline.set_state(Gst.State.NULL)


class Output:
    """Push BGR frames into a GStreamer sink: on-screen window or a WebM file."""

    def __init__(self, kind: str, width: int, height: int, fps: int, path: str = "out.webm"):
        caps = f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1"
        sinks = {
            "display": "videoconvert ! autovideosink sync=false",
            "file": f"videoconvert ! vp8enc deadline=1 ! webmmux ! filesink location={path}",
        }
        self.pipeline = Gst.parse_launch(
            f"appsrc name=src is-live=true format=time caps={caps} ! {sinks[kind]}")
        self.src = self.pipeline.get_by_name("src")
        self.frame_ns = Gst.SECOND // fps
        self.n = 0
        self.pipeline.set_state(Gst.State.PLAYING)

    def push(self, img: np.ndarray) -> None:
        buf = Gst.Buffer.new_wrapped(img.tobytes())
        buf.pts = self.n * self.frame_ns
        buf.duration = self.frame_ns
        self.n += 1
        self.src.emit("push-buffer", buf)

    def close(self) -> None:
        self.src.emit("end-of-stream")
        # wait for EOS so the file muxer can finalize
        self.pipeline.get_bus().timed_pop_filtered(5 * Gst.SECOND,
                                                    Gst.MessageType.EOS | Gst.MessageType.ERROR)
        self.pipeline.set_state(Gst.State.NULL)
