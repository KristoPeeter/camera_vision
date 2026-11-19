"""GStreamer capture (appsink) and output (appsrc) wrappers that speak numpy.

Capture:  <source> ! videoconvert ! BGR ! appsink   -> numpy frames + PTS
Output:   appsrc (BGR) ! videoconvert ! <sink>      <- annotated numpy frames
"""
from __future__ import annotations

from dataclasses import dataclass

import time

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
    "test": "videotestsrc {pattern} is-live=true num-buffers={num_buffers} "
            "! video/x-raw,width={width},height={height},framerate={fps}/1",
    "file": "filesrc location={device} ! decodebin ! videoconvert ! videoscale "
            "! video/x-raw,width={width},height={height}",
    # Receive H.264 over RTP/UDP. The jitter buffer reorders packets and absorbs network
    # jitter, but it holds every frame for `latency` ms - a direct latency/robustness trade-off.
    "rtp": "udpsrc port={port} caps=\"application/x-rtp,media=video,encoding-name=H264,"
           "clock-rate=90000,payload=96\" ! rtpjitterbuffer latency={jitter_ms} "
           "! rtph264depay ! h264parse ! avdec_h264",
    # Pull from an RTSP server (an IP camera, or our own --output rtsp).
    "rtsp": "rtspsrc location={url} latency={jitter_ms} ! rtph264depay ! h264parse ! avdec_h264",
}


@dataclass
class Frame:
    image: np.ndarray       # HxWx3 BGR
    pts_ns: int             # buffer presentation timestamp (pipeline running time)
    latency_ns: int         # pipeline clock "now" minus PTS when we received it
    capture_wall_ms: int    # wall-clock time of capture (now - latency), for end-to-end stamps


class Capture:
    def __init__(self, source: str = "test", device: str = "/dev/video0",
                 width: int = 640, height: int = 480, fps: int = 30, num_buffers: int = -1,
                 exposure: int | None = None, port: int = 5000, jitter_ms: int = 50,
                 pattern: str = "pattern=ball", url: str = ""):
        # exposure: manual exposure in 100us units (V4L2 exposure_time_absolute). None = auto.
        # In low light, auto-exposure lengthens exposure and silently lowers the frame rate:
        # frame period >= exposure, so 30 fps allows at most ~330 (33 ms).
        # V4L2 controls are device state and persist after we close it, so always set the
        # image controls explicitly instead of inheriting whatever the last program left.
        ae = "auto_exposure=3" if exposure is None else \
            f"auto_exposure=1,exposure_time_absolute={exposure}"
        controls = f'extra-controls="c,{ae},brightness=0,contrast=50,gamma=300"'
        src = SOURCES[source].format(device=device, width=width, height=height, fps=fps,
                                     num_buffers=num_buffers, controls=controls,
                                     port=port, jitter_ms=jitter_ms, pattern=pattern, url=url)
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
        return Frame(img, buf.pts, latency, (time.time_ns() - latency) // 1_000_000)

    def _raise_on_error(self) -> None:
        msg = self.pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
        if msg:
            err, dbg = msg.parse_error()
            raise RuntimeError(f"GStreamer error: {err.message}\n{dbg}")

    def stop(self) -> None:
        self.pipeline.set_state(Gst.State.NULL)


# H.264 encoders tuned for latency. bitrate in kbit/s, key_int = frames between keyframes.
ENCODERS = {
    # zerolatency: no B-frames, no lookahead, sliced threads -> each frame leaves immediately
    # Each encoder pinned to 4:2:0 input. Left free, videoconvert may pick 4:4:4, which the
    # baseline profile that RTSP/phone players ask for doesn't support.
    "x264": "video/x-raw,format=I420 ! x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate} key-int-max={key_int}",
    # NVENC on the GPU. nvcudah264enc is the newer element: the older nvh264enc fails on
    # current drivers with "Selected preset not supported" (legacy presets were removed).
    "nvenc": "video/x-raw,format=NV12 ! nvcudah264enc preset=p1 tune=ultra-low-latency zero-reorder-delay=true "
             "rate-control=cbr bitrate={bitrate} gop-size={key_int}",
}


class Output:
    """Push BGR frames into a GStreamer sink: on-screen window, WebM file or RTP/UDP stream."""

    def __init__(self, kind: str, width: int, height: int, fps: int, path: str = "out.webm",
                 host: str = "127.0.0.1", port: int = 5000, encoder: str = "x264",
                 bitrate: int = 2000, key_int: int = 30, loss_pct: float = 0, jitter_ms: int = 0):
        caps = f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1"
        enc = ENCODERS[encoder].format(bitrate=bitrate, key_int=key_int)
        # Optional bad-network simulation on the RTP packets, no root/tc needed:
        # drop loss_pct% of packets, and delay 50% of packets by 0..jitter_ms (causes reordering).
        net = ""
        if loss_pct or jitter_ms:
            net = (f"netsim drop-probability={loss_pct / 100} delay-probability={0.5 if jitter_ms else 0} "
                   f"min-delay=0 max-delay={jitter_ms} allow-reordering=true ! ")
        sinks = {
            "display": "videoconvert ! autovideosink sync=false",
            "file": f"videoconvert ! vp8enc deadline=1 ! webmmux ! filesink location={path}",
            # config-interval=-1 resends SPS/PPS with every keyframe, so a receiver can join late
            "rtp": f"videoconvert ! {enc} ! h264parse "
                   f"! rtph264pay config-interval=-1 pt=96 ! {net}udpsink host={host} port={port} sync=false",
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


class RtspOutput:
    """Serve the stream as rtsp://<address>:<port><mount>, the way IP cameras do.

    One shared pipeline: every client (VLC, ffplay, the receiver) gets the same encoded
    stream, so the encoder runs once no matter how many viewers connect. Frames are only
    pushed while at least one client is connected.
    """

    def __init__(self, width: int, height: int, fps: int, encoder: str = "x264",
                 bitrate: int = 2000, key_int: int = 30, address: str = "127.0.0.1",
                 port: int = 8554, mount: str = "/cam", tcp_only: bool = False):
        import threading

        gi.require_version("GstRtspServer", "1.0")
        from gi.repository import GLib, GstRtspServer

        caps = f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1"
        enc = ENCODERS[encoder].format(bitrate=bitrate, key_int=key_int)
        factory = GstRtspServer.RTSPMediaFactory()
        # do-timestamp: appsrc stamps each buffer with the pipeline clock when we push it
        factory.set_launch(f"( appsrc name=src is-live=true format=time do-timestamp=true "
                           f"caps={caps} ! videoconvert ! {enc} ! h264parse "
                           f"! rtph264pay name=pay0 pt=96 config-interval=-1 )")
        factory.set_shared(True)
        if tcp_only:
            # Send video inside the RTSP TCP connection (port 8554) instead of separate UDP
            # ports: works through firewalls/NAT with no client settings, at the cost of
            # head-of-line blocking (a lost packet delays everything behind it).
            gi.require_version("GstRtsp", "1.0")
            from gi.repository import GstRtsp
            factory.set_protocols(GstRtsp.RTSPLowerTrans.TCP)
        factory.connect("media-configure", self._on_media_configure)

        self.server = GstRtspServer.RTSPServer()
        self.server.set_address(address)
        self.server.set_service(str(port))
        self.server.get_mount_points().add_factory(mount, factory)
        if self.server.attach(None) == 0:
            raise RuntimeError(f"RTSP server could not listen on {address}:{port}")
        self.url = f"rtsp://{address}:{port}{mount}"
        self.appsrc = None
        self.loop = GLib.MainLoop()
        threading.Thread(target=self.loop.run, daemon=True).start()   # serves clients

    def _on_media_configure(self, factory, media) -> None:
        self.appsrc = media.get_element().get_child_by_name("src")
        media.connect("unprepared", lambda *_: setattr(self, "appsrc", None))  # last client left

    @property
    def clients(self) -> bool:
        return self.appsrc is not None

    def push(self, img: np.ndarray) -> None:
        src = self.appsrc
        if src is not None:
            src.emit("push-buffer", Gst.Buffer.new_wrapped(img.tobytes()))

    def close(self) -> None:
        self.loop.quit()
