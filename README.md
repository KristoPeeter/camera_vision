# camvision: GStreamer + OpenCV motion detection with latency metrics

A small demo of a camera pipeline: grab frames with **GStreamer**, detect motion with **OpenCV**,
push annotated frames back into GStreamer for display or recording, and measure what happens
on every frame.

```
v4l2src / videotestsrc / filesrc ─► decode ─► BGR ─► appsink ──► OpenCV MOG2 motion detection
                                                                     │
                     autovideosink / vp8enc ! webmmux ◄── appsrc ◄───┘ (boxes + HUD)
                                                                     │
                                                    metrics.csv + p50/p95 summary
```

## Run

```bash
python3 -m venv --system-site-packages .venv      # system site-packages for GStreamer's `gi`
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m camvision --source test --output display          # synthetic bouncing ball
.venv/bin/python -m camvision --source webcam --output display --frames -1
.venv/bin/python -m camvision --source webcam --exposure 1200 --output display   # bright, 8 fps
.venv/bin/python -m camvision --source webcam --exposure 330 --gain --output display  # 30 fps, noisier
.venv/bin/python -m pytest -q
```

## Metrics

| metric | meaning |
|---|---|
| `capture_latency_ms` | pipeline clock "now" minus the buffer PTS when Python receives it (time from capture to our code) |
| `process_ms` | time spent in the OpenCV detector |
| `source_interval_ms` | median PTS gap = the frame rate the camera *actually* delivers |
| `dropped` | frames missing between source and app (the appsink keeps only the newest frame) |

## Things learned on real hardware

1. **Caps negotiation matters.** Left to negotiate freely, the UVC webcam chose raw YUY2 at
   1280x720, which it can only send at 10 fps over USB2. Pinning `image/jpeg,640x480,30/1` fixed that.
2. **Exposure limits frame rate.** A frame can't be shorter than its exposure. In a dim room the
   image only looked right at `--exposure 1200` (120 ms), so the camera ran at **8.3 fps**, and
   auto-exposure picked the same value. For 30 fps the exposure has to be at most 33 ms (`330`).
   Measured on this webcam, which has no analog gain control:

   | setting | fps | mean luma | contrast (std) |
   |---|---|---|---|
   | exposure 1200 | 8 | 43 | 45 |
   | exposure 330, raw | 30 | 14 | 17 |
   | exposure 330 + camera gamma 500 | 30 | 20 | 30 |
   | exposure 330 + camera brightness +64 | 30 | 115 | low: washed-out grey |
   | **exposure 330 + OpenCV gain (`--gain`)** | **30** | **37** | **47** |

   Digital gain gives 30 fps with roughly the same look, but it is noisier. The long exposure is
   cleaner but blurs anything that moves, which is bad for motion detection.
   Also, **mean brightness alone is a misleading metric**: the camera's brightness control "won"
   on mean luma but just added a grey offset.
3. **V4L2 controls persist on the device.** A setting like `brightness=64` stays after the
   program exits and silently affected the next test. The capture code now sets every image
   control explicitly when it starts.
4. **Don't trust the requested fps in metrics.** The first drop counter assumed 30 fps and
   reported a 74% drop rate. Estimating the interval from the real PTS gaps shows 0 drops.
