FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1
# GStreamer + Python bindings come from apt; OpenCV/numpy from pip.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
        gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY camvision camvision
COPY tests tests
# Default: synthetic source, no camera needed. For a webcam:
#   docker run --rm --device /dev/video0 camvision --source webcam --frames 300
ENTRYPOINT ["python3", "-m", "camvision"]
