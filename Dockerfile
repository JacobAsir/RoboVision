# RoboVision — Streamlit + YOLO on Render (or any Docker host)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# OpenCV / Ultralytics runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Fail the build if demo videos/models were not copied (common Render/Git issue)
RUN test -f /app/main.py \
    && test -f /app/bottle-detection.mp4 \
    && test -f /app/5903898-hd_1920_1080_30fps.mp4 \
    && test -f /app/yolov8n.pt \
    && ls -lh /app/*.mp4 /app/*.pt \
    && echo "Demo assets OK"

# Render sets $PORT; default 8501 for local docker runs
ENV PORT=8501 \
    OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp
EXPOSE 8501

# Health-friendly Streamlit bind
CMD streamlit run main.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
