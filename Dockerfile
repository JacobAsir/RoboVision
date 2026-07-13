# RoboVision — Streamlit + YOLO on Render (or any Docker host)
# Tuned for ~512MB–2GB instances (ROBOVISION_LOW_MEM=1)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    ROBOVISION_LOW_MEM=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES="" \
    MALLOC_ARENA_MAX=2

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
# CPU-only torch is much smaller than default CUDA wheels on some platforms
RUN pip install --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Fail the build if required demo assets missing (skip heavy optional world model)
RUN test -f /app/main.py \
    && test -f /app/bottle-detection.mp4 \
    && test -f /app/5903898-hd_1920_1080_30fps.mp4 \
    && test -f /app/yolov8n.pt \
    && ls -lh /app/*.mp4 /app/yolov8n.pt \
    && echo "Demo assets OK"

ENV PORT=8501 \
    OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp
EXPOSE 8501

# Single worker / low memory Streamlit
CMD streamlit run main.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.maxUploadSize=10 \
    --browser.gatherUsageStats=false
