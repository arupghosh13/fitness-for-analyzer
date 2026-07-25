FROM python:3.12-slim

# System deps required by OpenCV/MediaPipe/streamlit-webrtc (av/ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the pose model at build time so the image is self-contained
# (see docs/setup.md for the model variant tradeoffs)
RUN mkdir -p models && curl -o models/pose_landmarker_lite.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

COPY src/ src/
COPY .env.example .env

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
