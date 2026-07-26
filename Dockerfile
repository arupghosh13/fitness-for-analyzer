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
# (see docs/setup.md for the model variant tradeoffs).
# -L: follow redirects (cloud storage URLs commonly redirect; without this,
#     curl silently saves the redirect response instead of the real file)
# -f: fail the build loudly on an HTTP error, instead of saving an error
#     page as if it were the model
# The size check catches ANY other silent-corruption case (e.g. a
# redirect to an unexpected small response) by failing the build outright
# rather than deploying a broken pose detector with no visible error until
# a user opens the app.
RUN mkdir -p models && \
    curl -Lf -o models/pose_landmarker_lite.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task && \
    MODEL_SIZE=$(stat -c%s models/pose_landmarker_lite.task) && \
    echo "Downloaded model size: ${MODEL_SIZE} bytes" && \
    if [ "$MODEL_SIZE" -lt 1000000 ]; then \
        echo "ERROR: model file is suspiciously small (${MODEL_SIZE} bytes) -- download likely failed or was redirected to an error page." && \
        exit 1; \
    fi

COPY src/ src/
COPY .env.example .env

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]