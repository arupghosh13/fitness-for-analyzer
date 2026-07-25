"""App configuration, loaded from environment variables (.env locally,
platform secrets in production). Never hardcode tunables elsewhere."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    model_path: str = os.getenv("POSE_MODEL_PATH", "models/pose_landmarker_lite.task")
    min_detection_confidence: float = float(os.getenv("MIN_DETECTION_CONFIDENCE", "0.5"))
    min_tracking_confidence: float = float(os.getenv("MIN_TRACKING_CONFIDENCE", "0.5"))
    audio_debounce_seconds: float = float(os.getenv("AUDIO_DEBOUNCE_SECONDS", "3.0"))
    enable_audio: bool = os.getenv("ENABLE_AUDIO", "true").lower() == "true"


config = Config()
