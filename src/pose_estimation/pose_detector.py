"""Pose detection wrapper around MediaPipe's Pose Landmarker (Tasks API).

Note: MediaPipe >=0.10.x removed the legacy `mp.solutions.pose` API in favor
of the Tasks API used here, which requires a downloaded .task model file.
See docs/setup.md for the download step (requires internet access, done
once on your local machine).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


class PoseDetectionError(Exception):
    """Raised when the underlying pose model fails to initialize."""


@dataclass(frozen=True)
class Landmark:
    """A single normalized body landmark."""

    x: float
    y: float
    z: float
    visibility: float


@dataclass(frozen=True)
class PoseResult:
    """Full pose detection result for one frame."""

    landmarks: list[Landmark]
    raw_result: vision.PoseLandmarkerResult


class PoseDetector:
    """Wraps MediaPipe's PoseLandmarker (Tasks API) for per-frame detection."""

    def __init__(
        self,
        model_path: str = "models/pose_landmarker_lite.task",
        min_pose_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not Path(model_path).is_file():
            raise PoseDetectionError(
                f"Pose model not found at '{model_path}'. Download it first "
                "(see docs/setup.md) — this requires internet access and "
                "only needs to be done once."
            )
        try:
            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:  # pragma: no cover - init failure path
            raise PoseDetectionError(
                f"Failed to initialize MediaPipe PoseLandmarker: {exc}"
            ) from exc

    def detect(self, frame: np.ndarray) -> Optional[PoseResult]:
        """Run pose detection on a single BGR frame.

        Returns None if no pose is detected.
        """
        if frame is None or frame.size == 0:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        landmarks = [
            Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility or 0.0)
            for lm in result.pose_landmarks[0]
        ]
        return PoseResult(landmarks=landmarks, raw_result=result)

    def draw_landmarks(self, frame: np.ndarray, result: PoseResult) -> np.ndarray:
        """Draw landmark points onto a copy of the frame for debugging."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        for lm in result.landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)
        return annotated

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()


if __name__ == "__main__":
    # Manual local test: opens webcam, shows live landmark overlay.
    # Only run this on a machine with a webcam and the model file present.
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            result = detector.detect(frame)
            if result is not None:
                frame = detector.draw_landmarks(frame, result)
            cv2.imshow("Pose Detector - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
