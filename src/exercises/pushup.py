"""Push-up: rep counted via elbow angle (shoulder-elbow-wrist)."""
from __future__ import annotations

from src.core.angle_calculator import calculate_angle
from src.exercises.base_exercise import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    BaseExercise,
)
from src.pose_estimation.pose_detector import Landmark


class Pushup(BaseExercise):
    name = "Push-up"
    # Arms extended (top of push-up) ~160-180deg -> "up".
    # Arms bent (bottom of push-up) ~<90deg -> "down".
    up_threshold = 160.0
    down_threshold = 90.0

    def get_primary_angle(self, landmarks: list[Landmark]) -> float:
        shoulder = landmarks[LEFT_SHOULDER]
        elbow = landmarks[LEFT_ELBOW]
        wrist = landmarks[LEFT_WRIST]
        self._check_visibility(shoulder, elbow, wrist)
        return calculate_angle(shoulder, elbow, wrist)
