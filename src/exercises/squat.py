"""Squat: rep counted via knee angle (hip-knee-ankle)."""
from __future__ import annotations

from src.core.angle_calculator import calculate_angle
from src.exercises.base_exercise import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    BaseExercise,
)
from src.pose_estimation.pose_detector import Landmark


class Squat(BaseExercise):
    name = "Squat"
    # Standing (leg straight) ~170-180deg -> "up". Full squat ~<90deg -> "down".
    up_threshold = 160.0
    down_threshold = 90.0

    def get_primary_angle(self, landmarks: list[Landmark]) -> float:
        hip = landmarks[LEFT_HIP]
        knee = landmarks[LEFT_KNEE]
        ankle = landmarks[LEFT_ANKLE]
        self._check_visibility(hip, knee, ankle)
        return calculate_angle(hip, knee, ankle)
