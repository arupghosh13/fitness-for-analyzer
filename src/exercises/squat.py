"""Squat: rep counted via knee angle (hip-knee-ankle).

Tries the left leg first; if it isn't confidently tracked this frame,
falls back to the right leg. This makes tracking robust to which side of
the body the camera happens to see clearly (e.g. if you're turned the
other way, or the camera is positioned to see your right side better).
"""
from __future__ import annotations

from src.core.angle_calculator import calculate_angle
from src.exercises.base_exercise import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    BaseExercise,
    LowVisibilityError,
)
from src.pose_estimation.pose_detector import Landmark


class Squat(BaseExercise):
    name = "Squat"
    # Standing (leg straight) ~170-180deg -> "up". Full squat ~<90deg -> "down".
    up_threshold = 160.0
    down_threshold = 90.0

    def get_primary_angle(self, landmarks: list[Landmark]) -> float:
        left = (landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
        if self._is_visible(*left):
            return calculate_angle(*left)

        right = (landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
        if self._is_visible(*right):
            return calculate_angle(*right)

        raise LowVisibilityError(
            "Neither left nor right hip/knee/ankle are confidently tracked this frame."
        )
