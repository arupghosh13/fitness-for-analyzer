"""Push-up: rep counted via elbow angle (shoulder-elbow-wrist).

Tries the left arm first; if it isn't confidently tracked this frame,
falls back to the right arm. See squat.py's docstring for why.
"""
from __future__ import annotations

from src.core.angle_calculator import calculate_angle
from src.exercises.base_exercise import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    BaseExercise,
    LowVisibilityError,
)
from src.pose_estimation.pose_detector import Landmark


class Pushup(BaseExercise):
    name = "Push-up"
    # Arms extended (top of push-up) ~160-180deg -> "up".
    # Arms bent (bottom of push-up) ~<90deg -> "down".
    up_threshold = 160.0
    down_threshold = 90.0

    def get_primary_angle(self, landmarks: list[Landmark]) -> float:
        left = (landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST])
        if self._is_visible(*left):
            return calculate_angle(*left)

        right = (landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST])
        if self._is_visible(*right):
            return calculate_angle(*right)

        raise LowVisibilityError(
            "Neither left nor right shoulder/elbow/wrist are confidently tracked this frame."
        )
