"""Bicep Curl: rep counted via elbow angle (shoulder-elbow-wrist).

Naming note: "up_threshold"/"down_threshold" refer to the ANGLE magnitude
(consistent with every other exercise in this codebase), not literal body
direction. For a curl: arm extended = high angle = "up" state; arm flexed
close to the shoulder = low angle = "down" state. A rep completes on the
down -> up transition, i.e. curl up then fully extend back down, same as
every other exercise's abstraction.

Tries the left arm first; if it isn't confidently tracked this frame,
falls back to the right arm. See squat.py's docstring for why. Note this
tracks ONE side's angle at a time (whichever is currently better tracked)
-- it does not independently count alternating single-arm curls as two
separate totals.
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


class BicepCurl(BaseExercise):
    name = "Bicep Curl"
    up_threshold = 160.0    # arm extended
    down_threshold = 50.0   # arm fully flexed

    # The wrist is harder for MediaPipe to track confidently during a curl
    # than the joints Squat/Push-up rely on (hip/knee/ankle, or a mostly-
    # static shoulder/elbow/wrist in a plank): at full flexion the wrist
    # moves close to the torso and is prone to self-occlusion and motion
    # blur, especially over a compressed feed (e.g. a phone camera via
    # DroidCam). The default 0.5 threshold caused most curl frames to be
    # discarded as "not confidently tracked," which froze the displayed
    # angle and stopped reps from ever being counted. Lowering it here
    # only affects Bicep Curl -- Squat and Push-up keep the stricter
    # default, since they weren't experiencing this problem. This applies
    # to both the left AND right side checks below.
    min_landmark_visibility = 0.3

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
