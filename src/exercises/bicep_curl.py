"""Bicep Curl: rep counted via elbow angle (shoulder-elbow-wrist).

Naming note: "up_threshold"/"down_threshold" refer to the ANGLE magnitude
(consistent with every other exercise in this codebase), not literal body
direction. For a curl: arm extended = high angle = "up" state; arm flexed
close to the shoulder = low angle = "down" state. A rep completes on the
down -> up transition, i.e. curl up then fully extend back down, same as
every other exercise's abstraction.
"""
from __future__ import annotations

from src.core.angle_calculator import calculate_angle
from src.exercises.base_exercise import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    BaseExercise,
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
    # default, since they weren't experiencing this problem.
    min_landmark_visibility = 0.3

    def get_primary_angle(self, landmarks: list[Landmark]) -> float:
        shoulder = landmarks[LEFT_SHOULDER]
        elbow = landmarks[LEFT_ELBOW]
        wrist = landmarks[LEFT_WRIST]
        self._check_visibility(shoulder, elbow, wrist)
        return calculate_angle(shoulder, elbow, wrist)