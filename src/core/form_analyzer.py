"""Form correctness rules per exercise.

Returns short, user-facing feedback strings. An empty list means "good form" —
callers should treat that as the success case, not as "no data".
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from src.core.angle_calculator import calculate_angle
from src.exercises.base_exercise import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    MIN_LANDMARK_VISIBILITY,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    landmarks_are_visible,
)
from src.pose_estimation.pose_detector import Landmark

# How far the knee may extend past the ankle's x-position, AS A FRACTION OF
# TORSO LENGTH (hip-to-shoulder distance) -- not a fixed normalized-image
# distance. Normalized camera coordinates scale with how far you stand from
# the camera, so a fixed distance threshold means something different at
# every distance; scaling by the person's own torso length makes this
# consistent regardless of distance or camera zoom.
KNEE_OVER_TOE_TOLERANCE_RATIO = 0.20

# Torso must stay within this angle (from vertical) at the bottom of the squat.
MAX_FORWARD_LEAN_DEGREES = 45.0

# Push-up: how far the shoulder-hip-ankle "plank line" angle may deviate
# from a perfectly straight 180 degrees before we flag sag/pike.
MAX_PLANK_DEVIATION_DEGREES = 20.0

# Bicep curl: how far the elbow may drift horizontally from the shoulder,
# as a fraction of torso length (see note above on why this is a ratio,
# not a fixed distance).
MAX_ELBOW_DRIFT_RATIO = 0.35

# Feedback smoothing: a rule must trigger on at least this many of the last
# FEEDBACK_HISTORY_WINDOW frames before it's actually shown. This is what
# stops the message flickering on/off every frame from single-frame jitter
# -- the same idea as the rep counter's angle smoothing, applied to feedback.
FEEDBACK_HISTORY_WINDOW = 5
FEEDBACK_TRIGGER_MAJORITY = 3

# Below this torso length (normalized coords), the person is too small/far
# in frame for distance-ratio checks to be reliable -- skip those rules
# rather than risk a division-by-near-zero blowup.
# Matches BicepCurl's own min_landmark_visibility override (see
# bicep_curl.py for why: wrist/elbow tracking during a curl is naturally
# less confident than the joints other exercises rely on). Using the
# stricter global default here would silence curl form feedback even
# while rep counting (which uses this same lower threshold) keeps working.
CURL_MIN_VISIBILITY = 0.3

MIN_TORSO_LENGTH_FOR_RATIO_CHECKS = 0.05


def _select_side(
    left: tuple[Landmark, ...], right: tuple[Landmark, ...]
) -> Optional[tuple[Landmark, ...]]:
    """Return whichever side's landmarks are ALL confidently tracked
    together (never mixing a left landmark with a right one), preferring
    left. Returns None if neither side is fully visible -- callers should
    skip form checks for that frame rather than compute on unreliable data."""
    if landmarks_are_visible(MIN_LANDMARK_VISIBILITY, *left):
        return left
    if landmarks_are_visible(MIN_LANDMARK_VISIBILITY, *right):
        return right
    return None


class FormAnalyzer:
    def __init__(self) -> None:
        self._history: dict[str, deque[bool]] = {}

    def analyze(self, exercise_name: str, landmarks: list[Landmark]) -> list[str]:
        if exercise_name == "Squat":
            raw_flags = self._check_squat(landmarks)
        elif exercise_name == "Push-up":
            raw_flags = self._check_pushup(landmarks)
        elif exercise_name == "Bicep Curl":
            raw_flags = self._check_bicep_curl(landmarks)
        else:
            raw_flags = {}

        return self._smooth(raw_flags)

    def _smooth(self, raw_flags: dict[str, tuple[bool, str]]) -> list[str]:
        """Only surface a warning if it triggered on a majority of the last
        few frames -- absorbs single-frame jitter without adding much lag."""
        feedback: list[str] = []
        for rule_key, (triggered, message) in raw_flags.items():
            history = self._history.setdefault(
                rule_key, deque(maxlen=FEEDBACK_HISTORY_WINDOW)
            )
            history.append(triggered)
            if sum(history) >= FEEDBACK_TRIGGER_MAJORITY:
                feedback.append(message)
        return feedback

    def reset(self) -> None:
        self._history.clear()

    def _check_squat(self, landmarks: list[Landmark]) -> dict[str, tuple[bool, str]]:
        left = (
            landmarks[LEFT_HIP], landmarks[LEFT_KNEE],
            landmarks[LEFT_ANKLE], landmarks[LEFT_SHOULDER],
        )
        right = (
            landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE],
            landmarks[RIGHT_ANKLE], landmarks[RIGHT_SHOULDER],
        )
        side = _select_side(left, right)
        if side is None:
            return {}  # can't confidently assess form this frame
        hip, knee, ankle, shoulder = side
        torso_length = _distance(hip, shoulder)

        flags: dict[str, tuple[bool, str]] = {}

        if torso_length >= MIN_TORSO_LENGTH_FOR_RATIO_CHECKS:
            knee_over_toe = (knee.x - ankle.x) > (
                KNEE_OVER_TOE_TOLERANCE_RATIO * torso_length
            )
            flags["squat_knee_over_toe"] = (
                knee_over_toe,
                "Keep your knee behind your toes",
            )

        vertical_ref = Landmark(x=hip.x, y=hip.y - 1.0, z=0.0, visibility=1.0)
        torso_angle_from_vertical = calculate_angle(vertical_ref, hip, shoulder)
        flags["squat_forward_lean"] = (
            torso_angle_from_vertical > MAX_FORWARD_LEAN_DEGREES,
            "Keep your back more upright",
        )

        return flags

    def _check_pushup(self, landmarks: list[Landmark]) -> dict[str, tuple[bool, str]]:
        left = (landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP], landmarks[LEFT_ANKLE])
        right = (landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP], landmarks[RIGHT_ANKLE])
        side = _select_side(left, right)
        if side is None:
            return {}
        shoulder, hip, ankle = side

        plank_angle = calculate_angle(shoulder, hip, ankle)
        return {
            "pushup_plank_alignment": (
                abs(180.0 - plank_angle) > MAX_PLANK_DEVIATION_DEGREES,
                "Keep your hips aligned - avoid sagging or piking",
            )
        }

    def _check_bicep_curl(self, landmarks: list[Landmark]) -> dict[str, tuple[bool, str]]:
        left = (landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], landmarks[LEFT_HIP])
        right = (landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW], landmarks[RIGHT_HIP])
        if landmarks_are_visible(CURL_MIN_VISIBILITY, *left):
            side = left
        elif landmarks_are_visible(CURL_MIN_VISIBILITY, *right):
            side = right
        else:
            return {}
        shoulder, elbow, hip = side
        torso_length = _distance(hip, shoulder)

        if torso_length < MIN_TORSO_LENGTH_FOR_RATIO_CHECKS:
            return {}

        elbow_drift = abs(elbow.x - shoulder.x) > (MAX_ELBOW_DRIFT_RATIO * torso_length)
        return {
            "curl_elbow_drift": (elbow_drift, "Keep your elbow close to your body"),
        }


def _distance(a: Landmark, b: Landmark) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
