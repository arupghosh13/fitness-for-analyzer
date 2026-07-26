"""Shared contract every exercise implementation follows."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from statistics import median
from typing import Optional

from src.core.rep_counter import RepCounter, RepEvent
from src.pose_estimation.pose_detector import Landmark

# MediaPipe PoseLandmarker landmark indices (verified against installed
# mediapipe==0.10.33 — do not assume these are stable across versions).
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# Landmarks with visibility below this are considered "not confidently
# tracked" (e.g. camera still focusing, limb briefly out of frame/occluded).
# Frames with low visibility on tracked joints are skipped rather than fed
# into the rep counter -- this is what stops spurious early-frame counts.
MIN_LANDMARK_VISIBILITY = 0.5

# Smooth the angle over a short rolling window before it reaches the rep
# counter. A median of the last few frames absorbs single-frame jitter
# spikes without adding noticeable lag (~100ms at 30fps for a window of 3).
ANGLE_SMOOTHING_WINDOW = 3


class LowVisibilityError(Exception):
    """Raised when the landmarks needed for this exercise aren't confidently
    tracked in the current frame (camera warming up, limb occluded, etc.)."""


def landmarks_are_visible(min_visibility: float, *landmarks: Landmark) -> bool:
    """Non-raising visibility check. Used to decide WHICH side (left or
    right) to track this frame, before committing to it -- unlike the
    raising check, this never throws, so it's safe to use for a left/right
    decision without needing a try/except around every attempt."""
    return all(lm.visibility >= min_visibility for lm in landmarks)


class BaseExercise(ABC):
    """Base class for a trackable exercise.

    Subclasses define which joint angle drives rep counting and the
    up/down thresholds for that angle.

    Bilateral tracking: subclasses should check BOTH the left and right
    side landmarks and use whichever side is currently more confidently
    tracked (see Squat/Pushup/BicepCurl for the pattern), rather than
    hardcoding one side. This makes tracking robust to which side of the
    body the camera happens to see clearly. Note this picks one side per
    frame for a single tracked angle -- it does not independently track
    alternating single-arm movements (e.g. alternating single-arm curls)
    as two separate rep counts.
    """

    name: str
    up_threshold: float
    down_threshold: float

    # Per-exercise override point. Defaults to the module-level constant so
    # existing exercises (Squat, Push-up) are completely unaffected unless
    # they explicitly opt into a different threshold.
    min_landmark_visibility: float = MIN_LANDMARK_VISIBILITY

    def __init__(self) -> None:
        self._rep_counter = RepCounter(
            up_threshold=self.up_threshold, down_threshold=self.down_threshold
        )
        self._angle_history: deque[float] = deque(maxlen=ANGLE_SMOOTHING_WINDOW)

    @abstractmethod
    def get_primary_angle(self, landmarks: list[Landmark]) -> float:
        """Return the joint angle (degrees) that drives rep counting.

        Implementations should try the left side via `self._is_visible(...)`,
        fall back to the right side the same way, and raise
        LowVisibilityError only if neither side is confidently tracked --
        process() catches that and skips the frame rather than counting on
        bad data.
        """
        raise NotImplementedError

    def _is_visible(self, *landmarks: Landmark) -> bool:
        """Non-raising visibility check using this exercise's own
        threshold. Use this to decide which side (left/right) to track."""
        return landmarks_are_visible(self.min_landmark_visibility, *landmarks)

    def _check_visibility(self, *landmarks: Landmark) -> None:
        """Raise LowVisibilityError if any given landmark isn't confidently
        tracked (using this exercise's own threshold)."""
        for lm in landmarks:
            if lm.visibility < self.min_landmark_visibility:
                raise LowVisibilityError(
                    f"Landmark visibility {lm.visibility:.2f} is below "
                    f"the {self.min_landmark_visibility} confidence threshold."
                )

    def process(self, landmarks: list[Landmark], now: Optional[float] = None) -> RepEvent:
        """Run one frame through smoothing + rep counting.

        If the relevant landmarks aren't confidently visible this frame,
        the frame is skipped (no crash, no bad data counted) and the last
        known count/state is returned unchanged.
        """
        try:
            raw_angle = self.get_primary_angle(landmarks)
        except LowVisibilityError:
            return self._rep_counter.current_event()

        self._angle_history.append(raw_angle)
        smoothed_angle = median(self._angle_history)
        return self._rep_counter.update(smoothed_angle, now=now)

    @property
    def current_angle(self) -> Optional[float]:
        """Most recent smoothed angle, or None if no frame has been
        successfully processed yet. Safe to call anytime -- never raises,
        unlike calling get_primary_angle() directly (which can raise
        LowVisibilityError on a bad frame). Callers that only need the
        angle for display should use this instead of recomputing it."""
        if not self._angle_history:
            return None
        return median(self._angle_history)

    def reset(self) -> None:
        self._rep_counter.reset()
        self._angle_history.clear()
