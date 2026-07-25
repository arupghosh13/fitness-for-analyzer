"""Rep counting state machine.

A rep only counts on a full down -> up transition. Hysteresis (the gap
between up_threshold and down_threshold) absorbs noisy angle readings
near a single threshold so reps aren't double-counted. A minimum time
interval between counted reps is a second, independent safety net against
jitter-induced rapid counting (no human rep is physically faster than
this, so anything quicker is noise, not a real rep).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional

State = Literal["up", "down", "unknown"]

# No real exercise rep completes faster than this. Anything counted quicker
# is landmark jitter, not a genuine rep -- this is the fix for reps racing
# ahead (11, 12, 13, 14 within a second) once movement gets fast.
DEFAULT_MIN_REP_INTERVAL_SECONDS = 0.3


@dataclass(frozen=True)
class RepEvent:
    count: int
    state: State


class RepCounter:
    def __init__(
        self,
        up_threshold: float,
        down_threshold: float,
        min_rep_interval_seconds: float = DEFAULT_MIN_REP_INTERVAL_SECONDS,
    ) -> None:
        if up_threshold <= down_threshold:
            raise ValueError("up_threshold must be greater than down_threshold")
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.min_rep_interval_seconds = min_rep_interval_seconds
        self._state: State = "unknown"
        self._count = 0
        self._last_rep_time: Optional[float] = None

    def update(self, angle: float, now: Optional[float] = None) -> RepEvent:
        """Feed the latest joint angle; returns the current count/state.

        Rep completes on a down -> up transition, but is discarded (state
        still updates, count does not increment) if it happens sooner than
        `min_rep_interval_seconds` after the last counted rep -- this is
        what filters out jitter-driven rapid counting.

        `now` is injectable for deterministic testing; defaults to the
        real clock.
        """
        current_time = now if now is not None else time.monotonic()

        if angle >= self.up_threshold:
            if self._state == "down":
                enough_time_passed = (
                    self._last_rep_time is None
                    or (current_time - self._last_rep_time) >= self.min_rep_interval_seconds
                )
                if enough_time_passed:
                    self._count += 1
                    self._last_rep_time = current_time
            self._state = "up"
        elif angle <= self.down_threshold:
            self._state = "down"
        # else: angle is in the hysteresis gap -> state unchanged (no flicker)

        return RepEvent(count=self._count, state=self._state)

    def current_event(self) -> RepEvent:
        """Return the current count/state without feeding a new angle."""
        return RepEvent(count=self._count, state=self._state)

    def reset(self) -> None:
        self._state = "unknown"
        self._count = 0
        self._last_rep_time = None
