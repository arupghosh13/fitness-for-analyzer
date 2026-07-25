"""Draws rep count, current angle, and form feedback onto a video frame."""
from __future__ import annotations

import cv2
import numpy as np

GREEN = (0, 200, 0)
RED = (0, 0, 220)
WHITE = (255, 255, 255)
BLACK_BG = (0, 0, 0)


def render_overlay(
    frame: np.ndarray,
    rep_count: int,
    feedback: list[str],
    angle: float,
) -> np.ndarray:
    """Return a copy of `frame` with rep count, angle, and feedback drawn on.

    Rep count is large and top-left. Feedback is color-coded: green when
    the list is empty (good form), red per warning message, stacked in the
    top-right so it never overlaps the tracked body in the frame center.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Rep count: large, top-left, with a dark backing box for legibility
    cv2.rectangle(annotated, (10, 10), (220, 90), BLACK_BG, thickness=-1)
    cv2.putText(
        annotated, f"Reps: {rep_count}", (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 1.3, WHITE, 3, cv2.LINE_AA,
    )

    # Current angle: small, below rep count
    cv2.putText(
        annotated, f"Angle: {angle:.0f} deg", (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA,
    )

    # Feedback: top-right, color-coded, one line per message
    if not feedback:
        cv2.putText(
            annotated, "Good form", (w - 260, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2, cv2.LINE_AA,
        )
    else:
        for i, message in enumerate(feedback):
            y = 40 + i * 35
            cv2.putText(
                annotated, message, (w - 400, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA,
            )

    return annotated
