"""Pure geometric utilities — no side effects, fully unit-testable."""
from __future__ import annotations

import math

from src.pose_estimation.pose_detector import Landmark


def calculate_angle(a: Landmark, b: Landmark, c: Landmark) -> float:
    """Return the angle at point b (in degrees, 0-180) formed by a-b-c.

    Uses 2D (x, y) vector math only — z is ignored since MediaPipe's z
    is a rough relative depth estimate, not reliable enough for angle math.
    """
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)

    if mag_ba == 0 or mag_bc == 0:
        raise ValueError("Cannot calculate angle: two landmarks are identical.")

    cosine_angle = dot_product / (mag_ba * mag_bc)
    # Clamp to avoid floating point domain errors just outside [-1, 1]
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    angle_rad = math.acos(cosine_angle)
    return math.degrees(angle_rad)
