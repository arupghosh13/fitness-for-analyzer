"""Shared test fixtures/helpers. Plain importable functions (not pytest
fixtures) so they work identically whether tests are run via `pytest` or
plain `python -m unittest`.
"""
from src.pose_estimation.pose_detector import Landmark

TOTAL_LANDMARKS = 29  # covers all indices used anywhere in this project


def make_landmarks(points: dict[int, tuple[float, float]]) -> list[Landmark]:
    """Build a landmark list with only the given indices set meaningfully;
    everything else defaults to the origin.

    Usage: make_landmarks({LEFT_HIP: (0, 0), LEFT_KNEE: (0, 1)})
    """
    landmarks = [
        Landmark(x=0.0, y=0.0, z=0.0, visibility=1.0) for _ in range(TOTAL_LANDMARKS)
    ]
    for index, (x, y) in points.items():
        landmarks[index] = Landmark(x=x, y=y, z=0.0, visibility=1.0)
    return landmarks
