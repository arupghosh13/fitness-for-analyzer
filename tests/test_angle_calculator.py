import unittest

from src.core.angle_calculator import calculate_angle
from src.pose_estimation.pose_detector import Landmark


def lm(x: float, y: float) -> Landmark:
    return Landmark(x=x, y=y, z=0.0, visibility=1.0)


class TestAngleCalculator(unittest.TestCase):
    def test_straight_line_is_180_degrees(self):
        a, b, c = lm(0, 0), lm(1, 0), lm(2, 0)
        self.assertAlmostEqual(calculate_angle(a, b, c), 180.0, places=3)

    def test_right_angle_is_90_degrees(self):
        a, b, c = lm(0, 1), lm(0, 0), lm(1, 0)
        self.assertAlmostEqual(calculate_angle(a, b, c), 90.0, places=3)

    def test_acute_angle_60_degrees(self):
        # Equilateral-triangle-like configuration -> 60 degrees at b
        a, b, c = lm(1, 0), lm(0, 0), lm(0.5, (3 ** 0.5) / 2)
        self.assertAlmostEqual(calculate_angle(a, b, c), 60.0, places=1)

    def test_identical_points_raises_value_error(self):
        a, b, c = lm(0, 0), lm(0, 0), lm(1, 0)
        with self.assertRaises(ValueError):
            calculate_angle(a, b, c)

    def test_angle_is_always_non_negative(self):
        a, b, c = lm(-1, -1), lm(0, 0), lm(1, -1)
        angle = calculate_angle(a, b, c)
        self.assertGreaterEqual(angle, 0.0)
        self.assertLessEqual(angle, 180.0)


if __name__ == "__main__":
    unittest.main()
