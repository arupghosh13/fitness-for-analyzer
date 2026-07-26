import unittest

from src.exercises.base_exercise import LEFT_ANKLE, LEFT_HIP, LEFT_KNEE
from src.exercises.squat import Squat
from tests.conftest import make_landmarks

STANDING = make_landmarks({LEFT_HIP: (0, 0), LEFT_KNEE: (0, 1), LEFT_ANKLE: (0, 2)})
SQUATTING = make_landmarks({
    LEFT_HIP: (1, 1),
    LEFT_KNEE: (0, 1),
    LEFT_ANKLE: (0.5, 1 + (3 ** 0.5) / 2),
})


def process_sequence(exercise, positions_and_counts):
    """Feed (landmarks, repeat_count) pairs with realistic timestamps,
    spaced well beyond both the smoothing window and the rep debounce."""
    events = []
    t = 0.0
    for landmarks, repeat in positions_and_counts:
        for _ in range(repeat):
            events.append(exercise.process(landmarks, now=t))
            t += 0.2
    return events


class TestSquat(unittest.TestCase):
    def setUp(self):
        self.squat = Squat()

    def test_standing_angle_is_near_180(self):
        angle = self.squat.get_primary_angle(STANDING)
        self.assertAlmostEqual(angle, 180.0, places=1)

    def test_squatting_angle_is_below_down_threshold(self):
        angle = self.squat.get_primary_angle(SQUATTING)
        self.assertLess(angle, Squat.down_threshold)

    def test_one_full_rep_counts_once(self):
        # Hold each position for 3 frames (matches the smoothing window)
        # so the median angle actually reflects the held position.
        events = process_sequence(
            self.squat, [(STANDING, 3), (SQUATTING, 3), (STANDING, 3)]
        )
        self.assertEqual(events[-1].count, 1)

    def test_three_reps_count_correctly(self):
        one_rep = [(STANDING, 3), (SQUATTING, 3), (STANDING, 3)]
        events = process_sequence(self.squat, one_rep * 3)
        self.assertEqual(events[-1].count, 3)

    def test_reset_clears_count(self):
        process_sequence(self.squat, [(STANDING, 3), (SQUATTING, 3), (STANDING, 3)])
        self.assertEqual(self.squat.process(STANDING, now=100.0).count, 1)
        self.squat.reset()
        events = process_sequence(self.squat, [(STANDING, 3)])
        self.assertEqual(events[-1].count, 0)

    def test_current_angle_is_none_before_any_frame_processed(self):
        self.assertIsNone(self.squat.current_angle)

    def test_current_angle_reflects_last_successfully_processed_frame(self):
        self.squat.process(STANDING, now=0.0)
        self.assertIsNotNone(self.squat.current_angle)
        self.assertAlmostEqual(self.squat.current_angle, 180.0, places=1)

    def test_current_angle_never_raises_on_low_visibility_frame(self):
        # This is the direct regression test for the video-freeze bug:
        # a low-visibility frame must not raise when read via current_angle,
        # even though get_primary_angle() itself would raise directly.
        from src.pose_estimation.pose_detector import Landmark

        low_visibility = [
            Landmark(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(29)
        ]
        low_visibility[LEFT_HIP] = Landmark(x=0, y=0, z=0, visibility=0.1)
        low_visibility[LEFT_KNEE] = Landmark(x=0, y=1, z=0, visibility=0.1)
        low_visibility[LEFT_ANKLE] = Landmark(x=0, y=2, z=0, visibility=0.1)

        self.squat.process(STANDING, now=0.0)  # establish a known-good angle
        self.squat.process(low_visibility, now=0.5)  # must not raise
        # last known-good angle is preserved, not wiped out
        self.assertAlmostEqual(self.squat.current_angle, 180.0, places=1)

    def test_falls_back_to_right_side_when_left_is_not_visible(self):
        # Direct proof of the left/right fallback feature: LEFT landmarks
        # are deliberately low-visibility (as if the camera can't see your
        # left side), but RIGHT landmarks show a clear, valid standing
        # position. The exercise must still track correctly via the right
        # side instead of failing outright.
        from src.exercises.base_exercise import RIGHT_ANKLE, RIGHT_HIP, RIGHT_KNEE
        from src.pose_estimation.pose_detector import Landmark

        right_side_standing = [
            Landmark(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(29)
        ]
        # Left side present in the data but not confidently tracked
        right_side_standing[LEFT_HIP] = Landmark(x=5, y=5, z=0, visibility=0.1)
        right_side_standing[LEFT_KNEE] = Landmark(x=5, y=6, z=0, visibility=0.1)
        right_side_standing[LEFT_ANKLE] = Landmark(x=5, y=7, z=0, visibility=0.1)
        # Right side clearly visible, standing position (collinear -> 180deg)
        right_side_standing[RIGHT_HIP] = Landmark(x=0, y=0, z=0, visibility=1.0)
        right_side_standing[RIGHT_KNEE] = Landmark(x=0, y=1, z=0, visibility=1.0)
        right_side_standing[RIGHT_ANKLE] = Landmark(x=0, y=2, z=0, visibility=1.0)

        angle = self.squat.get_primary_angle(right_side_standing)
        self.assertAlmostEqual(angle, 180.0, places=1)

    def test_raises_when_neither_side_is_visible(self):
        from src.pose_estimation.pose_detector import Landmark
        from src.exercises.base_exercise import LowVisibilityError

        both_sides_hidden = [
            Landmark(x=0.0, y=0.0, z=0.0, visibility=0.1) for _ in range(29)
        ]
        with self.assertRaises(LowVisibilityError):
            self.squat.get_primary_angle(both_sides_hidden)


if __name__ == "__main__":
    unittest.main()
