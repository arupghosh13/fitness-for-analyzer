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


if __name__ == "__main__":
    unittest.main()
