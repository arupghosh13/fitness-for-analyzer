import unittest

from src.exercises.base_exercise import LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST
from src.exercises.pushup import Pushup
from tests.conftest import make_landmarks

ARMS_EXTENDED = make_landmarks({
    LEFT_SHOULDER: (0, 0), LEFT_ELBOW: (0, 1), LEFT_WRIST: (0, 2),
})
ARMS_BENT = make_landmarks({
    LEFT_SHOULDER: (1, 1), LEFT_ELBOW: (0, 1), LEFT_WRIST: (0.5, 1 + (3 ** 0.5) / 2),
})


def process_sequence(exercise, positions_and_counts):
    events = []
    t = 0.0
    for landmarks, repeat in positions_and_counts:
        for _ in range(repeat):
            events.append(exercise.process(landmarks, now=t))
            t += 0.2
    return events


class TestPushup(unittest.TestCase):
    def setUp(self):
        self.pushup = Pushup()

    def test_extended_angle_is_near_180(self):
        angle = self.pushup.get_primary_angle(ARMS_EXTENDED)
        self.assertAlmostEqual(angle, 180.0, places=1)

    def test_bent_angle_is_below_down_threshold(self):
        angle = self.pushup.get_primary_angle(ARMS_BENT)
        self.assertLess(angle, Pushup.down_threshold)

    def test_one_full_rep_counts_once(self):
        events = process_sequence(
            self.pushup, [(ARMS_EXTENDED, 3), (ARMS_BENT, 3), (ARMS_EXTENDED, 3)]
        )
        self.assertEqual(events[-1].count, 1)

    def test_three_reps_count_correctly(self):
        one_rep = [(ARMS_EXTENDED, 3), (ARMS_BENT, 3), (ARMS_EXTENDED, 3)]
        events = process_sequence(self.pushup, one_rep * 3)
        self.assertEqual(events[-1].count, 3)


if __name__ == "__main__":
    unittest.main()
