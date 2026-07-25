import unittest

from src.exercises.base_exercise import LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST
from src.exercises.bicep_curl import BicepCurl
from tests.conftest import make_landmarks

ARM_EXTENDED = make_landmarks({
    LEFT_SHOULDER: (0, 0), LEFT_ELBOW: (0, 1), LEFT_WRIST: (0, 2),
})
ARM_FLEXED = make_landmarks({
    LEFT_SHOULDER: (1, 1), LEFT_ELBOW: (0, 1), LEFT_WRIST: (0.866, 1.5),
})


def process_sequence(exercise, positions_and_counts):
    events = []
    t = 0.0
    for landmarks, repeat in positions_and_counts:
        for _ in range(repeat):
            events.append(exercise.process(landmarks, now=t))
            t += 0.2
    return events


class TestBicepCurl(unittest.TestCase):
    def setUp(self):
        self.curl = BicepCurl()

    def test_extended_angle_is_near_180(self):
        angle = self.curl.get_primary_angle(ARM_EXTENDED)
        self.assertAlmostEqual(angle, 180.0, places=1)

    def test_flexed_angle_is_below_down_threshold(self):
        angle = self.curl.get_primary_angle(ARM_FLEXED)
        self.assertLess(angle, BicepCurl.down_threshold)

    def test_one_full_rep_counts_once(self):
        events = process_sequence(
            self.curl, [(ARM_EXTENDED, 3), (ARM_FLEXED, 3), (ARM_EXTENDED, 3)]
        )
        self.assertEqual(events[-1].count, 1)

    def test_three_reps_count_correctly(self):
        one_rep = [(ARM_EXTENDED, 3), (ARM_FLEXED, 3), (ARM_EXTENDED, 3)]
        events = process_sequence(self.curl, one_rep * 3)
        self.assertEqual(events[-1].count, 3)

    def test_low_visibility_landmarks_do_not_count_as_a_rep(self):
        # Simulates the camera still focusing / limb briefly occluded:
        # landmarks are present but visibility is below the confidence
        # threshold. These frames must be skipped, not counted.
        from src.pose_estimation.pose_detector import Landmark

        low_visibility_landmarks = [
            Landmark(x=0.0, y=0.0, z=0.0, visibility=1.0) for _ in range(29)
        ]
        low_visibility_landmarks[LEFT_SHOULDER] = Landmark(x=0, y=0, z=0, visibility=0.1)
        low_visibility_landmarks[LEFT_ELBOW] = Landmark(x=0, y=1, z=0, visibility=0.1)
        low_visibility_landmarks[LEFT_WRIST] = Landmark(x=0, y=2, z=0, visibility=0.1)

        events = process_sequence(self.curl, [(low_visibility_landmarks, 5)])
        self.assertEqual(events[-1].count, 0)
        self.assertEqual(events[-1].state, "unknown")  # state never updated either


if __name__ == "__main__":
    unittest.main()
