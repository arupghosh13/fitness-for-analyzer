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
            Landmark(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(29)
        ]
        low_visibility_landmarks[LEFT_SHOULDER] = Landmark(x=0, y=0, z=0, visibility=0.1)
        low_visibility_landmarks[LEFT_ELBOW] = Landmark(x=0, y=1, z=0, visibility=0.1)
        low_visibility_landmarks[LEFT_WRIST] = Landmark(x=0, y=2, z=0, visibility=0.1)

        events = process_sequence(self.curl, [(low_visibility_landmarks, 5)])
        self.assertEqual(events[-1].count, 0)
        self.assertEqual(events[-1].state, "unknown")  # state never updated either

    def test_moderately_low_visibility_still_counts_for_bicep_curl(self):
        # This is the direct regression test for the "angle frozen, reps
        # don't count" bug: 0.4 visibility would have been REJECTED under
        # the old global 0.5 threshold (which is still correct for Squat/
        # Push-up), but Bicep Curl now uses a lower 0.3 threshold because
        # wrist tracking is naturally less confident during a curl.
        from src.pose_estimation.pose_detector import Landmark

        def landmarks_at(shoulder_xy, elbow_xy, wrist_xy, visibility):
            lms = [Landmark(x=0, y=0, z=0, visibility=1.0) for _ in range(29)]
            lms[LEFT_SHOULDER] = Landmark(x=shoulder_xy[0], y=shoulder_xy[1], z=0, visibility=visibility)
            lms[LEFT_ELBOW] = Landmark(x=elbow_xy[0], y=elbow_xy[1], z=0, visibility=visibility)
            lms[LEFT_WRIST] = Landmark(x=wrist_xy[0], y=wrist_xy[1], z=0, visibility=visibility)
            return lms

        extended = landmarks_at((0, 0), (0, 1), (0, 2), visibility=0.4)
        flexed = landmarks_at((1, 1), (0, 1), (0.866, 1.5), visibility=0.4)

        events = process_sequence(
            self.curl, [(extended, 3), (flexed, 3), (extended, 3)]
        )
        # The angle must have actually updated (not frozen) and the rep
        # must have been counted, despite visibility being below the old
        # global threshold.
        self.assertIsNotNone(self.curl.current_angle)
        self.assertEqual(events[-1].count, 1)

    def test_squat_visibility_threshold_is_unaffected_by_curl_override(self):
        # Confirms the per-exercise override doesn't leak: Squat must still
        # use the stricter default 0.5 threshold, unchanged.
        from src.exercises.base_exercise import MIN_LANDMARK_VISIBILITY
        from src.exercises.squat import Squat

        squat = Squat()
        self.assertEqual(squat.min_landmark_visibility, MIN_LANDMARK_VISIBILITY)
        self.assertEqual(squat.min_landmark_visibility, 0.5)
        self.assertEqual(BicepCurl.min_landmark_visibility, 0.3)


if __name__ == "__main__":
    unittest.main()
