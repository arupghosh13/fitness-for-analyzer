import unittest

from src.core.form_analyzer import FormAnalyzer
from src.exercises.base_exercise import (
    LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER,
)
from tests.conftest import make_landmarks


def repeat_analyze(analyzer, exercise_name, landmarks, times=5):
    """Feed the same landmarks several times so the smoothing window fills
    (majority-vote feedback needs several consistent frames to trigger)."""
    feedback = []
    for _ in range(times):
        feedback = analyzer.analyze(exercise_name, landmarks)
    return feedback


class TestFormAnalyzerSquat(unittest.TestCase):
    def setUp(self):
        self.analyzer = FormAnalyzer()

    def test_good_form_returns_no_feedback(self):
        landmarks = make_landmarks({
            LEFT_HIP: (0, 2), LEFT_KNEE: (0, 3), LEFT_ANKLE: (0, 4), LEFT_SHOULDER: (0, 1),
        })
        feedback = repeat_analyze(self.analyzer, "Squat", landmarks)
        self.assertEqual(feedback, [])

    def test_knee_over_toe_is_flagged(self):
        # torso_length (hip-shoulder) = 1.0 here; knee.x - ankle.x = 0.3,
        # well beyond the 0.20 * torso_length tolerance.
        landmarks = make_landmarks({
            LEFT_HIP: (0, 2), LEFT_KNEE: (0.3, 3), LEFT_ANKLE: (0, 4), LEFT_SHOULDER: (0, 1),
        })
        feedback = repeat_analyze(self.analyzer, "Squat", landmarks)
        self.assertIn("Keep your knee behind your toes", feedback)

    def test_excessive_forward_lean_is_flagged(self):
        landmarks = make_landmarks({
            LEFT_HIP: (0, 2), LEFT_KNEE: (0, 3), LEFT_ANKLE: (0, 4), LEFT_SHOULDER: (1, 2),
        })
        feedback = repeat_analyze(self.analyzer, "Squat", landmarks)
        self.assertIn("Keep your back more upright", feedback)

    def test_upright_torso_is_not_flagged(self):
        landmarks = make_landmarks({
            LEFT_HIP: (0, 2), LEFT_KNEE: (0, 3), LEFT_ANKLE: (0, 4), LEFT_SHOULDER: (0, 0),
        })
        feedback = repeat_analyze(self.analyzer, "Squat", landmarks)
        self.assertNotIn("Keep your back more upright", feedback)

    def test_unknown_exercise_returns_empty_list(self):
        landmarks = make_landmarks({
            LEFT_HIP: (0, 2), LEFT_KNEE: (0, 3), LEFT_ANKLE: (0, 4), LEFT_SHOULDER: (0, 1),
        })
        feedback = repeat_analyze(self.analyzer, "Not A Real Exercise", landmarks)
        self.assertEqual(feedback, [])

    def test_single_noisy_frame_does_not_trigger_feedback(self):
        # A single bad-form frame among otherwise-good frames should NOT
        # flicker the warning on -- this is the flicker-fix regression test.
        good = make_landmarks({
            LEFT_HIP: (0, 2), LEFT_KNEE: (0, 3), LEFT_ANKLE: (0, 4), LEFT_SHOULDER: (0, 1),
        })
        bad = make_landmarks({
            LEFT_HIP: (0, 2), LEFT_KNEE: (0.3, 3), LEFT_ANKLE: (0, 4), LEFT_SHOULDER: (0, 1),
        })
        self.analyzer.analyze("Squat", good)
        self.analyzer.analyze("Squat", good)
        feedback = self.analyzer.analyze("Squat", bad)  # 1 bad frame only
        self.assertNotIn("Keep your knee behind your toes", feedback)


class TestFormAnalyzerPushup(unittest.TestCase):
    def setUp(self):
        self.analyzer = FormAnalyzer()

    def test_straight_plank_is_good_form(self):
        landmarks = make_landmarks({
            LEFT_SHOULDER: (0, 0), LEFT_HIP: (1, 0), LEFT_ANKLE: (2, 0),
        })
        feedback = repeat_analyze(self.analyzer, "Push-up", landmarks)
        self.assertEqual(feedback, [])

    def test_sagging_hips_is_flagged(self):
        landmarks = make_landmarks({
            LEFT_SHOULDER: (0, 0), LEFT_HIP: (1, 0.5), LEFT_ANKLE: (2, 0),
        })
        feedback = repeat_analyze(self.analyzer, "Push-up", landmarks)
        self.assertIn("Keep your hips aligned - avoid sagging or piking", feedback)


class TestFormAnalyzerBicepCurl(unittest.TestCase):
    def setUp(self):
        self.analyzer = FormAnalyzer()

    def test_elbow_close_to_body_is_good_form(self):
        # torso_length (hip-shoulder) = 1.0; elbow.x - shoulder.x = 0.02,
        # nowhere near the 0.35 * torso_length drift threshold.
        landmarks = make_landmarks({
            LEFT_SHOULDER: (0, 0), LEFT_ELBOW: (0.02, 1), LEFT_HIP: (0, 1),
        })
        feedback = repeat_analyze(self.analyzer, "Bicep Curl", landmarks)
        self.assertEqual(feedback, [])

    def test_elbow_drift_is_flagged(self):
        # elbow.x - shoulder.x = 0.6, well beyond 0.35 * torso_length (1.0)
        landmarks = make_landmarks({
            LEFT_SHOULDER: (0, 0), LEFT_ELBOW: (0.6, 1), LEFT_HIP: (0, 1),
        })
        feedback = repeat_analyze(self.analyzer, "Bicep Curl", landmarks)
        self.assertIn("Keep your elbow close to your body", feedback)

    def test_close_elbow_at_different_camera_distance_is_still_good_form(self):
        # Same PROPORTIONS as the good-form case above, but everything
        # scaled down 4x (simulating standing farther from the camera).
        # A fixed-distance threshold would behave inconsistently here;
        # a ratio-based threshold should not.
        landmarks = make_landmarks({
            LEFT_SHOULDER: (0, 0), LEFT_ELBOW: (0.005, 0.25), LEFT_HIP: (0, 0.25),
        })
        feedback = repeat_analyze(self.analyzer, "Bicep Curl", landmarks)
        self.assertEqual(feedback, [])


if __name__ == "__main__":
    unittest.main()
