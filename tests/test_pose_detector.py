import unittest

from src.pose_estimation.pose_detector import PoseDetectionError, PoseDetector


class TestPoseDetector(unittest.TestCase):
    def test_missing_model_file_raises_clean_error(self):
        """No model file is present in this environment (needs a one-time
        download with internet access) — the class should fail loudly and
        clearly rather than crashing with a raw MediaPipe traceback."""
        with self.assertRaises(PoseDetectionError):
            PoseDetector(model_path="models/does_not_exist.task")

    def test_error_message_mentions_setup_docs(self):
        try:
            PoseDetector(model_path="models/does_not_exist.task")
            self.fail("Expected PoseDetectionError")
        except PoseDetectionError as exc:
            self.assertIn("docs/setup.md", str(exc))


if __name__ == "__main__":
    unittest.main()
