import unittest

import numpy as np

from src.feedback.visual_feedback import render_overlay


class TestVisualFeedback(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((480, 640, 3), 50, dtype=np.uint8)

    def test_returns_same_shape(self):
        result = render_overlay(self.frame, rep_count=3, feedback=[], angle=170.5)
        self.assertEqual(result.shape, self.frame.shape)

    def test_does_not_mutate_original_frame(self):
        original = self.frame.copy()
        render_overlay(self.frame, rep_count=3, feedback=["test"], angle=90.0)
        np.testing.assert_array_equal(self.frame, original)

    def test_handles_empty_feedback_list(self):
        # Should not raise
        render_overlay(self.frame, rep_count=0, feedback=[], angle=180.0)

    def test_handles_multiple_feedback_messages(self):
        # Should not raise even with several stacked messages
        render_overlay(
            self.frame,
            rep_count=5,
            feedback=["Keep your knee behind your toes", "Keep your back more upright"],
            angle=75.0,
        )


if __name__ == "__main__":
    unittest.main()
