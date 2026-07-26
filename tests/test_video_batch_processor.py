import os
import tempfile
import unittest

import cv2
import numpy as np

from src.app.video_batch_processor import process_video
from src.exercises.base_exercise import LEFT_ANKLE, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER
from src.exercises.squat import Squat
from src.pose_estimation.pose_detector import PoseResult
from tests.conftest import make_landmarks

STANDING = make_landmarks({
    LEFT_HIP: (0, 0), LEFT_KNEE: (0, 1), LEFT_ANKLE: (0, 2), LEFT_SHOULDER: (0, -1),
})
SQUATTING = make_landmarks({
    LEFT_HIP: (1, 1), LEFT_KNEE: (0, 1), LEFT_ANKLE: (0.5, 1 + (3 ** 0.5) / 2),
    LEFT_SHOULDER: (1, 0),
})


class FakePoseDetector:
    """Returns a pre-scripted sequence of landmarks, one per detect() call,
    instead of running real MediaPipe inference -- lets the whole pipeline
    be tested without a real model file or camera."""

    def __init__(self, landmark_sequence: list[list]):
        self._sequence = landmark_sequence
        self._index = 0

    def detect(self, frame):
        if self._index >= len(self._sequence):
            return None
        landmarks = self._sequence[self._index]
        self._index += 1
        return PoseResult(landmarks=landmarks, raw_result=None)

    def draw_landmarks(self, frame, result):
        return frame  # no-op for the test; real drawing is tested elsewhere

    def close(self):
        pass


def make_synthetic_video(path: str, num_frames: int, fps: float = 10.0, size=(64, 64)) -> None:
    """Writes a tiny solid-color test video -- content doesn't matter since
    the fake pose detector supplies landmarks independently of pixel data."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    for _ in range(num_frames):
        writer.write(np.full((size[1], size[0], 3), 100, dtype=np.uint8))
    writer.release()


class TestVideoBatchProcessor(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.input_path = os.path.join(self.tmpdir, "input.mp4")
        self.output_path = os.path.join(self.tmpdir, "output.mp4")

    def test_processes_a_full_rep_and_writes_output_video(self):
        # 3 frames held per position, matching the smoothing window, same
        # pattern proven in test_squat.py.
        sequence = [STANDING] * 3 + [SQUATTING] * 3 + [STANDING] * 3
        make_synthetic_video(self.input_path, num_frames=len(sequence))

        final_count = process_video(
            input_path=self.input_path,
            output_path=self.output_path,
            exercise_name="Squat",
            pose_detector=FakePoseDetector(sequence),
            exercise=Squat(),
        )

        self.assertEqual(final_count, 1)
        self.assertTrue(os.path.exists(self.output_path))
        self.assertGreater(os.path.getsize(self.output_path), 0)

    def test_output_video_has_same_frame_count_as_input(self):
        sequence = [STANDING] * 5
        make_synthetic_video(self.input_path, num_frames=5)

        process_video(
            input_path=self.input_path,
            output_path=self.output_path,
            exercise_name="Squat",
            pose_detector=FakePoseDetector(sequence),
            exercise=Squat(),
        )

        input_cap = cv2.VideoCapture(self.input_path)
        output_cap = cv2.VideoCapture(self.output_path)
        input_frame_count = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_frame_count = int(output_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_cap.release()
        output_cap.release()

        self.assertEqual(input_frame_count, output_frame_count)

    def test_missing_input_file_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            process_video(
                input_path="/nonexistent/path/video.mp4",
                output_path=self.output_path,
                exercise_name="Squat",
                pose_detector=FakePoseDetector([]),
                exercise=Squat(),
            )

    def test_unknown_exercise_name_raises_clear_error(self):
        make_synthetic_video(self.input_path, num_frames=2)
        with self.assertRaises(ValueError):
            process_video(
                input_path=self.input_path,
                output_path=self.output_path,
                exercise_name="Not A Real Exercise",
                pose_detector=FakePoseDetector([]),
            )

    def test_frames_with_no_detected_pose_do_not_crash(self):
        # FakePoseDetector returns None once its sequence is exhausted,
        # simulating "no person detected" frames -- must not crash, and
        # should just keep showing the last known overlay.
        sequence = [STANDING] * 2  # video has 5 frames, sequence only has 2
        make_synthetic_video(self.input_path, num_frames=5)

        final_count = process_video(
            input_path=self.input_path,
            output_path=self.output_path,
            exercise_name="Squat",
            pose_detector=FakePoseDetector(sequence),
            exercise=Squat(),
        )
        self.assertEqual(final_count, 0)  # no rep completed, but no crash either

    def test_form_analysis_failure_does_not_discard_a_successful_rep_count(self):
        # Direct regression test: rep counting and form analysis must fail
        # independently. A landmark set with hip == shoulder breaks the
        # form analyzer's lean-angle math specifically, but the exercise's
        # own rep-counting logic (which doesn't need the shoulder at all)
        # must still succeed and be preserved.
        from src.pose_estimation.pose_detector import Landmark

        broken_shoulder_standing = make_landmarks({
            LEFT_HIP: (0, 0), LEFT_KNEE: (0, 1), LEFT_ANKLE: (0, 2),
            LEFT_SHOULDER: (0, 0),  # identical to hip -> breaks form_analyzer only
        })
        broken_shoulder_squatting = make_landmarks({
            LEFT_HIP: (1, 1), LEFT_KNEE: (0, 1), LEFT_ANKLE: (0.5, 1 + (3 ** 0.5) / 2),
            LEFT_SHOULDER: (1, 1),  # identical to hip -> breaks form_analyzer only
        })

        sequence = (
            [broken_shoulder_standing] * 3
            + [broken_shoulder_squatting] * 3
            + [broken_shoulder_standing] * 3
        )
        make_synthetic_video(self.input_path, num_frames=len(sequence))

        final_count = process_video(
            input_path=self.input_path,
            output_path=self.output_path,
            exercise_name="Squat",
            pose_detector=FakePoseDetector(sequence),
            exercise=Squat(),
        )
        # Rep counting must succeed despite form analysis failing on every
        # single frame.
        self.assertEqual(final_count, 1)


if __name__ == "__main__":
    unittest.main()
