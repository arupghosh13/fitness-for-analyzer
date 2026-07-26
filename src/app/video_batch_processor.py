"""Batch-process a pre-recorded video file through the full fitness-form
pipeline, producing an annotated output video.

Why this can be more accurate than the live webcam demo:
- No live WiFi-streaming compression/frame-drop artifacts (e.g. from
  DroidCam) -- a locally recorded file is read directly, frame by frame.
- No real-time performance pressure -- every frame gets fully processed,
  none are skipped to keep up with a live feed.
- You can review/redo the recording for good lighting and camera
  positioning before processing, rather than reacting live.

What this does NOT fix: monocular (single-camera) 2D pose estimation
cannot recover true depth. If a joint moves mostly toward/away from the
camera rather than across it, the 2D projection will always compress that
motion -- this is a physical limitation of single-camera pose estimation,
not a bug, and applies identically to recorded or live video from the
same camera angle. Camera positioning (side-on, full body in frame)
matters far more than live-vs-recorded for this specific issue.

Usage:
    python -m src.app.video_batch_processor --input squat.mp4 \
        --output squat_annotated.mp4 --exercise Squat
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2

from src.core.form_analyzer import FormAnalyzer
from src.exercises.base_exercise import BaseExercise
from src.exercises.bicep_curl import BicepCurl
from src.exercises.pushup import Pushup
from src.exercises.squat import Squat
from src.feedback.visual_feedback import render_overlay
from src.pose_estimation.pose_detector import PoseDetectionError, PoseDetector
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXERCISES = {
    "Squat": Squat,
    "Push-up": Pushup,
    "Bicep Curl": BicepCurl,
}


def process_video(
    input_path: str,
    output_path: str,
    exercise_name: str,
    pose_detector: Optional[PoseDetector] = None,
    exercise: Optional[BaseExercise] = None,
) -> int:
    """Process `input_path` frame by frame, writing an annotated video to
    `output_path`. Returns the final rep count.

    `pose_detector` and `exercise` are injectable (for testing without a
    real model file, and for reusing an exercise instance) -- default to
    constructing real ones from `exercise_name`.
    """
    if exercise_name not in EXERCISES:
        raise ValueError(
            f"Unknown exercise '{exercise_name}'. Choose from: {list(EXERCISES)}"
        )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Could not open output video for writing: {output_path}")

    owns_pose_detector = pose_detector is None
    if pose_detector is None:
        pose_detector = PoseDetector(
            model_path=config.model_path,
            min_pose_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

    if exercise is None:
        exercise = EXERCISES[exercise_name]()

    form_analyzer = FormAnalyzer()

    last_feedback: list[str] = []
    last_angle: float = 0.0
    last_rep_count = 0
    frame_index = 0
    frame_time_seconds = 1.0 / fps

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = pose_detector.detect(frame)
            if result is not None:
                frame = pose_detector.draw_landmarks(frame, result)

                # Rep counting and form analysis are independent concerns
                # (a rep can complete with bad form, or fail to complete
                # with good form) -- they get separate try/except blocks so
                # a failure in one never discards an already-successful
                # result from the other.
                try:
                    rep_event = exercise.process(
                        result.landmarks, now=frame_index * frame_time_seconds
                    )
                    current_angle = exercise.current_angle
                    last_rep_count = rep_event.count
                    if current_angle is not None:
                        last_angle = current_angle
                except Exception as exc:
                    logger.warning(
                        "Rep counting failed on frame %d: %s", frame_index, exc
                    )

                try:
                    last_feedback = form_analyzer.analyze(
                        exercise.name, result.landmarks
                    )
                except Exception as exc:
                    logger.warning(
                        "Form analysis failed on frame %d: %s", frame_index, exc
                    )

            annotated = render_overlay(
                frame, rep_count=last_rep_count, feedback=last_feedback, angle=last_angle
            )
            writer.write(annotated)
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        if owns_pose_detector:
            pose_detector.close()

    logger.info(
        "Processed %d frames. Final rep count: %d", frame_index, last_rep_count
    )
    return last_rep_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-process a recorded video through the fitness form pipeline."
    )
    parser.add_argument("--input", required=True, help="Path to the input video file")
    parser.add_argument(
        "--output", required=True, help="Path to write the annotated output video"
    )
    parser.add_argument(
        "--exercise",
        required=True,
        choices=list(EXERCISES.keys()),
        help="Which exercise to track",
    )
    args = parser.parse_args()

    try:
        final_count = process_video(args.input, args.output, args.exercise)
    except PoseDetectionError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(f"Done. Final rep count: {final_count}. Output saved to {args.output}")


if __name__ == "__main__":
    main()
