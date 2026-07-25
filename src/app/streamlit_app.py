"""Streamlit entrypoint — combines all modules into the live application.

Uses streamlit-webrtc (not plain cv2.VideoCapture) so the webcam feed works
once this is deployed to a browser, not just on localhost.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root (the folder containing src/) is on sys.path.
# Streamlit runs this file directly rather than as a package, so Python
# doesn't automatically know where the project root is -- without this,
# "from src...." imports fail with ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from src.core.form_analyzer import FormAnalyzer
from src.exercises.bicep_curl import BicepCurl
from src.exercises.pushup import Pushup
from src.exercises.squat import Squat
from src.feedback.audio_feedback import AudioFeedback
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

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


ROTATION_OPTIONS = {
    "No rotation": None,
    "Rotate 90° clockwise": cv2.ROTATE_90_CLOCKWISE,
    "Rotate 90° counter-clockwise": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "Rotate 180°": cv2.ROTATE_180,
}


class FitnessProcessor(VideoProcessorBase):
    """Runs the full pipeline (pose -> angle -> reps -> form -> overlay) per frame."""

    def __init__(self, exercise_name: str, rotation: int | None = None) -> None:
        self.rotation = rotation
        try:
            self.pose_detector: PoseDetector | None = PoseDetector(
                model_path=config.model_path,
                min_pose_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
        except PoseDetectionError as exc:
            logger.error("Pose model failed to load: %s", exc)
            self.pose_detector = None

        self.exercise = EXERCISES[exercise_name]()
        self.form_analyzer = FormAnalyzer()
        self.audio = (
            AudioFeedback(debounce_seconds=config.audio_debounce_seconds)
            if config.enable_audio
            else None
        )
        self.last_feedback: list[str] = []
        self.last_angle: float = 0.0
        self.last_rep_count: int = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if self.rotation is not None:
            img = cv2.rotate(img, self.rotation)

        if self.pose_detector is None:
            cv2.putText(
                img,
                "Pose model not loaded - see docs/setup.md",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        result = self.pose_detector.detect(img)
        if result is not None:
            img = self.pose_detector.draw_landmarks(img, result)
            try:
                # process() and analyze() are both safe -- they never raise
                # on bad/low-visibility landmarks, they just skip the frame
                # internally. We never call get_primary_angle() directly
                # here, since that CAN raise LowVisibilityError; doing so
                # unguarded is what caused the video to freeze on any frame
                # with uncertain tracking.
                rep_event = self.exercise.process(result.landmarks)
                feedback = self.form_analyzer.analyze(self.exercise.name, result.landmarks)

                current_angle = self.exercise.current_angle
                if current_angle is not None:
                    self.last_angle = current_angle
                self.last_rep_count = rep_event.count
                self.last_feedback = feedback

                if self.audio is not None and feedback:
                    for message in feedback:
                        self.audio.announce(message)
            except Exception as exc:
                # Defense in depth: NOTHING in this per-frame block should
                # ever be able to freeze the video feed. If something
                # unexpected still goes wrong, log it and keep showing the
                # last known-good overlay instead of crashing the stream.
                logger.warning("Skipping frame due to processing error: %s", exc)

        img = render_overlay(
            img,
            rep_count=self.last_rep_count,
            feedback=self.last_feedback,
            angle=self.last_angle,
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def __del__(self) -> None:
        if getattr(self, "audio", None) is not None:
            self.audio.close()


def main() -> None:
    st.set_page_config(page_title="AI Fitness Form Corrector", layout="wide")
    st.title("AI Fitness Form Corrector & Rep Counter")

    st.sidebar.header("Settings")
    exercise_name = st.sidebar.selectbox("Exercise", list(EXERCISES.keys()))
    rotation_label = st.sidebar.selectbox("Camera rotation", list(ROTATION_OPTIONS.keys()))
    rotation_value = ROTATION_OPTIONS[rotation_label]
    st.sidebar.markdown(
        "Stand side-on to the camera so your full body is visible in frame. "
        "If your phone camera (e.g. via DroidCam) appears sideways, use the "
        "rotation setting above until the video looks upright."
    )

    # Re-keying by exercise_name AND rotation forces a fresh processor
    # whenever either setting changes.
    webrtc_streamer(
        key=f"fitness-corrector-{exercise_name}-{rotation_label}",
        video_processor_factory=lambda: FitnessProcessor(exercise_name, rotation_value),
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "This tool gives general form awareness only — it is not medical "
        "or injury-diagnosis advice."
    )


if __name__ == "__main__":
    main()