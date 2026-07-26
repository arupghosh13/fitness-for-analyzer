"""Streamlit entrypoint — combines all modules into the live application.

Uses streamlit-webrtc (not plain cv2.VideoCapture) so the webcam feed works
once this is deployed to a browser, not just on localhost.
"""
from __future__ import annotations

import sys
import tempfile
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

from src.app.video_batch_processor import process_video
from src.core.form_analyzer import FormAnalyzer
from src.exercises.bicep_curl import BicepCurl
from src.exercises.pushup import Pushup
from src.exercises.squat import Squat
from src.feedback.audio_feedback import AudioFeedback
from src.feedback.visual_feedback import render_overlay
from src.pose_estimation.pose_detector import PoseDetectionError, PoseDetector
from src.utils.config import config
from src.utils.ice_servers import get_ice_servers
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXERCISES = {
    "Squat": Squat,
    "Push-up": Pushup,
    "Bicep Curl": BicepCurl,
}

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

            # Rep counting and form analysis are independent concerns (a
            # rep can complete with bad form, or fail to complete with
            # good form) -- separate try/except blocks so a failure in one
            # never silently discards an already-successful result from
            # the other. We never call get_primary_angle() directly here
            # either, since that CAN raise LowVisibilityError; doing so
            # unguarded is what originally caused the video to freeze.
            try:
                rep_event = self.exercise.process(result.landmarks)
                current_angle = self.exercise.current_angle
                self.last_rep_count = rep_event.count
                if current_angle is not None:
                    self.last_angle = current_angle
            except Exception as exc:
                logger.warning("Rep counting failed: %s", exc)

            try:
                self.last_feedback = self.form_analyzer.analyze(
                    self.exercise.name, result.landmarks
                )
                if self.audio is not None and self.last_feedback:
                    for message in self.last_feedback:
                        self.audio.announce(message)
            except Exception as exc:
                logger.warning("Form analysis failed: %s", exc)

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
    input_mode = st.sidebar.radio("Input source", ["Live Webcam", "Upload Recorded Video"])

    if input_mode == "Live Webcam":
        rotation_label = st.sidebar.selectbox(
            "Camera rotation", list(ROTATION_OPTIONS.keys())
        )
        rotation_value = ROTATION_OPTIONS[rotation_label]
        st.sidebar.markdown(
            "Stand side-on to the camera so your full body is visible in frame. "
            "If your phone camera (e.g. via DroidCam) appears sideways, use the "
            "rotation setting above until the video looks upright."
        )

        # Cached so we don't re-fetch a new Twilio token on every Streamlit
        # rerun (e.g. every time a widget changes) -- one token per session
        # is enough.
        ice_servers = st.cache_data(get_ice_servers)()
        rtc_configuration = RTCConfiguration({"iceServers": ice_servers})

        # Re-keying by exercise_name AND rotation forces a fresh processor
        # whenever either setting changes.
        webrtc_streamer(
            key=f"fitness-corrector-{exercise_name}-{rotation_label}",
            video_processor_factory=lambda: FitnessProcessor(exercise_name, rotation_value),
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
        )
    else:
        st.markdown(
            "Upload a recorded video to analyze it frame by frame. This can be "
            "more accurate than the live webcam: no live streaming compression "
            "artifacts (e.g. from DroidCam's WiFi feed), and no frames are "
            "skipped to keep up with real time."
        )
        uploaded_file = st.file_uploader(
            "Choose a video file", type=["mp4", "mov", "avi", "mkv"]
        )

        if uploaded_file is not None and st.button("Process Video"):
            with tempfile.TemporaryDirectory() as tmpdir:
                input_path = str(Path(tmpdir) / uploaded_file.name)
                output_path = str(Path(tmpdir) / "annotated_output.mp4")

                with open(input_path, "wb") as f:
                    f.write(uploaded_file.read())

                with st.spinner(
                    "Processing video... this can take a while depending on length."
                ):
                    try:
                        final_count = process_video(input_path, output_path, exercise_name)
                    except PoseDetectionError as exc:
                        st.error(f"Pose model error: {exc}")
                        return
                    except (FileNotFoundError, IOError, ValueError) as exc:
                        st.error(f"Could not process video: {exc}")
                        return

                # Read the output into memory BEFORE the temp directory is
                # cleaned up (it's deleted as soon as this `with` block
                # ends), so st.video()/download_button() below always have
                # valid bytes to work with, regardless of when Streamlit
                # actually renders them.
                with open(output_path, "rb") as f:
                    video_bytes = f.read()

            st.success(f"Done! Final rep count: {final_count}")
            st.video(video_bytes)
            st.download_button(
                "Download annotated video",
                data=video_bytes,
                file_name="annotated_output.mp4",
                mime="video/mp4",
            )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "This tool gives general form awareness only — it is not medical "
        "or injury-diagnosis advice."
    )


if __name__ == "__main__":
    main()
