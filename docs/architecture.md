# Architecture

## Data flow

```
Webcam (browser, via streamlit-webrtc)
        |
        v
PoseDetector.detect(frame)          [src/pose_estimation/pose_detector.py]
        | -> PoseResult(landmarks: list[Landmark])
        v
Exercise.process(landmarks)         [src/exercises/{squat,pushup,bicep_curl}.py]
        | -> uses angle_calculator.calculate_angle()  [src/core/angle_calculator.py]
        | -> uses RepCounter (state machine)           [src/core/rep_counter.py]
        | -> RepEvent(count, state)
        v
FormAnalyzer.analyze(exercise_name, landmarks)  [src/core/form_analyzer.py]
        | -> list[str] feedback messages (empty = good form)
        v
render_overlay(frame, rep_count, feedback, angle)   [src/feedback/visual_feedback.py]
        | -> annotated frame drawn back into the video stream
        v
AudioFeedback.announce(message)     [src/feedback/audio_feedback.py]
        | -> non-blocking, debounced, degrades gracefully if no TTS engine
        v
Streamlit UI (sidebar: exercise selector, live video, feedback)  [src/app/streamlit_app.py]
```

## Why these design decisions

- **Strategy pattern for exercises** (`BaseExercise`): adding a 4th exercise means
  writing one new class with a `get_primary_angle()` method — no branching logic
  added anywhere else.
- **Stateless `FormAnalyzer`**: form rules are pure per-frame checks (no history
  needed), keeping them trivially unit-testable with synthetic landmark data.
- **`RepCounter` hysteresis**: a single threshold would double-count on noisy
  angle readings near the boundary; two thresholds with a gap between them
  absorb that noise.
- **Pose Landmarker (Tasks API), not legacy `mp.solutions.pose`**: MediaPipe
  removed the legacy API in recent releases — the Tasks API is the only one
  that works with `mediapipe>=0.10`.
- **`streamlit-webrtc`, not `cv2.VideoCapture`**: the app is meant to run at a
  public URL. Browsers can't hand a raw camera device to a Python backend the
  way a local OS can — WebRTC is the actual browser-to-server video path.
- **Graceful degradation everywhere a dependency might be missing**
  (pose model file, TTS engine): the app should never crash on startup just
  because an optional feature's dependency isn't present; it should fall back
  and keep the core experience working.

## Landmark indices used (MediaPipe PoseLandmarker, verified against `mediapipe==0.10.33`)

| Landmark | Index |
|---|---|
| LEFT_SHOULDER | 11 |
| LEFT_ELBOW | 13 |
| LEFT_WRIST | 15 |
| LEFT_HIP | 23 |
| LEFT_KNEE | 25 |
| LEFT_ANKLE | 27 |

Defined once in `src/exercises/base_exercise.py` and imported everywhere else —
never hardcoded as magic numbers elsewhere in the codebase.
