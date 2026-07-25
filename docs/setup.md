# Setup: Pose Landmarker Model

This project uses MediaPipe's **Tasks API** (`mediapipe>=0.10`), which requires a
`.task` model file — it is not bundled with the `mediapipe` pip package.

## One-time download (run this locally, needs internet)

```bash
mkdir -p models
curl -o models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

Three model variants exist (trade accuracy for speed):
- `pose_landmarker_lite.task` — fastest, used by default in this project (best for real-time webcam)
- `pose_landmarker_full.task` — balanced
- `pose_landmarker_heavy.task` — most accurate, slowest

Swap the filename in the `curl` command and in `PoseDetector(model_path=...)` if you want a different tradeoff later.

## Why this matters
`PoseDetector.__init__` checks for this file and raises a clear `PoseDetectionError`
if it's missing, instead of letting MediaPipe fail with an opaque internal error.
This is why `tests/test_pose_detector.py` currently only tests the missing-file
error path — this sandbox has no internet access to download the model. Once you
have the file locally, you can additionally test positive detection against a
real photo/webcam frame.
