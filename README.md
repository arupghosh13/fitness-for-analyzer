# AI Fitness Form Corrector & Rep Counter

Real-time exercise form feedback and rep counting using a single webcam —
no wearables, no dataset training, just pose estimation and applied geometry.

> This tool gives general form awareness, not medical or injury-diagnosis
> advice.

## What it does
Tracks **Squats, Push-ups, and Bicep Curls** live through your webcam:
counts reps accurately (no double-counting on shaky movement), and flags
specific form issues in real time — visually on-screen and via audio cue.

## Live demo
`<add your deployed URL here after following docs/deployment.md>`

## Architecture
See [`docs/architecture.md`](docs/architecture.md) for the full data-flow
diagram and design rationale. Short version:

```
webcam -> pose landmarks -> joint-angle math -> rep-counting state machine
       -> form-rule checks -> on-screen overlay + audio feedback
```

## Tech stack
Python 3.12 · MediaPipe (Pose Landmarker, Tasks API) · OpenCV · Streamlit +
streamlit-webrtc · Docker · GitHub Actions CI

## Project structure
```
src/
  pose_estimation/   # MediaPipe wrapper
  core/               # angle math, rep-counting state machine, form rules
  exercises/          # Squat, Push-up, Bicep Curl (strategy pattern)
  feedback/           # visual overlay + debounced audio
  app/                # Streamlit entrypoint (wires everything together)
  utils/              # config (.env) and logging
tests/                 # 46 tests, unittest-style (pytest-compatible)
docs/                  # setup, architecture, deployment guides
```

## Setup (local)
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

Then download the pose model — see [`docs/setup.md`](docs/setup.md) (one-time,
needs internet).

```bash
cp .env.example .env
pytest --cov=src tests/            # 46 tests should pass
streamlit run src/app/streamlit_app.py
```

## Docker
```bash
docker build -t fitness-corrector .
docker run -p 8501:8501 fitness-corrector
```

## Deployment
See [`docs/deployment.md`](docs/deployment.md) — Streamlit Community Cloud,
Render, or Hugging Face Spaces all work with this setup.

## Development workflow
This project was built using a **spec-first, AI-implements, human-verifies**
loop: every module was written to an explicit interface contract, tested
immediately (46 tests, all passing), and only integrated once green. See
`docs/architecture.md` for the design decisions this uncovered along the way
(e.g. why the pose detector needs graceful degradation, why the audio engine
can't be allowed to crash the app on unsupported platforms).

## Roadmap
- Auto-calibration per user (first few reps set personal thresholds)
- Rep history + progress dashboard
- More exercises via the existing `BaseExercise` plugin pattern
- Mobile version
