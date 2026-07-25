# Deployment

This app needs a browser-accessible webcam feed, which `streamlit-webrtc`
handles — but it needs a public URL to actually grant camera permission in
a browser (not just `localhost`). Three free options, all tested to work
with `streamlit-webrtc`:

## Option A — Streamlit Community Cloud (simplest)
1. Push this repo to GitHub.
2. Go to share.streamlit.io, connect the repo, set main file to
   `src/app/streamlit_app.py`.
3. Add any `.env` values as "Secrets" in the app settings (not committed).
4. Note: Community Cloud installs from `requirements.txt` directly — the
   model file won't be downloaded automatically. Add a `packages.txt` with
   `curl` and a small startup step, or switch to Option B/C, which use the
   Dockerfile (which already handles the model download at build time).

## Option B — Render
1. Push to GitHub, create a new Web Service on Render, point it at this repo.
2. Render auto-detects the `Dockerfile` — no extra config needed, since the
   model download is already baked into the build step.
3. Set the port to `8501` in Render's settings.

## Option C — Hugging Face Spaces
1. Create a new Space, SDK = Docker.
2. Push this repo's contents to the Space's git remote.
3. HF Spaces builds the `Dockerfile` automatically — same as Render.

## After deploying
Open the public URL from a different device/network than the one you built
on, grant camera permission, and confirm live tracking works end to end.
**This is your actual portfolio demo link** — not a localhost screenshot.

## Local verification checklist before deploying
Run this on your machine (not in this sandbox) before pushing anywhere:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
mkdir -p models && curl -o models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
cp .env.example .env
pytest --cov=src tests/          # should show 46 passed
streamlit run src/app/streamlit_app.py   # test all 3 exercises live on your webcam
docker build -t fitness-corrector .
docker run -p 8501:8501 fitness-corrector   # confirm it matches local behavior
```

If anything fails at any step, that's the loop continuing on your end: fix,
re-run, repeat until green — then deploy.
