# VeriMedia

A working implementation of *VeriMedia: A Dual-Pipeline Browser Extension for
Deepfake and Misinformation Detection Using EfficientNet-GRU and
Retrieval-Based NLP* (City College of Calamba, BSCS thesis, Carcueva /
Gardoce / Gimutao / Ortiz).

Two parts, matching the thesis's system architecture (Figure 3-13):

```
verimedia/
├── backend/     Flask API — API Gateway, cache, task queue, the two
│                detection pipelines, and a SQLite-backed history/analytics
│                store.
└── extension/   Manifest V3 Chrome extension — context-menu integration,
                 interstitial warning overlay, and the Overview/History/
                 Settings dashboard (Figures 3-6 to 3-8).
```

## What's real here, and what's a placeholder

This matters for your defense, so it's stated plainly rather than buried:

- **Deepfake pipeline** (`backend/app/pipelines/forensics.py`,
  `deepfake.py`): runs genuine, deterministic digital-forensics signal
  analysis on every image/video — Error Level Analysis, FFT spectral
  anomaly/periodicity detection (the signature GAN upsampling artifacts
  leave behind), face/background noise-consistency analysis, illumination-
  direction consistency, and Haar-cascade face/eye tracking for
  frame-to-frame temporal consistency on video. These are real,
  peer-reviewed forensics techniques used as deepfake-detection baselines
  in the literature, computed from actual pixel data — not random numbers.
  They are **not** the trained EfficientNet-GRU model your thesis names,
  because training one needs a labeled deepfake dataset (FaceForensics++,
  DFDC, Celeb-DF, …) and GPU time. `model_backend.py` is a ready-made slot:
  export a trained model to ONNX, point `VERIMEDIA_ONNX_MODEL_PATH` at it,
  and the backend automatically blends its predictions in — no other code
  changes. See `scripts/export_onnx_example.py` and
  `scripts/export_gru_weights.py`.

- **Misinformation pipeline** (`backend/app/pipelines/misinformation.py`):
  runs genuine TextRank-style claim extraction and TF-IDF + cosine-
  similarity retrieval against a curated fact-check corpus
  (`corpus.json`, ~40 entries) — real retrieval, real similarity scoring,
  fully offline. `retrieval_backend.py` is a ready-made slot for live web
  retrieval (Google Fact Check Tools API, NewsAPI, or Bing) once you have
  an API key and normal internet access; the backend falls back to the
  local corpus automatically if it's not configured or a request fails.

- Both are **functional today** — see "Try it" below — and both are
  designed so that swapping in the exact models your thesis specifies is a
  config change, not a rewrite.

## Quick start

### 1. Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env
python run.py
```

The API is now listening on `http://127.0.0.1:8000`. Check
`http://127.0.0.1:8000/api/v1/health`.

Try the pipelines directly, e.g. with the sample images in `sample_data/`:

```bash
curl -X POST -F "file=@sample_data/sample_photo.jpg" http://127.0.0.1:8000/api/v1/analyze/image
curl -X POST -F "file=@sample_data/sample_video.mp4"  http://127.0.0.1:8000/api/v1/analyze/video
curl -X POST -H "Content-Type: application/json" \
     -d '{"text":"COVID-19 vaccines contain microchips used to track people."}' \
     http://127.0.0.1:8000/api/v1/analyze/text
```

### 2. Extension

1. Open `chrome://extensions` in Chrome (or any Chromium-based browser).
2. Turn on **Developer mode** (top right).
3. Click **Load unpacked** and select the `extension/` folder.
4. Pin VeriMedia to the toolbar. Click it once — the popup should show a
   green connection dot if the backend (step 1) is running.
5. On any webpage: **right-click an image or video** → "Verify Media
   (Image/Video) with VeriMedia", or **select some text** → right-click →
   "Fact-check with VeriMedia". A result card appears over the media (or
   top-right for text); flagged media is blurred with a "Reveal Anyway"
   button.
6. Click the toolbar icon → **Open Dashboard** for the Overview / History
   / Settings screens (Figures 3-6–3-8). Settings is where you point the
   extension at a different backend URL if you deploy one.

Both parts were tested end-to-end in development (see "How this was
verified" below) — the context-menu → backend → interstitial-overlay path
and the dashboard's live Overview/History/Settings all work against a
running backend.

## API reference

| Endpoint | Method | Notes |
|---|---|---|
| `/api/v1/analyze/image` | POST (multipart `file`, `source_url`) | Returns verdict, confidence, per-signal breakdown |
| `/api/v1/analyze/video` | POST (multipart `file`, `source_url`, `async`) | `async=true` returns `{job_id}` immediately; poll `/api/v1/jobs/<id>` |
| `/api/v1/analyze/text` | POST (JSON `text`, `source_url`) | Returns verdict (`true`/`misleading`/`false`/`unverified`), confidence, retrieved evidence |
| `/api/v1/history` | GET (`limit`, `offset`, `media_type`) | Paginated scan history |
| `/api/v1/analytics/overview` | GET | Stat tiles + activity trend, powers the Overview panel |
| `/api/v1/health` | GET | Used by the popup/dashboard connection indicator |

Full request/response shapes are in `backend/app/routes/*.py` and the
pipeline modules — every field returned is documented inline.

## Configuration

All backend behavior is controlled via `backend/.env` (see
`.env.example` for every option and its default): detection thresholds,
video frame-sampling budget, cache TTL, the ONNX model path, and the live
retrieval provider/API key.

## Scaling this to production

The code is written so each of these is a contained swap, not a rewrite:

- **Database**: `app/db.py` uses plain parameterised SQL via Python's
  built-in `sqlite3`. Swapping `sqlite3.connect(...)` for
  `psycopg2.connect(...)` (see `requirements-production.txt`) against a
  real Postgres instance — matching Figure 3-13's "Database (PostgreSQL)"
  — needs no query rewrites.
- **Task queue**: `app/queue.py` is an in-process thread pool draining a
  `queue.Queue`. Swap `submit()`'s body for `celery_task.delay(...)` with
  Celery + Redis for real horizontal scaling; call sites don't change.
- **Deep models / live retrieval**: see the "What's real here" section
  above — both are config-driven, not code-driven, swaps.
- **WSGI server**: `run.py` uses Flask's dev server. Run
  `gunicorn -w 4 -b 0.0.0.0:8000 run:app` in production instead.
- **CORS**: set `VERIMEDIA_ALLOWED_ORIGINS` to your packed extension's
  `chrome-extension://<id>` origin instead of `*` once you have a stable
  extension ID.

## How this was verified

Built and tested inside a network-restricted sandbox (no access to PyPI,
HuggingFace, or search engines beyond what's preinstalled), so verification
leaned on what's actually testable offline:

- Every backend endpoint (`analyze/image`, `analyze/video` sync + async job
  polling, `analyze/text`, `history`, `analytics/overview`, `health`) was
  exercised with `curl` against synthetic test images/video and a range of
  text claims (known-false, known-true, sensationalized, unverifiable),
  confirming correct verdicts, caching, and history/analytics aggregation.
- The forensic spectral-anomaly detector was unit-checked against a
  checkerboard (should score highly periodic), random noise (should score
  near zero), and a smooth gradient (should score low-to-moderate) to
  confirm it behaves monotonically, not just "runs without crashing."
- The extension was loaded unpacked into real Chromium (via Playwright +
  Xvfb) and driven through its **actual production code path** — the same
  handler functions the native right-click context menu calls — for both
  the image-verification and text-fact-check flows, confirming the
  background service worker correctly fetches media, calls the live
  backend, and that the content script renders the interstitial card,
  applies the blur, and the "Reveal Anyway" toggle works.
- The dashboard's Overview (stat tiles + activity trend chart), History
  (paginated table with live data), and Settings (connection test) panels
  were screenshotted against the live backend and visually confirmed to
  match the thesis's Figures 3-6–3-8.

What was **not** verified here, and is squarely the subject of your
thesis's own evaluation chapter: real-world accuracy/precision/recall/F1
against an actual deepfake dataset, and the ISO/IEC 25010 usability
evaluation with real respondents. The engine here is honest, functional,
and ready for that evaluation — it isn't a substitute for it.

## Known limitations

- The forensic heuristic engine is a rule-based baseline, not a trained
  classifier — treat its verdicts as indicative, not authoritative, until
  you've validated it (or a plugged-in trained model) against a real
  dataset.
- The misinformation corpus (`corpus.json`) is intentionally small (~40
  entries) and general-purpose; claims outside its topic coverage
  correctly return `unverified` rather than a guess. Expand the corpus or
  wire up live retrieval for broader coverage.
- Video analysis samples up to `VERIMEDIA_VIDEO_MAX_FRAMES` (default 24)
  frames for latency reasons — it is not a frame-exhaustive scan.
- The auto-protect (background page-scanning) feature is off by default
  and intentionally conservative (a handful of large images per page) to
  avoid hammering the backend or the pages you visit.
