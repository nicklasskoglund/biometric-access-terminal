# Biometric Access Terminal

A sci-fi-styled AI security terminal that scans a face live via webcam for
identity verification, liveness detection, and age/gender estimation. Built
as a large individual project for the course "Applied AI, Data Mining,
Machine Learning and Deep Learning" at [AI Developer/JENSEN Yrkeshögskola].

The project demonstrates the full ML pipeline: data preparation, EDA,
unsupervised learning, supervised learning including deep learning, and
evaluation.

## Status

The ML pipeline (notebooks 01–06) is complete, and the live Streamlit
dashboard with a full sci-fi HUD is functional end-to-end. See
[Project Structure](#project-structure) below for an overview.

**Done:**
- Data preparation and EDA on the IMDB-WIKI dataset (WIKI subset),
  including handling of missing values, implausible ages, and other
  quality issues in the source data
- Face embeddings via a pretrained ArcFace model (44,312 images)
- Unsupervised learning: K-means, UMAP, and HDBSCAN clustering of
  embeddings
- Access classification (authorized/unauthorized): custom data
  collection, two compared methods (cosine similarity baseline and a
  trained neural network)
- Age and gender estimation: a shared deep learning model (DEX-style age
  classification + binary gender classification) on top of ArcFace
  embeddings
- Face detection for the live pipeline (MediaPipe, empirically tuned
  threshold to filter false positives)
- Summary evaluation of the full pipeline against the course's five
  requirements
- Liveness detection (`src/liveness.py`): EAR-based blink detection via
  MediaPipe Face Mesh, empirically calibrated and validated against a
  real webcam (both a live person and a held-up photo as a spoofing
  test). Blinking is the decisive signal — a photo cannot blink; motion
  analysis is computed as supplementary diagnostics but does not decide
  the status alone, after empirical testing showed it caused too many
  false rejections for legitimate, naturally still users
- **Streamlit dashboard** (`app/streamlit_app.py`): live webcam pipeline
  (detection → embedding → classification → liveness) wired up end to
  end, with a sci-fi HUD driven by an explicit scan state machine
  (fixed guide frame, animated scan line, corner brackets, ACCESS
  GRANTED/DENIED screens, a mocked "Welcome" screen). See
  [`docs/hud-state-machine.md`](docs/hud-state-machine.md) for the full
  design.

**Upcoming:**
- Deploy to Streamlit Community Cloud (see [Deployment](#deployment-streamlit-community-cloud))

## Project Structure

```
biometric-access-terminal/
├── data/               # Raw and processed data (not in git)
├── notebooks/          # Exploratory analysis, one per pipeline step (01–06)
├── src/                # Reusable Python code
├── app/                # Streamlit dashboard
├── docs/               # Design documentation
├── models/             # Models used by the app (tracked in git, see below)
└── reports/
    ├── figures/        # Generated visualizations (in git)
    └── metrics/        # Per-model evaluation metrics, JSON (in git)
```

**Notebooks:**

| Notebook | Contents |
|---|---|
| `01_data_preparation.ipynb` | Loading raw WIKI metadata |
| `02_eda.ipynb` | Age computation, quality flagging, statistical exploration |
| `03_unsupervised_clustering.ipynb` | Embedding extraction, K-means/UMAP/HDBSCAN |
| `04_supervised_classification.ipynb` | Access classification (authorized/unauthorized) |
| `05_age_gender_estimation.ipynb` | Age and gender estimation (shared deep learning model) |
| `06_evaluation.ipynb` | Summary evaluation of the full pipeline |

## Setup

### Python environment

Requires Python 3.12.

Windows (Git Bash):

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
```

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements-dev.txt
```

- `requirements.txt`: runtime dependencies for the Streamlit app only
  (this is what Streamlit Community Cloud installs).
- `requirements-dev.txt`: everything in `requirements.txt` plus notebooks,
  training and tests (`pytest`).

To only run the app, `pip install -r requirements.txt` is enough.

### Dataset

The project uses the WIKI subset of the IMDB-WIKI dataset (the
"faces only" version).

```bash
# Download and extract into data/raw/wiki_crop/
# Source: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar
```

### Models

All model files the app needs at runtime are small (< 5 MB each) and are
tracked in git, so no manual download is required:

| File | Purpose | Source |
|---|---|---|
| `models/pretrained/face_detector/blaze_face_short_range.tflite` | Face detection (BlazeFace) | [MediaPipe](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite) |
| `models/pretrained/face_mesh/face_landmarker.task` | 478 face landmarks for EAR-based blink detection (`src/liveness.py`) | [MediaPipe](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) |
| `models/trained/authorization_classifier.keras` | Access classification | Trained in notebook 04 |
| `models/trained/age_gender_model.keras` | Age and gender estimation | Trained in notebook 05 |

The MediaPipe models are © Google and distributed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The ArcFace weights (~137 MB) are not in the repo. DeepFace downloads them
automatically to `~/.deepface/weights/` the first time an embedding is
computed, so the first scan after a fresh start is slower.

Only the files listed above are tracked. Other files under `models/`
(e.g. experiments or checkpoints) stay gitignored until they are
explicitly added to `.gitignore`'s exceptions.

### Your own authorized face images

Notebook 04 (access classification) requires your own webcam images as
the positive class. These images are personally identifiable and are
stored in `data/raw/authorized_captures/`, which is gitignored — anyone
using this repo needs to collect their own images via notebook 04, Part
B, before the rest of that notebook can run.

Note that the committed `authorization_classifier.keras` is trained on the
author's face, so other people will get ACCESS DENIED unless they retrain
it on their own images.

### Running the dashboard

```bash
streamlit run app/streamlit_app.py
```

No secrets are needed locally: without a TURN configuration the app logs
a warning and falls back to Google's public STUN server, which is enough
on a local network.

### Reproducibility

Model training in notebooks 04 and 05 uses `keras.utils.set_random_seed(42)`
for deterministic, reproducible results. Evaluation metrics are saved to
`reports/metrics/*.json` at training time and read from there by
notebook 06, rather than being hardcoded.

## Deployment (Streamlit Community Cloud)

The app is deployed from the `main` branch.

1. On [share.streamlit.io](https://share.streamlit.io), create an app from
   this repository, branch `main`, main file path `app/streamlit_app.py`.
2. Under **Advanced settings**, select **Python 3.12**. The Python version
   cannot be changed after deployment without deleting and redeploying
   the app.
3. In the **Secrets** field, paste the `[turn]` section from
   [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)
   with your own TURN server values (see below).
4. Deploy. Community Cloud installs `requirements.txt` (Python packages)
   and `packages.txt` (system libraries needed by OpenCV).

### TURN server

WebRTC on Community Cloud needs a TURN server; STUN alone does not
establish the connection. The app reads the TURN server from the `[turn]`
section in `st.secrets` (`get_rtc_configuration()` in
`app/streamlit_app.py`) and passes it to `streamlit-webrtc` together with
Google's public STUN server. The deployment uses
[ExpressTURN](https://www.expressturn.com/)'s free plan (1,000 GB/month).

For local use, copy the template to `.streamlit/secrets.toml` (gitignored).

### Known limitations

- **Static TURN credentials are visible in the browser.** The free
  ExpressTURN plan only offers long-term username/password credentials,
  which are sent to the browser as part of the ICE configuration. Anyone
  using the app can read them (e.g. via `chrome://webrtc-internals`) and
  use the monthly quota. If that happens, rotate them with "Refresh
  Credentials" in the ExpressTURN dashboard and update the app's secrets.
- **Port 3478 only.** The free plan does not support TURN over port 443,
  so networks that only allow port 443 (some corporate or school networks)
  may block the video connection.
- **Memory.** Measured locally, the app uses about 1.1 GB with one active
  session and about 1.16 GB with two (TensorFlow, ArcFace and MediaPipe).
  This is within Community Cloud's documented limit, but if the app shows
  "This app has gone over its resource limits", [Render](https://render.com)
  is the fallback hosting option.
