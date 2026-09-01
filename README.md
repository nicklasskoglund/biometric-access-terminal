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
- Upload trained models (`authorization_classifier.keras`,
  `age_gender_model.keras`) to Hugging Face Hub, for cloud deployment
- Deploy to Streamlit Community Cloud

## Project Structure

```
biometric-access-terminal/
├── data/               # Raw and processed data (not in git)
├── notebooks/          # Exploratory analysis, one per pipeline step (01–06)
├── src/                # Reusable Python code
├── app/                # Streamlit dashboard
├── docs/               # Design documentation
├── models/             # Trained and pretrained models (not in git)
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

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### Dataset

The project uses the WIKI subset of the IMDB-WIKI dataset (the
"faces only" version).

```bash
# Download and extract into data/raw/wiki_crop/
# Source: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar
```

### Pretrained face detection model (BlazeFace)

The live pipeline uses MediaPipe's BlazeFace model (short range) for face
detection. This file is not downloaded automatically and must be fetched
manually once:

```bash
mkdir -p models/pretrained/face_detector
curl -L -o models/pretrained/face_detector/blaze_face_short_range.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
```

### Pretrained face landmark model (Face Mesh)

Liveness detection uses MediaPipe's FaceLandmarker model to extract 478
face landmarks (among other things, for EAR-based blink detection, see
`src/liveness.py`). This file is not downloaded automatically and must
be fetched manually once, separately from the BlazeFace model above:

```bash
mkdir -p models/pretrained/face_mesh
curl -L -o models/pretrained/face_mesh/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### Your own authorized face images

Notebook 04 (access classification) requires your own webcam images as
the positive class. These images are personally identifiable and are
stored in `data/raw/authorized_captures/`, which is gitignored — anyone
using this repo needs to collect their own images via notebook 04, Part
B, before the rest of that notebook can run.

### Running the dashboard

```bash
streamlit run app/streamlit_app.py
```

Requires the pretrained models above (BlazeFace, Face Mesh) plus your
own trained `authorization_classifier.keras` and `age_gender_model.keras`
in `models/trained/` (produced by notebooks 04 and 05).

### Reproducibility

Model training in notebooks 04 and 05 uses `keras.utils.set_random_seed(42)`
for deterministic, reproducible results. Evaluation metrics are saved to
`reports/metrics/*.json` at training time and read from there by
notebook 06, rather than being hardcoded.