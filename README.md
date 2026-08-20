# Biometric Access Terminal

Sci-fi-styrd AI-säkerhetsterminal som scannar ansiktet live via webcam för identitetsverifiering,
liveness detection samt ålders-/könsestimering. Byggd som stort individuellt projekt i kursen
"Tillämpad AI, datautvinning, maskininlärning och deep learning" vid [AI developer/JENSEN Yrkeshögskola].

Projektet demonstrerar hela ML-pipelinen: dataförberedelse, EDA, unsupervised learning,
supervised learning inklusive deep learning, samt utvärdering.

## Status

ML-pipelinen (notebook 01–06) är komplett. Se [Projektstruktur](#projektstruktur) nedan för
en översikt av vad som är klart.

**Klart:**
- Dataförberedelse och EDA på IMDB-WIKI-datasetet (WIKI-delen), inklusive hantering av
  saknade värden, orimliga åldrar och andra kvalitetsproblem i källdatan
- Ansiktsembeddings via förtränad ArcFace-modell (44 312 bilder)
- Unsupervised learning: K-means, UMAP och HDBSCAN-klustring av embeddings
- Åtkomstklassificering (auktoriserad/ej auktoriserad): egen datainsamling, två jämförda
  metoder (cosine similarity-baslinje och tränat neuralt nätverk)
- Ålder- och könsestimering: delad deep learning-modell (DEX-stil åldersklassificering +
  binär könsklassificering) ovanpå ArcFace-embeddings
- Ansiktsdetektion för live-pipelinen (MediaPipe, empiriskt tröskelvärde för att filtrera
  falska positiver)
- Sammanfattande utvärdering av hela pipelinen mot kursens fem krav
- Liveness detection (`src/liveness.py`): EAR-baserad blinkdetektion via MediaPipe Face
  Mesh, empiriskt kalibrerad och validerad mot en riktig webcam (både levande person och
  hållet foto som spoofing-test). Blink är den avgörande signalen — ett foto kan inte
  blinka; rörelseanalys beräknas som kompletterande diagnostik men avgör inte statusen
  ensam, efter att empirisk testning visat att den gav för många falska avvisningar av
  legitima, naturligt stillasittande användare

**Kommande:**
- Streamlit-dashboard som binder ihop live-pipelinen (webcam → detektion → embedding →
  klassificering → liveness → HUD)

## Projektstruktur

```
biometric-access-terminal/
├── data/               # Rå och bearbetad data (ej i git)
├── notebooks/          # Utforskande analys, en per pipeline-steg (01–06)
├── src/                # Återanvändbar Python-kod
├── app/                # Streamlit-dashboard (kommande)
├── models/             # Tränade och förtränade modeller (ej i git)
└── reports/
    ├── figures/        # Genererade visualiseringar (i git)
    └── metrics/        # Utvärderingsmått per modell, JSON (i git)
```

**Notebooks:**

| Notebook | Innehåll |
|---|---|
| `01_data_preparation.ipynb` | Inläsning av rå WIKI-metadata |
| `02_eda.ipynb` | Åldersberäkning, kvalitetsflaggning, statistisk kartläggning |
| `03_unsupervised_clustering.ipynb` | Embedding-extraktion, K-means/UMAP/HDBSCAN |
| `04_supervised_classification.ipynb` | Åtkomstklassificering (auktoriserad/ej auktoriserad) |
| `05_age_gender_estimation.ipynb` | Ålder- och könsestimering (delad deep learning-modell) |
| `06_evaluation.ipynb` | Sammanfattande utvärdering av hela pipelinen |

## Setup

### Python-miljö

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### Dataset

Projektet använder WIKI-delen av IMDB-WIKI-datasetet ("faces only"-versionen).

```bash
# Ladda ner och packa upp i data/raw/wiki_crop/
# Källa: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar
```

### Förtränad modell för ansiktsdetektion (BlazeFace)

Live-pipelinen använder MediaPipes BlazeFace-modell (short range) för ansiktsdetektion.
Denna fil laddas inte ner automatiskt och måste hämtas manuellt en gång:

```bash
mkdir -p models/pretrained/face_detector
curl -L -o models/pretrained/face_detector/blaze_face_short_range.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
```

### Förtränad modell för ansiktslandmärken (Face Mesh)

Liveness detection använder MediaPipes FaceLandmarker-modell för att extrahera 478
ansiktslandmärken (bl.a. för EAR-baserad blinkdetektion, se `src/liveness.py`). Denna fil
laddas inte ner automatiskt och måste hämtas manuellt en gång, separat från BlazeFace-modellen
ovan:

```bash
mkdir -p models/pretrained/face_mesh
curl -L -o models/pretrained/face_mesh/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### Egna auktoriserade ansiktsbilder

Notebook 04 (åtkomstklassificering) kräver egna webcam-bilder som positiv klass. Dessa
bilder är personidentifierbara och sparas i `data/raw/authorized_captures/`, som är
gitignorad — varje användare av detta repo behöver samla in sina egna bilder via
notebook 04, Del B, innan resten av den notebooken kan köras.

### Reproducerbarhet

Modellträning i notebook 04 och 05 använder `keras.utils.set_random_seed(42)` för
deterministiska, reproducerbara resultat. Utvärderingsmått sparas till `reports/metrics/*.json`
vid träningstillfället och läses därifrån av notebook 06, snarare än att hårdkodas.
