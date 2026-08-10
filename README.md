# Biometric Access Terminal

Sci-fi-styrd AI-säkerhetsterminal som scannar ansiktet live via webcam för identitetsverifiering,
liveness detection samt ålders-/könsestimering. Byggd som stort individuellt projekt i kursen
"Tillämpad AI, datautvinning, maskininlärning och deep learning" vid [AI developer/JENSEN Yrkeshögskola].

Projektet demonstrerar hela ML-pipelinen: dataförberedelse, EDA, unsupervised learning,
supervised learning inklusive deep learning, samt utvärdering.

## Status

Under aktiv utveckling. Se [Projektstruktur](#projektstruktur) nedan för vad som är klart.

**Klart:**
- Dataförberedelse och EDA på IMDB-WIKI-datasetet (WIKI-delen)
- Missing-values-hantering och åldersvalidering

**Pågående:**
- Ansiktsdetektion för live-pipelinen (MediaPipe)

**Kommande:**
- Ansiktsembeddings, klustring, klassificeringsmodeller
- Streamlit-dashboard

## Projektstruktur

```
biometric-access-terminal/
├── data/               # Rå och bearbetad data (ej i git)
├── notebooks/          # Utforskande analys, en per pipeline-steg
├── src/                # Återanvändbar Python-kod
├── app/                # Streamlit-dashboard
├── models/             # Tränade och förtränade modeller (ej i git)
└── reports/            # Slutrapport och figurer
```

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

### Förtränad modell för ansiktsdetektion

Live-pipelinen använder MediaPipes BlazeFace-modell (short range) för ansiktsdetektion.
Denna fil laddas inte ner automatiskt och måste hämtas manuellt en gång:

```bash
mkdir -p models/pretrained/face_detector
curl -L -o models/pretrained/face_detector/blaze_face_short_range.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
```