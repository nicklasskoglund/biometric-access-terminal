"""Cachade modell-laddningsfunktioner för Streamlit-dashboarden.

Samtliga funktioner är dekorerade med @st.cache_resource, vilket innebär
att respektive modell/detektor bara initieras en gång per Streamlit-
session och sedan återanvänds mellan reruns, snarare än att laddas om
för varje interaktion i UI:t.
"""

import streamlit as st

from pathlib import Path
from tensorflow import keras

from src.face_detection import create_face_detector_for_images, create_face_mesh_detector


@st.cache_resource
def load_face_detector():
    """Laddar och cachar BlazeFace-ansiktsdetektorn (IMAGE-läge).

    Returns
    -------
    mediapipe.tasks.python.vision.FaceDetector
        Initierad detektor, redo att anropas synkront via detect().
    """
    return create_face_detector_for_images()


@st.cache_resource
def load_face_mesh_detector():
    """Laddar och cachar MediaPipe Face Mesh-landmärkesdetektorn.

    Returns
    -------
    mediapipe.tasks.python.vision.FaceLandmarker
        Initierad landmärkesdetektor, redo att anropas synkront via detect().
    """
    return create_face_mesh_detector()


TRAINED_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "trained"
AUTHORIZATION_CLASSIFIER_PATH = TRAINED_MODELS_DIR / "authorization_classifier.keras"


@st.cache_resource
def load_authorization_classifier():
    """Laddar och cachar den tränade auktoriseringsklassificeraren.

    Returns
    -------
    keras.Model
        Tränad modell som tar en 512-dimensionell ArcFace-embedding och
        returnerar en sigmoid-sannolikhet för "auktoriserad".

    Raises
    ------
    FileNotFoundError
        Om modellfilen inte hittas på AUTHORIZATION_CLASSIFIER_PATH.
    """
    if not AUTHORIZATION_CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Modellfil saknas: {AUTHORIZATION_CLASSIFIER_PATH}. "
            "Träna modellen via notebooks/04_supervised_classification.ipynb, "
            "eller se README.md för nedladdningsinstruktioner."
        )
    return keras.models.load_model(AUTHORIZATION_CLASSIFIER_PATH)