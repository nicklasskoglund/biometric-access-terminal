"""Cachade modell-laddningsfunktioner för Streamlit-dashboarden.

Samtliga funktioner är dekorerade med @st.cache_resource, vilket innebär
att respektive modell/detektor bara initieras en gång per Streamlit-
session och sedan återanvänds mellan reruns, snarare än att laddas om
för varje interaktion i UI:t.
"""

import streamlit as st

from src.face_detection import create_face_detector_for_images


@st.cache_resource
def load_face_detector():
    """Laddar och cachar BlazeFace-ansiktsdetektorn (IMAGE-läge).

    Returns
    -------
    mediapipe.tasks.python.vision.FaceDetector
        Initierad detektor, redo att anropas synkront via detect().
    """
    return create_face_detector_for_images()