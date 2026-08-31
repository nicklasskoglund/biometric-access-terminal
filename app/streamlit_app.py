"""Biometric Access Terminal — Streamlit-dashboard.

Sci-fi-styrd AI-säkerhetsterminal som scannar ansiktet live via webcam,
verifierar auktorisation, estimerar ålder/kön och kör liveness detection.

Detta är den körbara Streamlit-entrypointen. Pipeline-logik (ansikts-
detektion, embedding, klassificering, liveness) kopplas in stegvis i
kommande commits.
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Biometric Access Terminal")
st.write("Steg 1: grundläggande webcam-ström via streamlit-webrtc.")

webrtc_streamer(
    key="biometric-access-terminal",
    media_stream_constraints={"video": True, "audio": False},
)