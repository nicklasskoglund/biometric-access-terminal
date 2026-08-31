"""Biometric Access Terminal — Streamlit-dashboard.

Sci-fi-styrd AI-säkerhetsterminal som scannar ansiktet live via webcam,
verifierar auktorisation, estimerar ålder/kön och kör liveness detection.

Detta är den körbara Streamlit-entrypointen. Pipeline-logik (ansikts-
detektion, embedding, klassificering, liveness) kopplas in stegvis i
kommande commits.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from app.model_loader import load_face_detector

st.title("Biometric Access Terminal")
st.write("Steg 2: ansiktsdetektion (bounding box) live via BlazeFace.")

face_detector = load_face_detector()


def video_frame_callback(frame):
    """Kör ansiktsdetektion på varje frame och ritar ut en bounding box.

    Anropas synkront av streamlit-webrtc för varje mottagen videoframe,
    på en egen tråd skild från Streamlits huvudtråd. Detektorn körs i
    MediaPipes IMAGE-läge (synkront detect()), vilket garanterar ett
    resultat per frame utan risk för dropped frames, till skillnad
    från LIVE_STREAM-läget.

    Parameters
    ----------
    frame : av.VideoFrame
        Inkommande videoframe från streamlit-webrtc.

    Returns
    -------
    av.VideoFrame
        Utgående videoframe, med bounding box inritad om ett ansikte
        detekterades.
    """
    img = frame.to_ndarray(format="bgr24")

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

    result = face_detector.detect(mp_image)

    for detection in result.detections:
        bbox = detection.bounding_box
        cv2.rectangle(
            img,
            (bbox.origin_x, bbox.origin_y),
            (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
            color=(0, 255, 0),
            thickness=2,
        )
        confidence = detection.categories[0].score
        cv2.putText(
            img,
            f"{confidence:.2f}",
            (bbox.origin_x, bbox.origin_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color=(0, 255, 0),
            thickness=2,
        )

    return frame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="biometric-access-terminal",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)