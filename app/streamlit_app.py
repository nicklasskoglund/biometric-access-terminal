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

from app.model_loader import load_face_detector, load_face_mesh_detector
from src.face_detection import landmarks_to_pixel_array
from src.liveness import LEFT_EYE_INDICES, RIGHT_EYE_INDICES

st.title("Biometric Access Terminal")
st.write("Steg 3: ansiktsdetektion (bounding box) + ögonlandmärken (Face Mesh) live.")

face_detector = load_face_detector()
face_mesh_detector = load_face_mesh_detector()

EYE_INDICES = LEFT_EYE_INDICES + RIGHT_EYE_INDICES


def video_frame_callback(frame):
    """Kör ansiktsdetektion och ögonlandmärkesdetektion på varje frame.

    Ritar ut en bounding box runt detekterat ansikte samt de 12
    ögonlandmärken (vänster + höger öga) som liveness.py använder för
    EAR-beräkning, som en visuell verifiering innan liveness-logiken
    kopplas in i ett senare steg.

    Parameters
    ----------
    frame : av.VideoFrame
        Inkommande videoframe från streamlit-webrtc.

    Returns
    -------
    av.VideoFrame
        Utgående videoframe, med bounding box och ögonlandmärken
        inritade om ett ansikte detekterades.
    """
    img = frame.to_ndarray(format="bgr24")
    frame_height, frame_width = img.shape[:2]

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

    detection_result = face_detector.detect(mp_image)

    for detection in detection_result.detections:
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

    mesh_result = face_mesh_detector.detect(mp_image)

    if mesh_result.face_landmarks:
        pixel_landmarks = landmarks_to_pixel_array(
            mesh_result.face_landmarks[0], frame_width, frame_height
        )
        for idx in EYE_INDICES:
            x, y = pixel_landmarks[idx]
            cv2.circle(img, (int(x), int(y)), 2, color=(0, 200, 255), thickness=-1)

    return frame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="biometric-access-terminal",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)