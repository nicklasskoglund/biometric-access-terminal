"""Biometric Access Terminal — Streamlit-dashboard.

Sci-fi-styrd AI-säkerhetsterminal som scannar ansiktet live via webcam,
verifierar auktorisation, estimerar ålder/kön och kör liveness detection.

Detta är den körbara Streamlit-entrypointen. Pipeline-logik (ansikts-
detektion, embedding, klassificering, liveness) kopplas in stegvis i
kommande commits.
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from app.model_loader import (
    load_age_gender_model,
    load_authorization_classifier,
    load_face_detector,
    load_face_mesh_detector,
)
from src.embeddings import crop_face_with_padding, get_face_embedding
from src.face_detection import landmarks_to_pixel_array
from src.liveness import LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from src.utils import BoundingBox

st.title("Biometric Access Terminal")
st.write("Steg 5: auktoriseringsklassificering + ålder/kön-estimering i bakgrundstråd.")

face_detector = load_face_detector()
face_mesh_detector = load_face_mesh_detector()
authorization_classifier = load_authorization_classifier()
age_gender_model = load_age_gender_model()

EYE_INDICES = LEFT_EYE_INDICES + RIGHT_EYE_INDICES
NUM_AGE_CLASSES = 121


class AuthorizationResultStore:
    """Trådsäker lagring av det senaste auktoriserings- och ålder/kön-resultatet."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = {
            "status": "väntar på ansikte",
            "probability": None,
            "age": None,
            "gender": None,
        }

    def update(self, status, probability, age=None, gender=None):
        with self._lock:
            self._latest = {
                "status": status,
                "probability": probability,
                "age": age,
                "gender": gender,
            }

    def get(self):
        with self._lock:
            return self._latest.copy()


class LatestFrameStore:
    """Trådsäker lagring av den senaste framen och dess bounding box.

    video_frame_callback skriver hit varje frame (billigt). En separat
    bakgrundstråd läser härifrån i sin egen takt och kör den tunga
    embedding-beräkningen, helt frikopplat från videoleveransen, så att
    en långsam embedding-beräkning aldrig fördröjer en utgående videoframe.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame_id = 0
        self._img = None
        self._bbox = None

    def update(self, img, bbox):
        with self._lock:
            self._frame_id += 1
            self._img = img
            self._bbox = bbox

    def get(self):
        with self._lock:
            return self._frame_id, self._img, self._bbox


@st.cache_resource
def get_result_store():
    return AuthorizationResultStore()


@st.cache_resource
def get_latest_frame_store():
    return LatestFrameStore()


result_store = get_result_store()
latest_frame_store = get_latest_frame_store()


def video_frame_callback(frame):
    """Kör ansiktsdetektion och landmärkesdetektion, ritar overlay.

    Skriver senaste frame + bounding box till latest_frame_store för
    att den tunga embedding-beräkningen ska kunna ske i en separat
    bakgrundstråd, utan att blockera denna callback.

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

    if detection_result.detections:
        detection = detection_result.detections[0]
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

        our_bbox = BoundingBox(
            origin_x=bbox.origin_x,
            origin_y=bbox.origin_y,
            width=bbox.width,
            height=bbox.height,
        )
        latest_frame_store.update(img.copy(), our_bbox)
    else:
        latest_frame_store.update(None, None)

    mesh_result = face_mesh_detector.detect(mp_image)

    if mesh_result.face_landmarks:
        pixel_landmarks = landmarks_to_pixel_array(
            mesh_result.face_landmarks[0], frame_width, frame_height
        )
        for idx in EYE_INDICES:
            x, y = pixel_landmarks[idx]
            cv2.circle(img, (int(x), int(y)), 2, color=(0, 200, 255), thickness=-1)

    return frame.from_ndarray(img, format="bgr24")


def embedding_worker():
    """Bakgrundstråd: kör embedding, auktoriseringsklassificering och
    ålder/kön-estimering i sin egen takt.

    Läser kontinuerligt den senaste framen från latest_frame_store.
    Bearbetar bara nya frames (spårat via frame_id) för att undvika att
    processa samma frame flera gånger. Eftersom denna tråd är helt
    frikopplad från video_frame_callback påverkar den tunga
    beräkningstiden aldrig videons framerate.

    Åldern beräknas som softmax-outputens förväntade värde över de 121
    åldersklasserna (DEX-metoden), inte enbart argmax, enligt
    build_age_gender_model()-dokumentationen i src/models.py.
    """
    last_processed_id = -1
    age_class_indices = np.arange(NUM_AGE_CLASSES)

    while True:
        frame_id, img, bbox = latest_frame_store.get()

        if img is None:
            result_store.update("väntar på ansikte", None)
            time.sleep(0.05)
            continue

        if frame_id == last_processed_id:
            time.sleep(0.02)
            continue

        last_processed_id = frame_id
        face_crop = crop_face_with_padding(img, bbox)

        try:
            start_time = time.time()
            embedding = get_face_embedding(face_crop)
            embedding_batch = embedding[np.newaxis, :]

            auth_probability = float(
                authorization_classifier.predict(embedding_batch, verbose=0)[0][0]
            )
            auth_status = "auktoriserad" if auth_probability > 0.5 else "ej auktoriserad"

            age_probs, gender_prob = age_gender_model.predict(embedding_batch, verbose=0)
            estimated_age = float(np.sum(age_probs[0] * age_class_indices))
            estimated_gender = "man" if gender_prob[0][0] > 0.5 else "kvinna"

            elapsed_ms = (time.time() - start_time) * 1000
            result_store.update(
                f"{auth_status} ({elapsed_ms:.0f} ms)",
                auth_probability,
                age=estimated_age,
                gender=estimated_gender,
            )
        except Exception as error:
            result_store.update(f"fel: {error}", None)


if "embedding_worker_started" not in st.session_state:
    worker_thread = threading.Thread(target=embedding_worker, daemon=True)
    worker_thread.start()
    st.session_state.embedding_worker_started = True


ctx = webrtc_streamer(
    key="biometric-access-terminal",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

status_placeholder = st.empty()

while ctx.state.playing:
    result = result_store.get()
    if result["age"] is not None:
        status_placeholder.write(
            f"**Status:** {result['status']} | **Ålder:** {result['age']:.0f} år | "
            f"**Kön:** {result['gender']}"
        )
    else:
        status_placeholder.write(f"**Status:** {result['status']}")
    time.sleep(0.2)