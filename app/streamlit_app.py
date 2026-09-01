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
from src.liveness import LEFT_EYE_INDICES, LivenessDetector, RIGHT_EYE_INDICES
from src.utils import BoundingBox
from app.hud_overlay import (
    COLOR_TEAL,
    draw_corner_brackets,
    draw_full_screen_message,
    draw_hud_text,
    draw_scan_line,
    get_denied_blink_color,
    get_guide_frame,
)
from app.scan_state_machine import ScanState, ScanStateMachine

st.title("Biometric Access Terminal")
st.write("Steg 6: auktorisering + ålder/kön (bakgrundstråd) + liveness detection (varje frame).")

face_detector = load_face_detector()
face_mesh_detector = load_face_mesh_detector()
authorization_classifier = load_authorization_classifier()
age_gender_model = load_age_gender_model()

EYE_INDICES = LEFT_EYE_INDICES + RIGHT_EYE_INDICES
NUM_AGE_CLASSES = 121


class AuthorizationResultStore:
    """Trådsäker, mergande lagring av senaste resultat.

    Flera trådar skriver till denna store samtidigt: video_frame_callback
    uppdaterar liveness-status varje frame, medan embedding_worker
    uppdaterar auktorisering/ålder/kön i sin egen takt. update() mergar
    därför bara in de nyckelord som anges, istället för att skriva över
    hela resultatet, så att de olika trådarna inte raderar varandras fält.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = {
            "status": "väntar på ansikte",
            "probability": None,
            "age": None,
            "gender": None,
            "liveness": "väntar på ansikte",
            "auth_state": "no_face",
        }

    def update(self, **kwargs):
        with self._lock:
            self._latest.update(kwargs)

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


@st.cache_resource
def get_liveness_detector():
    return LivenessDetector()


@st.cache_resource
def get_scan_state_machine():
    return ScanStateMachine()


result_store = get_result_store()
latest_frame_store = get_latest_frame_store()
liveness_detector = get_liveness_detector()
scan_state_machine = get_scan_state_machine()


def video_frame_callback(frame):
    """Kör ansiktsdetektion, landmärkesdetektion, liveness detection och
    driver HUD-scanningsflödet (ScanStateMachine).

    Ansiktsdetektion (BlazeFace) och landmärkesdetektion (Face Mesh) körs
    varje frame oavsett HUD-tillstånd, eftersom liveness detection
    (blinkdetektion) kräver en obruten sekvens av frames för att fungera
    tillförlitligt - se LivenessDetector-kalibreringen i src/liveness.py.

    HUD:en ritas relativt en fast, centrerad guide-ram (get_guide_frame())
    snarare än den riktiga, dynamiskt detekterade bounding boxen, för att
    undvika det jitter som annars uppstår i hörnhakarna. Den riktiga
    bounding boxen används fortfarande internt för ansiktsbeskärning till
    embedding-modellen, via latest_frame_store.

    ScanStateMachine har exakt en skrivare (denna callback), enligt
    Single Writer-principen: embedding_worker skriver bara till
    result_store, som denna callback läser för att avgöra auth_result.
    Detta gör att ScanStateMachine slipper egen trådsäkerhet.

    Parameters
    ----------
    frame : av.VideoFrame
        Inkommande videoframe från streamlit-webrtc.

    Returns
    -------
    av.VideoFrame
        Utgående videoframe, med HUD-overlay inritad enligt aktuellt
        ScanState.
    """
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)  # spegelvänd horisontellt - kameraströmmen
                            # kommer ospeglad från webbläsaren, medan
                            # användare förväntar sig en "spegel"-vy
                            # (rörelser åt höger ska synas åt höger).
    frame_height, frame_width = img.shape[:2]
    guide_box = get_guide_frame(frame_width, frame_height)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

    detection_result = face_detector.detect(mp_image)
    mesh_result = face_mesh_detector.detect(mp_image)

    face_detected = bool(detection_result.detections and mesh_result.face_landmarks)

    if face_detected:
        detection = detection_result.detections[0]
        bbox = detection.bounding_box

        our_bbox = BoundingBox(
            origin_x=bbox.origin_x,
            origin_y=bbox.origin_y,
            width=bbox.width,
            height=bbox.height,
        )
        latest_frame_store.update(img.copy(), our_bbox)

        pixel_landmarks = landmarks_to_pixel_array(
            mesh_result.face_landmarks[0], frame_width, frame_height
        )
        face_bbox_tuple = (bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
        liveness_status = liveness_detector.update(img, pixel_landmarks, face_bbox_tuple)
        result_store.update(liveness=liveness_status)
    else:
        latest_frame_store.update(None, None)
        liveness_detector.reset()
        liveness_status = "osäker_ännu"
        result_store.update(liveness="väntar på ansikte")

    current_auth_state = result_store.get().get("auth_state")
    auth_result = {"authorized": "authorized", "unauthorized": "unauthorized"}.get(current_auth_state)

    scan_state_machine.update(
        face_detected=face_detected,
        liveness_status=liveness_status,
        auth_result=auth_result,
    )
    state = scan_state_machine.state

    if state in (ScanState.IDLE, ScanState.STABILIZING, ScanState.SCANNING):
        img = draw_corner_brackets(img, guide_box, color=COLOR_TEAL, glow=False)
        if state == ScanState.SCANNING:
            img = draw_scan_line(img, guide_box, color=COLOR_TEAL)
        img = draw_hud_text(img, scan_state_machine.display_text(), guide_box, color=COLOR_TEAL)
    elif state == ScanState.RESULT_GRANTED:
        img = draw_full_screen_message(img, "ACCESS GRANTED", color=COLOR_TEAL)
    elif state == ScanState.RESULT_DENIED:
        img = draw_full_screen_message(img, "ACCESS DENIED", color=get_denied_blink_color())
    elif state == ScanState.WELCOME:
        img = draw_full_screen_message(img, "WELCOME", color=COLOR_TEAL)

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
            result_store.update(
                status="väntar på ansikte", probability=None, age=None, gender=None, auth_state="no_face"
            )
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
            auth_state = "authorized" if auth_probability > 0.5 else "unauthorized"

            age_probs, gender_prob = age_gender_model.predict(embedding_batch, verbose=0)
            estimated_age = float(np.sum(age_probs[0] * age_class_indices))
            estimated_gender = "man" if gender_prob[0][0] > 0.5 else "kvinna"

            elapsed_ms = (time.time() - start_time) * 1000
            result_store.update(
                status=f"{auth_status} ({elapsed_ms:.0f} ms)",
                probability=auth_probability,
                age=estimated_age,
                gender=estimated_gender,
                auth_state=auth_state,
            )
        except Exception as error:
            result_store.update(
                status=f"fel: {error}", probability=None, age=None, gender=None, auth_state="error"
            )


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
            f"**Kön:** {result['gender']} | **Liveness:** {result['liveness']}"
        )
    else:
        status_placeholder.write(
            f"**Status:** {result['status']} | **Liveness:** {result['liveness']}"
        )
    time.sleep(0.2)