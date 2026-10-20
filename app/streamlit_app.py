"""Biometric Access Terminal — Streamlit-dashboard.

Sci-fi-styrd AI-säkerhetsterminal som scannar ansiktet live via webcam,
verifierar auktorisation, estimerar ålder/kön och kör liveness detection.

Detta är den körbara Streamlit-entrypointen. Återanvändbar pipeline-logik
(ansiktsdetektion, embedding, liveness) ligger i src/, medan HUD-rendering
och scanningsflödets tillståndsmaskin ligger i app/.
"""

import faulthandler

# TEMP (fix/cloud-segfault): skriver ut Python-stacken för alla trådar
# till stderr vid en segfault, så att den syns i Community Cloud-loggen.
# Måste ligga före de tunga importerna (cv2, mediapipe, tensorflow).
faulthandler.enable()

import logging
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

face_detector = load_face_detector()
face_mesh_detector = load_face_mesh_detector()
authorization_classifier = load_authorization_classifier()
age_gender_model = load_age_gender_model()

EYE_INDICES = LEFT_EYE_INDICES + RIGHT_EYE_INDICES
NUM_AGE_CLASSES = 121
# Hur länge embedding_worker väntar på en ny frame innan den antar att
# webrtc-strömmen har upphört (t.ex. stängd flik, där ingen rerun sker
# som kan sätta stoppsignalen) och avslutar sig själv. video_frame_callback
# skriver till latest_frame_store varje frame, även utan ansikte.
STREAM_TIMEOUT_SECONDS = 10.0

logger = logging.getLogger(__name__)


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


# Scanningstillståndet (frames, resultat, liveness-historik, HUD-tillstånd)
# lagras per session i st.session_state, inte via @st.cache_resource.
# cache_resource delas mellan alla sessioner i processen, vilket skulle
# låta besökare B:s embedding_worker bearbeta besökare A:s frames och visa
# A:s ålder/kön/auktorisering för B. Modellerna ovan delas däremot avsiktligt,
# eftersom de är tillståndslösa.
if "result_store" not in st.session_state:
    st.session_state.result_store = AuthorizationResultStore()
    st.session_state.latest_frame_store = LatestFrameStore()
    st.session_state.liveness_detector = LivenessDetector()
    st.session_state.scan_state_machine = ScanStateMachine()

# video_frame_callback körs i en webrtc-tråd utan åtkomst till
# st.session_state, men läser dessa globals från sin egen sessions
# script-körning (Streamlit kör varje rerun i en egen modulnamnrymd).
result_store = st.session_state.result_store
latest_frame_store = st.session_state.latest_frame_store
liveness_detector = st.session_state.liveness_detector
scan_state_machine = st.session_state.scan_state_machine


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


def embedding_worker(latest_frame_store, result_store, stop_event):
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

    Tråden avslutas när stop_event sätts (användaren stoppade strömmen),
    eller när ingen ny frame kommit på STREAM_TIMEOUT_SECONDS (strömmen
    har upphört utan rerun, t.ex. stängd flik). Utan detta skulle varje
    avslutad session lämna kvar en tråd som håller sin sista frame i minnet.

    Parameters
    ----------
    latest_frame_store : LatestFrameStore
        Sessionens frame-store, som video_frame_callback skriver till.
    result_store : AuthorizationResultStore
        Sessionens resultat-store, som denna tråd skriver till.
    stop_event : threading.Event
        Stoppsignal, sätts av sync_embedding_worker().
    """
    last_processed_id = -1
    last_seen_id = -1
    last_seen_time = time.monotonic()
    age_class_indices = np.arange(NUM_AGE_CLASSES)

    while not stop_event.is_set():
        frame_id, img, bbox = latest_frame_store.get()

        now = time.monotonic()
        if frame_id != last_seen_id:
            last_seen_id, last_seen_time = frame_id, now
        elif now - last_seen_time > STREAM_TIMEOUT_SECONDS:
            break

        if img is None:
            result_store.update(
                status="väntar på ansikte", probability=None, age=None, gender=None, auth_state="no_face"
            )
            stop_event.wait(0.05)
            continue

        if frame_id == last_processed_id:
            stop_event.wait(0.02)
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
        except Exception:
            # Hela exceptionen (med traceback) loggas till serverloggen,
            # medan besökaren bara ser ett generiskt meddelande - råa
            # felmeddelanden kan avslöja sökvägar och interna detaljer.
            logger.exception("Analys av frame misslyckades i embedding_worker")
            result_store.update(
                status="fel vid analys, försök igen", probability=None, age=None, gender=None, auth_state="error"
            )

        # Medveten paus mellan klassificeringar. Ingen funktionell anledning
        # att köra så tätt som möjligt - MIN_SCAN_DURATION garanterar redan
        # minst 2 sekunders scanning oavsett hur snabbt resultatet blir
        # klart. Minskar allokeringstakten (färre numpy-temporärer per
        # sekund), vilket sannolikt minskar frekvensen av Pythons GC-pauser
        # och därmed störningar på aiortc:s asynkrona händelseloop.
        stop_event.wait(0.3)


def sync_embedding_worker(ctx):
    """Säkerställer exakt en embedding_worker per session medan strömmen spelar.

    Thread-objektet och dess stoppsignal lagras i st.session_state, så att
    reruns inte startar nya trådar. En ny tråd startas bara om ingen finns,
    om den tidigare har avslutats (t.ex. efter timeout) eller om den
    håller på att stoppas. När strömmen inte spelar sätts stoppsignalen.

    Parameters
    ----------
    ctx : streamlit_webrtc.WebRtcStreamerContext
        Kontexten från webrtc_streamer(), vars state.playing styr workern.
    """
    thread = st.session_state.get("embedding_worker_thread")
    stop_event = st.session_state.get("embedding_worker_stop_event")

    if ctx.state.playing:
        if thread is None or not thread.is_alive() or stop_event.is_set():
            stop_event = threading.Event()
            thread = threading.Thread(
                target=embedding_worker,
                args=(latest_frame_store, result_store, stop_event),
                name="embedding_worker",
                daemon=True,
            )
            thread.start()
            st.session_state.embedding_worker_thread = thread
            st.session_state.embedding_worker_stop_event = stop_event
    elif stop_event is not None:
        stop_event.set()


def get_rtc_configuration():
    """Bygger WebRTC-konfigurationen med TURN-server från st.secrets.

    På Streamlit Community Cloud räcker inte STUN för att etablera
    WebRTC-anslutningen, så en TURN-server krävs. Uppgifterna läses från
    sektionen [turn] i .streamlit/secrets.toml lokalt, eller i appens
    Secrets-inställningar på Community Cloud. Se
    .streamlit/secrets.toml.example.

    Säkerhet: ExpressTURN:s gratisplan använder statiska (long-term)
    credentials. De skickas till webbläsaren som en del av ICE-
    konfigurationen och kan läsas av alla som öppnar appen, t.ex. via
    chrome://webrtc-internals. Det värsta som kan hända är att någon annan
    förbrukar månadskvoten. Vid misstänkt missbruk roteras de med
    "Refresh Credentials" i ExpressTURN-dashboarden, varefter [turn] i
    Cloud-secrets uppdateras med de nya värdena.

    Returns
    -------
    dict | None
        rtc_configuration för webrtc_streamer(), eller None om TURN-secrets
        saknas. Vid None använder streamlit-webrtc Googles publika
        STUN-server, vilket räcker för lokal körning. Fallbacken loggas
        som varning, så att en felkonfiguration syns i Cloud-loggen.
    """
    try:
        turn = st.secrets["turn"]
        urls = turn["urls"]
        turn_server = {
            # list() på en sträng skulle ge enskilda tecken utan att något
            # fel uppstår, så en ensam URL-sträng läggs i en lista.
            "urls": [urls] if isinstance(urls, str) else list(urls),
            "username": turn["username"],
            "credential": turn["credential"],
        }
    except (KeyError, FileNotFoundError):
        logger.warning("TURN not configured in st.secrets, falling back to STUN")
        return None
    # STUN först, så att direkta anslutningar används när det går. Då
    # reläas bara trafik som verkligen behöver det, och kvoten sparas.
    return {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}, turn_server]}


ctx = webrtc_streamer(
    key="biometric-access-terminal",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration=get_rtc_configuration(),
)

@st.fragment(run_every=1.0)
def render_status_panel():
    """Renderar statusrad och (villkorlig) reset-knapp, och övervakar
    embedding_worker via sync_embedding_worker() så att en worker som
    avslutats under pågående ström (t.ex. efter ett nätverksavbrott
    längre än STREAM_TIMEOUT_SECONDS) startas om inom en sekund.

    Körs som en Streamlit-fragment snarare än en blockerande while-loop:
    en fragment kan köras om både periodiskt (run_every) och vid
    användarinteraktion (knapptryck) utan att blockera resten av sidan.
    En vanlig while-loop med time.sleep() blockerar Streamlits
    körningsmotor helt, vilket gör att knapptryck aldrig hinner
    registreras - det var grundorsaken till att "Scanna igen"-knappen
    varken syntes eller fungerade i tidigare version.

    scan_state_machine.force_reset() anropas härifrån (huvudtråden vid
    knapptryck), som ett medvetet undantag från att video_frame_callback
    annars är ensam skrivare - se force_reset()-docstring i
    scan_state_machine.py.
    """
    sync_embedding_worker(ctx)

    state = scan_state_machine.state

    if state in (ScanState.WELCOME, ScanState.RESULT_DENIED):
        if st.button("Scanna igen", key="reset_scan_button"):
            scan_state_machine.force_reset()

    result = result_store.get()
    if result["age"] is not None:
        st.write(
            f"**Status:** {result['status']} | **Ålder:** {result['age']:.0f} år | "
            f"**Kön:** {result['gender']} | **Liveness:** {result['liveness']}"
        )
    else:
        st.write(f"**Status:** {result['status']} | **Liveness:** {result['liveness']}")


render_status_panel()