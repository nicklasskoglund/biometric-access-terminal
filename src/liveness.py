"""
liveness.py

Liveness detection för Biometric Access Terminal.

Syfte: skilja en levande person från ett statiskt foto som hålls upp mot
kameran, genom att kombinera två passiva signaler över tid:

1. Blinkdetektion via Eye Aspect Ratio (EAR), beräknad från MediaPipe
   Face Mesh-landmärken kring ögonen.
2. Enkel rörelseanalys (frame-diff) inom ansiktsregionen, för att fånga
   naturlig mikrorörelse som ett hållet foto saknar.

Detta är en passiv, empiriskt kalibrerad ansats — inte fullskalig
kommersiell liveness detection (ingen textur-/rPPG-analys), utan en
medveten avgränsning som är rimlig för projektets scope.
"""

import time
from collections import deque

import cv2
import numpy as np


# MediaPipe Face Mesh landmärkes-index för EAR-beräkning.
# Ordning: [p1, p2, p3, p4, p5, p6] enligt Soukupová & Čech-formeln
# (p1/p4 = horisontella hörn, p2/p3/p5/p6 = vertikala ögonlockspunkter).
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Standardvärden för blinkdetektion — kalibreras empiriskt mot er
# egen webcam/MediaPipe-uppsättning via testscriptet, se README.
EAR_THRESHOLD = 0.21
EAR_CONSEC_FRAMES = 2

# Platshållarvärden — kalibreras empiriskt mot er webcam via testscriptet.
MOTION_THRESHOLD = 1.5


def compute_ear(landmarks, eye_indices):
    """
    Beräknar Eye Aspect Ratio (EAR) för ett öga utifrån MediaPipe
    Face Mesh-landmärken.

    EAR-formeln (Soukupová & Čech, 2016) mäter förhållandet mellan
    ögats vertikala och horisontella avstånd. Ett öppet öga ger ett
    högre värde (~0,25–0,35), ett slutet öga ett lägre (~0,0–0,15).

    Args:
        landmarks: lista/array med (x, y)-koordinater för samtliga
            478 ansiktslandmärken från MediaPipe Face Mesh.
        eye_indices: lista med 6 landmärkes-index i ordningen
            [p1, p2, p3, p4, p5, p6], där p1/p4 är de horisontella
            hörnpunkterna och p2/p3/p5/p6 de vertikala ögonlockspunkterna.

    Returns:
        float: EAR-värdet för ögat.
    """
    p1, p2, p3, p4, p5, p6 = [np.array(landmarks[i]) for i in eye_indices]

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def detect_blink(ear_history, threshold=EAR_THRESHOLD, consec_frames=EAR_CONSEC_FRAMES):
    """
    Avgör om en blink inträffat baserat på en EAR-historik över tid.

    En blink räknas som ett mönster där EAR-värdet ligger under
    `threshold` i minst `consec_frames` sammanhängande frames, och
    därefter återhämtar sig till över tröskeln. Detta skiljer en
    faktisk blink från enstaka brusiga lågvärden.

    Args:
        ear_history: deque/lista med senaste EAR-värdena i
            kronologisk ordning (äldst först, senast sist).
        threshold: EAR-värde under vilket ögat räknas som slutet.
        consec_frames: minsta antal sammanhängande frames under
            tröskeln för att räknas som en avsiktlig blink.

    Returns:
        bool: True om en blink precis avslutats (dipp + återhämtning
            observerad i slutet av historiken), annars False.
    """
    if len(ear_history) < consec_frames + 1:
        return False

    history = list(ear_history)
    latest = history[-1]

    if latest < threshold:
        # Ögat är fortfarande stängt — blinken är inte klar än.
        return False

    # Räkna hur många frames omedelbart före den senaste (öppna)
    # frame som låg under tröskeln.
    closed_run = 0
    for ear in reversed(history[:-1]):
        if ear < threshold:
            closed_run += 1
        else:
            break

    return closed_run >= consec_frames


def compute_frame_motion(prev_frame, curr_frame, face_bbox):
    """
    Beräknar ett enkelt rörelsemått mellan två på varandra följande
    frames, avgränsat till ansiktets bounding box.

    Metoden konverterar båda frames till gråskala, beskär till
    ansiktsregionen, och beräknar medelvärdet av pixelvis absolut
    differens. Ett stillbildsfoto som hålls upp mot kameran ger ett
    lågt, stabilt värde nära noll; en levande person ger naturlig
    mikrorörelse (andning, små huvudrörelser, ljusvariation) som
    höjer värdet något över tid.

    Begränsning till ansiktets bounding box är avsiktlig — annars
    kan bakgrundsrörelse (t.ex. någon som rör sig bakom personen)
    ge falska liveness-signaler.

    Args:
        prev_frame: föregående videoframe, BGR (numpy-array från cv2).
        curr_frame: aktuell videoframe, BGR (numpy-array från cv2).
        face_bbox: tuple (x, y, w, h) med ansiktets bounding box i
            pixelkoordinater, samma för båda frames.

    Returns:
        float: medelvärdet av den absoluta pixeldifferensen inom
            ansiktsregionen (0.0–255.0). Högre värde = mer rörelse.
    """
    x, y, w, h = face_bbox

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    prev_face = prev_gray[y:y + h, x:x + w]
    curr_face = curr_gray[y:y + h, x:x + w]

    diff = cv2.absdiff(prev_face, curr_face)
    motion_score = float(np.mean(diff))

    return motion_score


class LivenessDetector:
    """
    Stateful liveness-detektor som kombinerar blink- och rörelsesignal
    över tid för att avgöra om ansiktet framför kameran tillhör en
    levande person eller ett hållet foto.

    Kombinationslogik: blinkdetektion är den avgörande signalen för
    status "levande", eftersom ett statiskt foto (vårt uttalade
    spoofing-scope, se moduldocstring) aldrig kan blinka — bekräftat
    empiriskt vid kalibrering (se scripts/test_liveness.py-körningar):
    ett hållet foto gav noll blink-events under upprepade tester,
    medan en levande person konsekvent blinkade inom tidsfönstret.

    Rörelsesignalen beräknas och exponeras fortfarande (motion_history,
    motion_threshold) som kompletterande diagnostik, men används INTE
    som ett blockerande AND-krav: empirisk kalibrering visade att en
    legitim användare som sitter naturligt stilla ofta inte genererar
    tillräcklig rörelse för att nå tröskeln inom det korta rullande
    fönstret, vilket gav oacceptabelt hög andel falska avvisningar
    (BPCER) för helt legitima användare — en avvägning som inte var
    värd säkerhetsvinsten, eftersom blink-signalen ensam redan
    fullständigt avslöjar det deklarerade hotscenariot (statiskt foto).

    Attribut:
        ear_history: rullande buffer med senaste EAR-värdena
            (medelvärde av vänster/höger öga), används av detect_blink().
        blink_timestamps: tidsstämplar för detekterade blinkar inom
            det aktuella tidsfönstret.
        motion_history: rullande buffer med senaste rörelsemåtten
            från compute_frame_motion().
        prev_frame: föregående videoframe, för rörelseberäkning.
        frame_count: totalt antal frames som behandlats, används för
            att avgöra om tillräckligt underlag finns för ett beslut.
    """

    def __init__(
        self,
        ear_history_len=10,
        blink_window_seconds=6.0,
        motion_window_frames=15,
        motion_threshold=MOTION_THRESHOLD,
        min_frames_before_decision=15,
    ):
        self.ear_history = deque(maxlen=ear_history_len)
        self.blink_timestamps = deque()
        self.motion_history = deque(maxlen=motion_window_frames)
        self.prev_frame = None
        self.frame_count = 0

        self.blink_window_seconds = blink_window_seconds
        self.motion_threshold = motion_threshold
        self.min_frames_before_decision = min_frames_before_decision

    def update(self, frame, landmarks, face_bbox):
        """
        Matar in en ny frame och returnerar aktuell liveness-status.

        Args:
            frame: aktuell videoframe, BGR (numpy-array från cv2).
            landmarks: MediaPipe Face Mesh-landmärken för aktuell frame.
            face_bbox: tuple (x, y, w, h), ansiktets bounding box.

        Returns:
            str: en av "levande", "misstänkt_foto", "osäker_ännu".
        """
        self.frame_count += 1

        left_ear = compute_ear(landmarks, LEFT_EYE_INDICES)
        right_ear = compute_ear(landmarks, RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2.0
        self.ear_history.append(avg_ear)

        if detect_blink(self.ear_history):
            self.blink_timestamps.append(time.time())

        now = time.time()
        while self.blink_timestamps and now - self.blink_timestamps[0] > self.blink_window_seconds:
            self.blink_timestamps.popleft()

        if self.prev_frame is not None:
            motion = compute_frame_motion(self.prev_frame, frame, face_bbox)
            self.motion_history.append(motion)
        self.prev_frame = frame.copy()

        if self.frame_count < self.min_frames_before_decision:
            return "osäker_ännu"

        blink_ok = len(self.blink_timestamps) > 0
        motion_ok = (
            len(self.motion_history) > 0
            and (sum(self.motion_history) / len(self.motion_history)) > self.motion_threshold
        )

        return "levande" if blink_ok else "misstänkt_foto"

    def reset(self):
        """
        Återställer detektorns state. Används t.ex. när en ny person
        träder framför kameran i Streamlit-appen, för att inte blanda
        ihop signaler från olika personer/sessioner.
        """
        self.ear_history.clear()
        self.blink_timestamps.clear()
        self.motion_history.clear()
        self.prev_frame = None
        self.frame_count = 0