"""
Tillståndsmaskin för HUD-scanningsflödet: styr övergångar mellan att vänta
på ett ansikte, stabilisera, aktivt scanna (liveness + auktorisering), och
visa resultat. Helt fristående från OpenCV/MediaPipe/Streamlit för att
kunna testas isolerat med simulerad tid.
"""

import time
from enum import Enum, auto


class ScanState(Enum):
    IDLE = auto()
    STABILIZING = auto()
    SCANNING = auto()
    RESULT_GRANTED = auto()
    WELCOME = auto()
    RESULT_DENIED = auto()


# Tidsparametrar (sekunder)
STABILIZE_DURATION = 0.5
SCAN_TIMEOUT = 4.0
GRANTED_DISPLAY_DURATION = 1.5
DENIED_DISPLAY_DURATION = 3.0


class ScanStateMachine:
    """
    Driver övergångar mellan ScanState-värden baserat på inmatning från
    ansiktsdetektion, liveness-detektion och auktoriseringsklassificering.

    Anropas en gång per videoframe via update(), med aktuell information
    om huruvida ett ansikte syns, om en blinkning upptäckts, och om ett
    klassificeringsresultat finns tillgängligt från embedding_worker.
    """

    def __init__(self):
        self._state = ScanState.IDLE
        self._state_entered_at = time.time()

    @property
    def state(self) -> ScanState:
        return self._state

    def _transition_to(self, new_state: ScanState):
        self._state = new_state
        self._state_entered_at = time.time()

    def _time_in_state(self) -> float:
        return time.time() - self._state_entered_at

    def update(self, face_detected: bool, liveness_status: str, auth_result: str | None):
        """
        Kör en tillståndsövergång baserat på aktuell input.

        Args:
            face_detected: om ett ansikte syns i den fasta scanningsramen
                just nu (denna frame).
            liveness_status: returvärdet från LivenessDetector.update(),
                ett av "levande", "misstänkt_foto", "osäker_ännu". Redan
                ackumulerat över ett rullande tidsfönster internt i
                LivenessDetector, ingen egen blink-bokföring behövs här.
            auth_result: None om embedding_worker inte hunnit klassificera
                än, annars "authorized" eller "unauthorized".
        """
        if self._state == ScanState.IDLE:
            if face_detected:
                self._transition_to(ScanState.STABILIZING)

        elif self._state == ScanState.STABILIZING:
            if not face_detected:
                self._transition_to(ScanState.IDLE)
            elif self._time_in_state() >= STABILIZE_DURATION:
                self._transition_to(ScanState.SCANNING)

        elif self._state == ScanState.SCANNING:
            if not face_detected:
                self._transition_to(ScanState.IDLE)
            elif liveness_status == "levande" and auth_result == "authorized":
                self._transition_to(ScanState.RESULT_GRANTED)
            elif liveness_status == "levande" and auth_result == "unauthorized":
                self._transition_to(ScanState.RESULT_DENIED)
            elif self._time_in_state() >= SCAN_TIMEOUT:
                # Antingen aldrig blev "levande" (misstänkt foto), eller
                # klassificeringen aldrig hann bli klar - i båda fallen
                # nekas åtkomst.
                self._transition_to(ScanState.RESULT_DENIED)

        elif self._state == ScanState.RESULT_GRANTED:
            if self._time_in_state() >= GRANTED_DISPLAY_DURATION:
                self._transition_to(ScanState.WELCOME)

        elif self._state == ScanState.RESULT_DENIED:
            if self._time_in_state() >= DENIED_DISPLAY_DURATION:
                self._transition_to(ScanState.IDLE)

        elif self._state == ScanState.WELCOME:
            pass  # Stannar kvar tills vidare, ingen auto-återgång än.

    def display_text(self) -> str:
        """Text att visa i HUD:en för det aktuella tillståndet."""
        return {
            ScanState.IDLE: "Position your face within the frame",
            ScanState.STABILIZING: "",
            ScanState.SCANNING: "Hold still and blink",
            ScanState.RESULT_GRANTED: "ACCESS GRANTED",
            ScanState.RESULT_DENIED: "ACCESS DENIED",
            ScanState.WELCOME: "Welcome",
        }[self._state]