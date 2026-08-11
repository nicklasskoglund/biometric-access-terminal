"""
Ansiktsdetektion för live-pipelinen via webcam.

Denna modul hanterar detektion av ansikten i realtid från webcam-frames med
hjälp av MediaPipes förtränade BlazeFace-modell (Tasks API). Detta är separat
från face_score-flaggan i wiki-metadatan (se preprocessing.py), som är ett
kvalitetsmått för det statiska IMDB-WIKI-datasetet snarare än detektion på
levande frames.

Notes
-----
Kräver en lokalt nedladdad modellfil, se README.md för instruktioner.
"""

from pathlib import Path

import mediapipe as mp
import threading
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = Path("models/pretrained/face_detector/blaze_face_short_range.tflite")


class DetectionResultStore:
    """
    Trådsäker lagring av det senaste ansiktsdetektionsresultatet.

    MediaPipes LIVE_STREAM-läge levererar resultat asynkront via en
    callback som körs på en separat tråd. Denna klass fungerar som en
    brygga mellan callbacken och huvudtråden (t.ex. Streamlit-loopen),
    genom att skydda läsning/skrivning med ett lås.

    Notes
    -----
    Endast det senaste resultatet sparas — äldre resultat skrivs över.
    Detta är avsiktligt: HUD:en ska alltid visa aktuellt tillstånd, inte
    en historik av tidigare frames.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_result: mp_vision.FaceDetectorResult | None = None

    def update(self, result: mp_vision.FaceDetectorResult) -> None:
        """
        Sparar ett nytt detektionsresultat, skriver över det tidigare.

        Parameters
        ----------
        result : mediapipe.tasks.python.vision.FaceDetectorResult
            Resultatet från MediaPipes detect_async-callback.
        """
        with self._lock:
            self._latest_result = result

    def get(self) -> mp_vision.FaceDetectorResult | None:
        """
        Hämtar det senast sparade detektionsresultatet.

        Returns
        -------
        mediapipe.tasks.python.vision.FaceDetectorResult | None
            Senaste resultatet, eller None om ingen detektion körts än.
        """
        with self._lock:
            return self._latest_result


def create_face_detector(result_store: DetectionResultStore) -> mp_vision.FaceDetector:
    """
    Skapar och initierar en MediaPipe FaceDetector för livestream-läge.

    Parameters
    ----------
    result_store : DetectionResultStore
        Container dit detektionsresultat sparas asynkront. Skapas av
        anroparen och delas med resten av pipelinen för läsning.

    Returns
    -------
    mediapipe.tasks.python.vision.FaceDetector
        Initierad detektor redo att ta emot frames via detect_async().

    Raises
    ------
    FileNotFoundError
        Om modellfilen inte hittas på MODEL_PATH. Se README.md för
        nedladdningsinstruktioner.

    Notes
    -----
    Detektorn skapas i LIVE_STREAM-läge. Resultat levereras asynkront
    till result_store via en intern callback, snarare än som returvärde
    från detect_async().
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modellfil saknas: {MODEL_PATH}. Se README.md under "
            "'Förtränad modell för ansiktsdetektion' för nedladdningsinstruktioner."
        )

    def _on_result(
        result: mp_vision.FaceDetectorResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        result_store.update(result)

    options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        result_callback=_on_result,
    )
    return mp_vision.FaceDetector.create_from_options(options)