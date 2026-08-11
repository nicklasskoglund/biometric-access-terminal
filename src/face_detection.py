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

import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
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
        self._latest_result: "mp_vision.FaceDetectorResult | None" = None  # type: ignore[reportInvalidTypeForm]

    def update(self, result: "mp_vision.FaceDetectorResult") -> None:  # type: ignore[reportInvalidTypeForm]
        """
        Sparar ett nytt detektionsresultat, skriver över det tidigare.

        Parameters
        ----------
        result : mediapipe.tasks.python.vision.FaceDetectorResult
            Resultatet från MediaPipes detect_async-callback.
        """
        with self._lock:
            self._latest_result = result

    def get(self) -> "mp_vision.FaceDetectorResult | None":  # type: ignore[reportInvalidTypeForm]
        """
        Hämtar det senast sparade detektionsresultatet.

        Returns
        -------
        mediapipe.tasks.python.vision.FaceDetectorResult | None
            Senaste resultatet, eller None om ingen detektion körts än.
        """
        with self._lock:
            return self._latest_result


def create_face_detector(result_store: DetectionResultStore) -> "mp_vision.FaceDetector":  # type: ignore[reportInvalidTypeForm]
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
        result: "mp_vision.FaceDetectorResult",  # type: ignore[reportInvalidTypeForm]
        output_image: "mp.Image",
        timestamp_ms: int,
    ) -> None:
        result_store.update(result)

    options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        result_callback=_on_result,
    )
    return mp_vision.FaceDetector.create_from_options(options)


def _bgr_frame_to_mp_image(frame: np.ndarray) -> mp.Image:
    """
    Konverterar en OpenCV-frame (BGR) till MediaPipes bildformat (RGB).

    Parameters
    ----------
    frame : numpy.ndarray
        En enskild frame från cv2.VideoCapture, i BGR-format.

    Returns
    -------
    mediapipe.Image
        Frame konverterad till RGB och inpackad i MediaPipes bildobjekt.

    Notes
    -----
    OpenCV läser frames i BGR-ordning som standard, medan MediaPipe
    förväntar sig RGB. Konverteringen är nödvändig för korrekt detektion.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def run_webcam_face_detection(
    detector: "mp_vision.FaceDetector",  # type: ignore[reportInvalidTypeForm]
    result_store: DetectionResultStore,
    camera_index: int = 0,
) -> None:
    """
    Kör en manuell testloop: läser webcam-frames, kör detektion, visar resultat.

    Öppnar ett OpenCV-fönster som ritar ut ramar runt detekterade ansikten
    i realtid. Avsedd för visuell verifiering under utveckling, inte för
    slutlig produktionsanvändning (Streamlit-integrationen byggs separat).

    Parameters
    ----------
    detector : mediapipe.tasks.python.vision.FaceDetector
        En initierad detektor, se create_face_detector().
    result_store : DetectionResultStore
        Samma result_store-instans som användes vid skapandet av detector.
    camera_index : int, default=0
        Index för vilken webcam som ska användas.

    Notes
    -----
    Tryck 'q' i fönstret för att avsluta loopen.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kunde inte öppna webcam med index {camera_index}.")

    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            mp_image = _bgr_frame_to_mp_image(frame)
            timestamp_ms = int((time.time() - start_time) * 1000)
            detector.detect_async(mp_image, timestamp_ms)

            result = result_store.get()
            if result is not None:
                for detection in result.detections:
                    bbox = detection.bounding_box
                    cv2.rectangle(
                        frame,
                        (bbox.origin_x, bbox.origin_y),
                        (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                        color=(0, 255, 0),
                        thickness=2,
                    )

            cv2.imshow("Face Detection Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()