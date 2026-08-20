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


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pretrained" / "face_detector" / "blaze_face_short_range.tflite"
MESH_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pretrained" / "face_mesh" / "face_landmarker.task"


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


def create_face_detector(
    result_store: DetectionResultStore,
    min_detection_confidence: float = 0.75,
) -> "mp_vision.FaceDetector":  # type: ignore[reportInvalidTypeForm]
    """
    Skapar och initierar en MediaPipe FaceDetector för livestream-läge.

    Parameters
    ----------
    result_store : DetectionResultStore
        Container dit detektionsresultat sparas asynkront. Skapas av
        anroparen och delas med resten av pipelinen för läsning.
    min_detection_confidence : float, default=0.75
        Lägsta konfidensnivå (0.0–1.0) för att en detektion ska
        rapporteras. Höjt från MediaPipes standardvärde (0.5) efter
        manuell testning: verkligt ansikte gav 0.90–0.95, medan en
        hudfärgad tatuering (falsk positiv) gav 0.50–0.64. 0.75 ger
        marginal mot båda observerade intervallen.

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
        min_detection_confidence=min_detection_confidence,
        result_callback=_on_result,
    )
    return mp_vision.FaceDetector.create_from_options(options)


def create_face_detector_for_images(
    min_detection_confidence: float = 0.75,
) -> "mp_vision.FaceDetector":  # type: ignore[reportInvalidTypeForm]
    """
    Skapar en MediaPipe FaceDetector för synkron bearbetning av fristående
    stillbilder (IMAGE-läge), till skillnad från create_face_detector()
    som är konfigurerad för asynkron livestream från webcam.

    Parameters
    ----------
    min_detection_confidence : float, default=0.75
        Se create_face_detector() för motivering av standardvärdet.

    Returns
    -------
    mediapipe.tasks.python.vision.FaceDetector
        Initierad detektor. Anropas synkront via detect(), inte detect_async().

    Raises
    ------
    FileNotFoundError
        Om modellfilen inte hittas på MODEL_PATH.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modellfil saknas: {MODEL_PATH}. Se README.md under "
            "'Förtränad modell för ansiktsdetektion' för nedladdningsinstruktioner."
        )

    options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_detection_confidence=min_detection_confidence,
    )
    return mp_vision.FaceDetector.create_from_options(options)


def create_face_mesh_detector(
    num_faces: int = 1,
) -> "mp_vision.FaceLandmarker":  # type: ignore[reportInvalidTypeForm]
    """
    Skapar en MediaPipe FaceLandmarker för synkron landmärkesdetektion
    på fristående frames (IMAGE-läge).

    Till skillnad från create_face_detector_for_images(), som ger en
    grov bounding box via BlazeFace, ger FaceLandmarker 478 detaljerade
    ansiktslandmärken per detekterat ansikte — nödvändigt underlag för
    EAR-beräkning i liveness.py.

    Parameters
    ----------
    num_faces : int, default=1
        Max antal ansikten som detekteras samtidigt. 1 är tillräckligt
        och snabbast för denna access-terminal, där endast en person
        förväntas stå framför kameran åt gången.

    Returns
    -------
    mediapipe.tasks.python.vision.FaceLandmarker
        Initierad landmärkesdetektor. Anropas synkront via detect(),
        på samma sätt som FaceDetector i IMAGE-läge.

    Raises
    ------
    FileNotFoundError
        Om modellfilen inte hittas på MESH_MODEL_PATH. Se README.md
        för nedladdningsinstruktioner.

    Notes
    -----
    Detta är en annan MediaPipe-modell (face_landmarker.task) än den
    som används för bounding box-detektion (blaze_face_short_range.tflite)
    och laddas ner separat.
    """
    if not MESH_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modellfil saknas: {MESH_MODEL_PATH}. Se README.md under "
            "'Förtränad modell för ansiktslandmärken' för nedladdningsinstruktioner."
        )

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MESH_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=num_faces,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def landmarks_to_pixel_array(
    face_landmarks: list,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """
    Konverterar MediaPipes normaliserade landmärken ([0, 1]) till
    pixelkoordinater för en given frames dimensioner.

    MediaPipe Face Mesh ger landmärkeskoordinater normaliserade separat
    för x/y relativt bildens bredd/höjd. Om dessa används direkt i EAR-
    beräkningen (som jämför horisontella och vertikala avstånd) skulle
    resultatet bli skevt för icke-kvadratiska frames, eftersom x och y
    då skalas olika. Konvertering till en gemensam pixelrymd är därför
    ett nödvändigt steg innan landmärkena skickas till liveness.py.

    Parameters
    ----------
    face_landmarks : list
        Lista med normaliserade landmärkesobjekt (med .x, .y-attribut)
        för ETT ansikte, t.ex. result.face_landmarks[0] från en
        FaceLandmarkerResult.
    frame_width : int
        Framens bredd i pixlar.
    frame_height : int
        Framens höjd i pixlar.

    Returns
    -------
    np.ndarray
        Array med shape (478, 2), pixelkoordinater [x, y] per landmärke,
        i samma ordning som indata.
    """
    return np.array(
        [(lm.x * frame_width, lm.y * frame_height) for lm in face_landmarks]
    )


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
                    confidence = detection.categories[0].score
                    cv2.putText(
                        frame,
                        f"{confidence:.2f}",
                        (bbox.origin_x, bbox.origin_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color=(0, 255, 0),
                        thickness=2,
                    )

            cv2.imshow("Face Detection Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()