"""Ansiktsembeddings via förtränad ArcFace-modell (DeepFace).

Detta modul ansvarar för att generera embedding-vektorer för ansiktsbilder,
antingen från redan beskurna dataset-bilder (wiki_crop) eller från
beskurna live-frames i webcam-pipelinen.
"""

import numpy as np
from deepface import DeepFace
from utils import BoundingBox


def get_face_embedding(
    image: str | np.ndarray,
    model_name: str = "ArcFace",
) -> np.ndarray:
    """Extraherar en ansiktsembedding för en redan beskuren ansiktsbild.

    Använder en förtränad DeepFace-modell för att representera ett ansikte
    som en fast-dimensionell embedding-vektor. Funktionen förutsätter att
    bilden redan är beskuren till ett ansikte (t.ex. wiki_crop-bilder eller
    en beskuren live-frame från `crop_face_with_padding()`), och kör därför
    DeepFace utan egen ansiktsdetektion (`detector_backend="skip"`).

    Parameters
    ----------
    image : str eller np.ndarray
        Antingen en filsökväg till en beskuren ansiktsbild, eller en
        redan inläst bild som numpy-array i BGR-format (t.ex. från OpenCV).
    model_name : str, default "ArcFace"
        Vilken förtränad DeepFace-modell som ska generera embeddingen.

    Returns
    -------
    np.ndarray
        Embedding-vektorn för ansiktet, som en endimensionell numpy-array.

    Notes
    -----
    `detector_backend="skip"` innebär att DeepFace litar på att hela bilden
    redan är ett ansikte och hoppar över sin egen ansiktsdetektion. Detta är
    ett medvetet designval eftersom vi redan hanterar ansiktsdetektion själva
    (via `face_detection.py` för live-data, och via wiki_crop-datasetets
    egen förbehandling för träningsdata).
    """
    result = DeepFace.represent(
        img_path=image,
        model_name=model_name,
        detector_backend="skip",
    )

    if not result:
        raise ValueError("DeepFace returnerade ingen embedding för bilden.")

    return np.array(result[0]["embedding"])


def crop_face_with_padding(
    frame: np.ndarray,
    bbox: BoundingBox,
    padding_ratio: float = 0.3,
) -> np.ndarray:
    """Beskär ett ansikte ur en frame med extra marginal runt bounding boxen.

    MediaPipes BlazeFace-detektor ger en tight bounding box fokuserad på
    kärnansiktsdrag (ögon/näsa/mun), snarare än hela huvudet inklusive
    panna och hjässa. Embedding-modeller som ArcFace förväntar sig ett
    ansikte som fyller större delen av bilden, men en för tight beskärning
    riskerar att tappa information som modellen behöver. Denna funktion
    lägger därför på procentuell padding runt boxens kanter innan
    beskärning, vilket är standardpraxis i ansiktsigenkänningspipelines.

    Parameters
    ----------
    frame : np.ndarray
        Hela frame (t.ex. från webcam) i BGR-format, som bounding boxen
        avser koordinater inom.
    bbox : BoundingBox
        Ansiktets bounding box enligt MediaPipes detektion, konverterad
        till projektets ramverksoberoende typ (se utils.py).
    padding_ratio : float, default=0.3
        Hur mycket padding som läggs på i förhållande till boxens
        bredd/höjd. 0.3 innebär 30% extra marginal på varje sida.

    Returns
    -------
    np.ndarray
        Den beskurna ansiktsregionen, i samma format (BGR) som frame.

    Notes
    -----
    Paddingen klipps mot framens faktiska gränser, så ett ansikte nära
    bildkanten beskärs inte utanför bilden.
    """
    frame_height, frame_width = frame.shape[:2]

    pad_x = int(bbox.width * padding_ratio)
    pad_y = int(bbox.height * padding_ratio)

    x1 = max(bbox.origin_x - pad_x, 0)
    y1 = max(bbox.origin_y - pad_y, 0)
    x2 = min(bbox.origin_x + bbox.width + pad_x, frame_width)
    y2 = min(bbox.origin_y + bbox.height + pad_y, frame_height)

    return frame[y1:y2, x1:x2]