"""Ansiktsembeddings via förtränad ArcFace-modell (DeepFace).

Detta modul ansvarar för att generera embedding-vektorer för ansiktsbilder,
antingen från redan beskurna dataset-bilder (wiki_crop) eller från
beskurna live-frames i webcam-pipelinen.
"""

import csv
from pathlib import Path

import numpy as np
from deepface import DeepFace
from tqdm import tqdm

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


def extract_embeddings_for_dataset(
    image_paths: list[str],
    model_name: str = "ArcFace",
    show_progress: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Extraherar embeddings för en lista av bildsökvägar, med felhantering.

    Loopar över samtliga bildsökvägar och extraherar en embedding per bild
    via get_face_embedding(). Bilder som av någon anledning inte kan
    bearbetas (korrupt fil, oläsbar bild, etc.) hoppas över och loggas,
    snarare än att avbryta hela körningen.

    Parameters
    ----------
    image_paths : list[str]
        Sökvägar till redan beskurna ansiktsbilder (t.ex. wiki_crop-bilder
        som passerat face_detected-filtreringen i preprocessing.py).
    model_name : str, default "ArcFace"
        Vilken förtränad DeepFace-modell som ska generera embeddingarna.
    show_progress : bool, default True
        Om True visas en tqdm-progressbar. Sätts till False vid anrop
        från extract_embeddings_chunked(), som istället visar progress
        på chunk-nivå.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Ett (N, 512)-formad array med embeddings, samt en lista med
        motsvarande bildsökvägar i samma ordning. Listan är kortare än
        image_paths om någon bild misslyckades.

    Notes
    -----
    Misslyckade bilder skrivs ut som varningar men avbryter inte körningen,
    eftersom ett enskilt korrupt fall inte ska förlora resten av batchen.
    """
    embeddings: list[np.ndarray] = []
    successful_paths: list[str] = []

    iterator = (
        tqdm(image_paths, desc="Extraherar embeddings")
        if show_progress
        else image_paths
    )

    for path in iterator:
        try:
            embedding = get_face_embedding(path, model_name=model_name)
            embeddings.append(embedding)
            successful_paths.append(path)
        except Exception as error:
            print(f"Varning: hoppar över {path} ({error})")

    return np.array(embeddings), successful_paths


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


def extract_embeddings_chunked(
    image_paths: list[str],
    output_dir: str,
    chunk_size: int = 750,
    model_name: str = "ArcFace",
) -> None:
    """Extraherar embeddings i chunkar och sparar löpande till disk.

    Delar upp image_paths i chunkar om chunk_size och bearbetar en chunk
    i taget via extract_embeddings_for_dataset(). Varje chunk sparas som
    en .npy-fil (embeddings) tillsammans med en matchande .csv-fil
    (motsvarande bildsökvägar, i samma ordning som embedding-raderna).

    Stödjer resume: om en chunks .npy-fil redan finns i output_dir hoppas
    den över, så en avbruten körning kan startas om utan att göra om
    redan slutfört arbete. Bilder som misslyckas i en chunk loggas till
    output_dir/failed_images.csv istället för att ge tysta luckor eller
    platshållarvärden i embedding-arrayerna.

    Parameters
    ----------
    image_paths : list[str]
        Samtliga bildsökvägar som ska bearbetas, i den ordning de
        tilldelas chunk-index.
    output_dir : str
        Mapp där chunk-filer och failed_images.csv sparas. Måste redan
        finnas.
    chunk_size : int, default 750
        Antal bilder per chunk.
    model_name : str, default "ArcFace"
        Vidarebefordras oförändrat till extract_embeddings_for_dataset().

    Notes
    -----
    Funktionen har inget returvärde - resultatet läses från disk i
    notebooken vid sammanslagningssteget (en efterföljande cell slår
    ihop samtliga chunk_*.npy/.csv-par till embeddings.npy och
    embeddings_index.csv).
    """
    output_path = Path(output_dir)
    chunks = [
        image_paths[i : i + chunk_size]
        for i in range(0, len(image_paths), chunk_size)
    ]

    for chunk_index, chunk_paths in enumerate(tqdm(chunks, desc="Chunks")):
        embeddings_file = output_path / f"embeddings_chunk_{chunk_index:04d}.npy"
        paths_file = output_path / f"paths_chunk_{chunk_index:04d}.csv"

        if embeddings_file.exists() and paths_file.exists():
            print(f"Chunk {chunk_index:04d} finns redan, hoppar över.")
            continue

        chunk_embeddings, successful_paths = extract_embeddings_for_dataset(
            chunk_paths,
            model_name=model_name,
            show_progress=False,
        )
        np.save(embeddings_file, chunk_embeddings)

        with open(paths_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path"])
            writer.writerows([[p] for p in successful_paths])

        failed_paths = set(chunk_paths) - set(successful_paths)
        if failed_paths:
            _log_failed_images(output_path / "failed_images.csv", failed_paths)


def _log_failed_images(log_file: Path, failed_paths: set[str]) -> None:
    """Loggar misslyckade bildsökvägar till en gemensam CSV-fil.

    Skriver header endast om filen inte redan finns, så anrop från flera
    chunkar kan ackumulera i samma fil över en hel körning (inklusive
    vid resume efter avbrott).

    Parameters
    ----------
    log_file : Path
        Sökväg till failed_images.csv.
    failed_paths : set[str]
        Bildsökvägar som misslyckades i den aktuella chunken.
    """
    file_exists = log_file.exists()
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["image_path"])
        writer.writerows([[p] for p in sorted(failed_paths)])