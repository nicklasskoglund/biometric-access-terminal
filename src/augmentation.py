"""Pixelnivå-augmentation av auktoriserade ansiktsbilder.

Denna modul ansvarar för att skapa syntetisk variation (rotation,
ljusstyrka/kontrast, spegling, skalning, brus) av ett litet antal
egna, redan beskurna ansiktsbilder — i syfte att skapa ett robustare
och större underlag för den positiva klassen (auktoriserad) i
åtkomstklassificeraren, se notebook 04.

Augmentationen appliceras på redan beskurna ansiktscrop (inte råa
webcam-bilder), och parametrarna är medvetet konservativa eftersom
aggressiva transformationer riskerar att förvränga ansiktsdrag som
ArcFace-embeddingen förlitar sig på.
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np


def create_augmentation_pipeline(seed: int | None = None) -> A.Compose:
    """Bygger en albumentations-pipeline för att skapa realistisk variation
    i redan beskurna, tajt paddade ansiktsbilder.

    Parametrarna är konservativt satta eftersom input redan är ett normaliserat
    ansiktscrop (se crop_face_with_padding i embeddings.py) — aggressiva
    transformationer (t.ex. CoarseDropout, RandomCrop, extrem rotation) riskerar
    att förstöra ansiktsdrag som ArcFace-embeddingen förlitar sig på, snarare
    än att simulera realistisk variation.

    Parameters
    ----------
    seed : int | None
        Slumpfrö för reproducerbarhet. None ger olika resultat vid varje körning.

    Returns
    -------
    A.Compose
        Sammansatt augmentationspipeline. Anropas som pipeline(image=arr)["image"].

    Notes
    -----
    Varje transform har sin egen sannolikhet (p), så genererade varianter får
    naturlig variation i VILKA transformationer som faktiskt appliceras —
    inte bara i deras styrka.
    """
    return A.Compose(
        [
            A.Rotate(limit=15, p=0.7, border_mode=0),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.7
            ),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), p=0.5),
            A.GaussNoise(std_range=(0.04, 0.11), p=0.3),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=(3, 5)),
                ],
                p=0.2,
            ),
        ],
        seed=seed,
    )
    
    
def augment_face_image(
    image: np.ndarray,
    pipeline: A.Compose,
    n_variants: int,
) -> list[np.ndarray]:
    """Genererar n_variants augmenterade varianter av en redan beskuren
    ansiktsbild.

    Parameters
    ----------
    image : np.ndarray
        Ett redan beskuret ansikte, RGB-format, som pipelinen appliceras på.
    pipeline : A.Compose
        En augmentationspipeline, se create_augmentation_pipeline().
    n_variants : int
        Antal varianter att generera från denna enskilda bild.

    Returns
    -------
    list[np.ndarray]
        n_variants augmenterade bilder, varje med oberoende slumpmässigt
        applicerade transformationer enligt pipelinens sannolikheter.
    """
    return [pipeline(image=image)["image"] for _ in range(n_variants)]


def augment_authorized_dataset(
    cropped_image_paths: list[Path],
    output_dir: Path,
    variants_per_image: int,
    seed: int | None = None,
) -> None:
    """Augmenterar samtliga beskurna auktoriserade ansiktsbilder och sparar
    resultatet till disk.

    Loopar över cropped_image_paths, genererar variants_per_image varianter
    per bild via augment_face_image(), och sparar varje variant som en egen
    JPEG-fil i output_dir. Filnamnsmönster:
    {originalfilnamn utan ändelse}_variant_{n:03d}.jpg.

    Parameters
    ----------
    cropped_image_paths : list[Path]
        Sökvägar till de redan beskurna källbilderna (RGB eller BGR på disk,
        läses via cv2.imread och konverteras till RGB för augmentation).
    output_dir : Path
        Mapp dit augmenterade varianter sparas. Skapas om den inte finns.
    variants_per_image : int
        Antal augmenterade varianter att generera per källbild.
    seed : int | None
        Slumpfrö vidarebefordrat till create_augmentation_pipeline(), för
        reproducerbarhet.

    Notes
    -----
    Bilder sparas via cv2.imwrite, vilket förväntar sig BGR - konvertering
    tillbaka från RGB sker internt i denna funktion.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = create_augmentation_pipeline(seed=seed)

    for image_path in cropped_image_paths:
        bgr_image = cv2.imread(str(image_path))
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        variants = augment_face_image(rgb_image, pipeline, variants_per_image)

        stem = image_path.stem
        for i, variant in enumerate(variants):
            variant_bgr = cv2.cvtColor(variant, cv2.COLOR_RGB2BGR)
            output_path = output_dir / f"{stem}_variant_{i:03d}.jpg"
            cv2.imwrite(str(output_path), variant_bgr)