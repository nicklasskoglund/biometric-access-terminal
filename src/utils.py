"""Delade hjälpfunktioner och typer som används av flera moduler i src/."""

from typing import NamedTuple


class BoundingBox(NamedTuple):
    """Ramverksoberoende representation av en ansikts-bounding box.

    Frikopplad från MediaPipes egna typer, så att moduler som embeddings.py
    inte behöver bero på mediapipe-paketet. Anroparen (t.ex. Streamlit-appen)
    ansvarar för att konvertera från MediaPipes bounding_box till denna typ.

    Attributes
    ----------
    origin_x : int
        Bounding boxens övre vänstra hörn, x-koordinat i pixlar.
    origin_y : int
        Bounding boxens övre vänstra hörn, y-koordinat i pixlar.
    width : int
        Bounding boxens bredd i pixlar.
    height : int
        Bounding boxens höjd i pixlar.
    """

    origin_x: int
    origin_y: int
    width: int
    height: int