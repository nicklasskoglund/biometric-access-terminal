"""Inläsningsfunktioner för IMDB-WIKI-datasetet (WIKI-delen).

Denna modul ansvarar enbart för att läsa den råa .mat-metadatafilen till
en pandas DataFrame, utan någon rensning eller transformation. Rensning
(hantering av saknade värden, konvertering av MATLAB-datum till riktiga
datum, kodning av kategoriska variabler) hör hemma i `preprocessing.py`,
enligt projektets separation mellan rå inläsning och databeredelse.
"""

from pathlib import Path
from typing import Union

import pandas as pd
from scipy.io import loadmat


def load_wiki_mat(mat_path: Union[str, Path]) -> pd.DataFrame:
    """Läs in WIKI-delens metadata från wiki.mat till en platt DataFrame.

    Parameters
    ----------
    mat_path : str eller Path
        Sökväg till wiki.mat-filen (t.ex. data/raw/wiki_crop/wiki.mat).

    Returns
    -------
    pd.DataFrame
        En rad per ansiktsbild, med följande råa kolumner:

        - full_path : str, relativ sökväg till det beskurna ansiktsfotot
        - dob_matlab : int, födelsedatum som ett MATLAB serial date number
          (ÄNNU EJ konverterat till ett riktigt datum — se preprocessing.py)
        - photo_taken : int, året fotot togs
        - gender : float, 1.0 = man, 0.0 = kvinna, NaN = okänt
        - face_score : float, ansiktsdetektorns konfidensvärde;
          -inf betyder att inget ansikte detekterades
        - second_face_score : float, konfidensvärde för ett eventuellt
          andra ansikte i bilden, annars NaN
        - name : str eller None, celebritetens namn enligt Wikipedia
        - face_location : np.ndarray, [x1, y1, x2, y2] boundingbox

    Notes
    -----
    Inga rader tas bort och inga saknade värden imputeras här. Funktionen
    formar enbart om MATLAB-structen (struct-of-arrays) till en tidy
    DataFrame.
    """
    mat_path = Path(mat_path)
    mat = loadmat(mat_path)
    wiki = mat["wiki"][0, 0]

    full_path = [p[0] for p in wiki["full_path"][0]]
    name = [n[0] if n.size > 0 else None for n in wiki["name"][0]]
    face_location = [fl[0] for fl in wiki["face_location"][0]]

    df = pd.DataFrame(
        {
            "full_path": full_path,
            "dob_matlab": wiki["dob"][0],
            "photo_taken": wiki["photo_taken"][0],
            "gender": wiki["gender"][0],
            "face_score": wiki["face_score"][0],
            "second_face_score": wiki["second_face_score"][0],
            "name": name,
            "face_location": face_location,
        }
    )

    return df