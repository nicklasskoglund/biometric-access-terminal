"""
preprocessing.py

Funktioner för att rensa och berika den råa WIKI-metadata-DataFrame som
produceras av data_loader.load_wiki_mat(). Innehåller åldersberäkning,
hantering av saknade värden och filtrering av orimliga datapunkter.

Denna modul muterar aldrig indata in-place; alla funktioner returnerar
en ny/kopierad DataFrame.
"""


from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def matlab_datenum_to_datetime(matlab_datenum: float) -> datetime:
    """
    Konverterar ett MATLAB serial date number till en Python datetime.

    Parameters
    ----------
    matlab_datenum : float
        MATLAB datenum-värde (dagar sedan år 0, dag 1 = 0001-01-01).

    Returns
    -------
    datetime
        Motsvarande Python datetime-objekt.
    """
    return datetime.fromordinal(int(matlab_datenum)) \
        + timedelta(days=matlab_datenum % 1) \
        - timedelta(days=366)


def compute_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Beräknar en 'age'-kolumn utifrån dob_matlab och photo_taken.

    Ålder approximeras som photo_taken (fotoåret) minus födelseåret,
    extraherat från dob_matlab via matlab_datenum_to_datetime.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame som innehåller kolumnerna 'dob_matlab' och 'photo_taken'.

    Returns
    -------
    pd.DataFrame
        Kopia av df med en tillagd 'age'-kolumn (float, NaN där dob_matlab saknas).

    Notes
    -----
    Ålder beräknas enbart utifrån födelseår, inte exakt födelsedatum,
    eftersom photo_taken bara är ett årtal och därmed inte tillåter högre
    precision. Vissa rader kan ge orimliga eller negativa åldrar på grund
    av kända fel i IMDB-WIKI:s namn-till-person-matchning (se steg 3:
    filtrering av orimliga åldrar). Denna funktion filtrerar inte bort
    sådana rader — det hanteras separat.
    """
    df = df.copy()
    birth_year = df["dob_matlab"].apply(
        lambda x: matlab_datenum_to_datetime(x).year if pd.notna(x) else pd.NA
    )
    df["age"] = df["photo_taken"] - birth_year
    return df


def flag_face_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flaggar rader där ingen ansikte detekterades i bilden.

    face_score == -inf betyder att ansiktsdetektorn inte hittade något
    ansikte alls i bilden. Detta är inte "saknad data" i statistisk
    mening utan en kvalitetsindikator från detektorn, och hanteras
    därför separat från t.ex. gender/name-brister.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame som innehåller kolumnen 'face_score'.

    Returns
    -------
    pd.DataFrame
        Kopia av df med en tillagd boolesk kolumn 'face_detected'
        (False där face_score == -inf, annars True).

    Notes
    -----
    Denna funktion tar INTE bort några rader. Rader utan detekterat
    ansikte är oanvändbara för embedding-extraktion längre fram i
    pipelinen, men filtreringen görs medvetet separat (se
    filter_valid_faces) så att bortfallet kan redovisas och
    visualiseras i EDA innan det faktiskt sker.
    """
    df = df.copy()
    df["face_detected"] = df["face_score"] != -np.inf
    return df


def filter_valid_faces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtrerar bort rader utan detekterat ansikte.

    Kräver att flag_face_detection() redan har körts på df, så att
    kolumnen 'face_detected' finns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame som innehåller den booleska kolumnen 'face_detected'.

    Returns
    -------
    pd.DataFrame
        Kopia av df där endast rader med face_detected == True behålls,
        med index återställt.

    Raises
    ------
    KeyError
        Om kolumnen 'face_detected' inte finns i df.

    Notes
    -----
    Denna funktion anropas medvetet separat från flag_face_detection,
    och först på den punkt i pipelinen där rader utan ansikte faktiskt
    blir oanvändbara (t.ex. precis innan embedding-extraktion), inte
    redan i den generella dataförberedelsen.
    """
    if "face_detected" not in df.columns:
        raise KeyError(
            "Kolumnen 'face_detected' saknas — kör flag_face_detection() först."
        )
    return df[df["face_detected"]].reset_index(drop=True)