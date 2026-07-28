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