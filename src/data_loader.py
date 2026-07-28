"""Data loading utilities for the IMDB-WIKI (WIKI subset) dataset.

This module is only responsible for reading the raw .mat metadata file
into a pandas DataFrame, with no cleaning or transformation. Cleaning
(handling missing values, converting MATLAB date numbers to real dates,
encoding categorical variables) belongs in `preprocessing.py`, per the
project's separation between raw ingestion and data preparation.
"""

from pathlib import Path
from typing import Union

import pandas as pd
from scipy.io import loadmat


def load_wiki_mat(mat_path: Union[str, Path]) -> pd.DataFrame:
    """Load the WIKI subset metadata from wiki.mat into a flat DataFrame.

    Parameters
    ----------
    mat_path : str or Path
        Path to the wiki.mat file (e.g. data/raw/wiki_crop/wiki.mat).

    Returns
    -------
    pd.DataFrame
        One row per face image, with the following raw columns:

        - full_path : str, relative path to the cropped face image
        - dob_matlab : int, date of birth as a MATLAB serial date number
          (NOT yet converted to a real date — see preprocessing.py)
        - photo_taken : int, year the photo was taken
        - gender : float, 1.0 = male, 0.0 = female, NaN = unknown
        - face_score : float, face detector confidence;
          -inf means no face was detected
        - second_face_score : float, confidence of a second face in the
          image if one was detected, else NaN
        - name : str or None, celebrity name as listed on Wikipedia
        - face_location : np.ndarray, [x1, y1, x2, y2] bounding box

    Notes
    -----
    No rows are dropped and no missing values are imputed here. This
    function only reshapes the MATLAB struct-of-arrays into a tidy
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