"""
tests/test_liveness.py

Enhetstester för liveness-detektionslogiken i src/liveness.py.

Testar rena funktioner (compute_ear, detect_blink, compute_frame_motion)
med syntetiska/konstruerade indata, samt LivenessDetector som helhet
med syntetiska landmärkessekvenser. Kräver ingen riktig webcam eller
MediaPipe-modell — det är poängen med att isolera logiken i liveness.py
från detektionen i face_detection.py, som redan är validerad manuellt
mot en riktig webcam (se scripts/test_liveness.py).
"""

import numpy as np
import pytest

from src.liveness import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    EAR_THRESHOLD,
    EAR_CONSEC_FRAMES,
    compute_ear,
    detect_blink,
    compute_frame_motion,
    LivenessDetector,
)


def make_eye_landmarks(indices, vertical_gap):
    """
    Bygger ett dummy-landmärkes-set där BÅDA ögonens 6 punkter vardera
    är placerade så att EAR blir förutsägbar utifrån vertical_gap.

    LivenessDetector.update() läser både LEFT_EYE_INDICES och
    RIGHT_EYE_INDICES och tar medelvärdet — därför fylls alltid båda
    ögonens index identiskt, oavsett vilket indices-argument som ges,
    så att inget öga lämnas som odefinierade nollkoordinater (vilket
    skulle ge division med noll i compute_ear()).

    Horisontellt avstånd (p1-p4) hålls konstant på 10.0 per öga. De
    vertikala punktparen (p2-p6, p3-p5) sätts till samma vertical_gap,
    vilket ger EAR = vertical_gap / 10.0 enligt formeln i compute_ear().

    Parameters
    ----------
    indices : list[int]
        Behålls som argument för läsbarhet i testfallen (vilket öga
        testet konceptuellt handlar om), men båda ögonen fylls alltid.
    vertical_gap : float
        Önskat vertikalt avstånd, styr det resulterande EAR-värdet.

    Returns
    -------
    np.ndarray
        Array med shape (max(alla index) + 1, 2).
    """
    all_indices = list(LEFT_EYE_INDICES) + list(RIGHT_EYE_INDICES)
    landmarks = np.zeros((max(all_indices) + 1, 2))

    for eye_indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
        p1, p2, p3, p4, p5, p6 = eye_indices
        landmarks[p1] = [0.0, 0.0]
        landmarks[p4] = [10.0, 0.0]
        landmarks[p2] = [3.0, vertical_gap / 2]
        landmarks[p6] = [3.0, -vertical_gap / 2]
        landmarks[p3] = [7.0, vertical_gap / 2]
        landmarks[p5] = [7.0, -vertical_gap / 2]

    return landmarks


class TestComputeEar:
    def test_open_eye_gives_higher_ear_than_closed_eye(self):
        open_eye = make_eye_landmarks(LEFT_EYE_INDICES, vertical_gap=6.0)
        closed_eye = make_eye_landmarks(LEFT_EYE_INDICES, vertical_gap=1.0)

        open_ear = compute_ear(open_eye, LEFT_EYE_INDICES)
        closed_ear = compute_ear(closed_eye, LEFT_EYE_INDICES)

        assert open_ear > closed_ear

    def test_ear_matches_expected_formula_result(self):
        landmarks = make_eye_landmarks(LEFT_EYE_INDICES, vertical_gap=4.0)
        ear = compute_ear(landmarks, LEFT_EYE_INDICES)

        # vertical_gap=4.0 ger vertikala avstånd 4.0 (två gånger),
        # horisontellt avstånd 10.0 -> EAR = (4.0 + 4.0) / (2 * 10.0) = 0.4
        assert ear == pytest.approx(0.4, abs=1e-6)


class TestDetectBlink:
    def test_dip_and_recovery_counts_as_blink(self):
        # öppet, öppet, slutet, slutet, öppet (2 sammanhängande under tröskeln)
        ear_history = [0.30, 0.30, 0.10, 0.10, 0.30]
        assert detect_blink(ear_history, threshold=EAR_THRESHOLD, consec_frames=2) is True

    def test_single_low_frame_is_not_a_blink(self):
        # bara 1 frame under tröskeln, kravet är 2 sammanhängande
        ear_history = [0.30, 0.30, 0.10, 0.30]
        assert detect_blink(ear_history, threshold=EAR_THRESHOLD, consec_frames=2) is False

    def test_eye_still_closed_is_not_yet_a_completed_blink(self):
        # senaste värdet är fortfarande under tröskeln -> blinken är inte klar
        ear_history = [0.30, 0.10, 0.10, 0.10]
        assert detect_blink(ear_history, threshold=EAR_THRESHOLD, consec_frames=2) is False

    def test_too_short_history_returns_false(self):
        ear_history = [0.30]
        assert detect_blink(ear_history, threshold=EAR_THRESHOLD, consec_frames=2) is False


class TestComputeFrameMotion:
    def test_identical_frames_give_zero_motion(self):
        frame = np.full((50, 50, 3), 128, dtype=np.uint8)
        bbox = (0, 0, 50, 50)

        motion = compute_frame_motion(frame, frame.copy(), bbox)

        assert motion == pytest.approx(0.0, abs=1e-6)

    def test_different_frames_give_positive_motion(self):
        prev_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        curr_frame = np.full((50, 50, 3), 255, dtype=np.uint8)
        bbox = (0, 0, 50, 50)

        motion = compute_frame_motion(prev_frame, curr_frame, bbox)

        assert motion > 0.0


class TestLivenessDetector:
    def _feed_stable_open_eyes(self, detector, frame, bbox, n_frames):
        open_eye_landmarks = make_eye_landmarks(LEFT_EYE_INDICES, vertical_gap=6.0)
        for _ in range(n_frames):
            detector.update(frame, open_eye_landmarks, bbox)

    def test_no_blink_gives_suspected_photo(self):
        detector = LivenessDetector(min_frames_before_decision=5)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        bbox = (0, 0, 50, 50)

        self._feed_stable_open_eyes(detector, frame, bbox, n_frames=10)
        status = detector.update(frame, make_eye_landmarks(LEFT_EYE_INDICES, 6.0), bbox)

        assert status == "misstänkt_foto"

    def test_blink_sequence_gives_alive(self):
        detector = LivenessDetector(min_frames_before_decision=5)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        bbox = (0, 0, 50, 50)

        open_eyes = make_eye_landmarks(LEFT_EYE_INDICES, vertical_gap=6.0)
        closed_eyes = make_eye_landmarks(LEFT_EYE_INDICES, vertical_gap=1.0)

        detector.update(frame, open_eyes, bbox)
        detector.update(frame, open_eyes, bbox)
        detector.update(frame, closed_eyes, bbox)
        detector.update(frame, closed_eyes, bbox)
        status = detector.update(frame, open_eyes, bbox)

        assert status == "levande"

    def test_reset_clears_all_state(self):
        detector = LivenessDetector(min_frames_before_decision=5)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        bbox = (0, 0, 50, 50)

        self._feed_stable_open_eyes(detector, frame, bbox, n_frames=10)
        detector.reset()

        assert len(detector.ear_history) == 0
        assert len(detector.blink_timestamps) == 0
        assert len(detector.motion_history) == 0
        assert detector.prev_frame is None
        assert detector.frame_count == 0