"""
HUD-overlay: sci-fi-inspirerade ritfunktioner som appliceras direkt på
videoframes i video_frame_callback. Alla funktioner är avsedda att vara
billiga per frame (endast OpenCV-primitiver, inga tunga beräkningar).
"""

import time
import cv2
import numpy as np

# Sci-fi-palett (samma som i presentationen)
COLOR_TEAL = (163, 217, 0)      # #00D9A3 i BGR (OpenCV använder BGR, inte RGB)
COLOR_RED = (87, 71, 255)       # #FF4757 i BGR
COLOR_NAVY = (32, 18, 11)       # #0B1220 i BGR

SCAN_PERIOD_SECONDS = 2.0  # tid för en fullständig scanning-cykel (topp till botten)


def get_status_color(auth_status: str) -> tuple[int, int, int]:
    """
    Mappar auktoriseringsstatus till en BGR-färg, delad av scanning-linjen
    och hörnhakarna för en sammanhållen "target lock"-känsla.

    Args:
        auth_status: t.ex. "scanning", "authorized", "unauthorized",
            eller "liveness_failed".

    Returns:
        BGR-färgtuple.
    """
    if auth_status == "authorized":
        return COLOR_TEAL
    elif auth_status in ("unauthorized", "liveness_failed"):
        return COLOR_RED
    else:
        # "scanning" / okänt / väntar på ansikte
        return COLOR_TEAL
    
    
def get_status_visuals(auth_status: str) -> tuple[tuple[int, int, int], bool]:
    """
    Mappar auktoriseringsstatus till (färg, glow) för scanning-linjen och
    hörnhakarna, för en sammanhållen "target lock"-känsla.

    Args:
        auth_status: t.ex. "no_face", "scanning", "authorized",
            "unauthorized", eller "liveness_failed".

    Returns:
        Tuple av (BGR-färg, om glow ska ritas).
    """
    if auth_status == "authorized":
        return COLOR_TEAL, True
    elif auth_status in ("unauthorized", "liveness_failed"):
        return COLOR_RED, True
    elif auth_status == "error":
        return COLOR_RED, False
    else:
        # "scanning" / "no_face" / okänt — skarpa linjer, ingen glöd
        return COLOR_TEAL, False


def draw_scan_line(frame: np.ndarray, face_box: tuple[int, int, int, int], color: tuple[int, int, int] = COLOR_TEAL) -> np.ndarray:
    """
    Ritar en horisontell, tidsbaserad "scanning"-linje som glider vertikalt
    över det detekterade ansiktets bounding box.

    Args:
        frame: BGR-bild (frame.to_ndarray(format="bgr24") från streamlit-webrtc).
        face_box: (x, y, w, h) för ansiktets bounding box i pixelkoordinater.
        color: BGR-färg för scanning-linjen.

    Returns:
        Samma frame, med linjen inritad (ritas in-place men returneras ändå
        för bekväm kedjning).
    """
    x, y, w, h = face_box

    # Tidsbaserad fas 0.0 -> 1.0, oberoende av exakt frame-timing/framerate.
    phase = (time.time() % SCAN_PERIOD_SECONDS) / SCAN_PERIOD_SECONDS
    scan_y = int(y + phase * h)

    cv2.line(frame, (x, scan_y), (x + w, scan_y), color, thickness=2)

    return frame


def draw_corner_brackets(
    frame: np.ndarray,
    face_box: tuple[int, int, int, int],
    color: tuple[int, int, int] = COLOR_TEAL,
    bracket_length: int = 24,
    thickness: int = 3,
    glow: bool = False,
) -> np.ndarray:
    """
    Ritar fyra hörnhakar (L-formade linjer) runt ansiktets bounding box,
    i stil med ett "target lock"-sikte.

    Args:
        frame: BGR-bild.
        face_box: (x, y, w, h) för ansiktets bounding box i pixelkoordinater.
        color: BGR-färg för hakarna.
        bracket_length: hur långa de två linjerna i varje hörn är, i pixlar.
        thickness: linjetjocklek för den skarpa kärnlinjen.
        glow: om True, ritas en bredare, halvtransparent "glöd" under
            kärnlinjen (används för t.ex. auktoriserad-status).

    Returns:
        Samma frame, med hörnhakarna inritade.
    """
    x, y, w, h = face_box
    l = bracket_length

    corners = [
        (x, y, 1, 1),
        (x + w, y, -1, 1),
        (x, y + h, 1, -1),
        (x + w, y + h, -1, -1),
    ]

    if glow:
        overlay = frame.copy()
        glow_thickness = thickness + 6
        for cx, cy, dx, dy in corners:
            cv2.line(overlay, (cx, cy), (cx + dx * l, cy), color, glow_thickness)
            cv2.line(overlay, (cx, cy), (cx, cy + dy * l), color, glow_thickness)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)

    for cx, cy, dx, dy in corners:
        cv2.line(frame, (cx, cy), (cx + dx * l, cy), color, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy * l), color, thickness)

    return frame