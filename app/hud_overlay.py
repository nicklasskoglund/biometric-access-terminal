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


def get_guide_frame(frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    """
    Beräknar en fast, centrerad guide-ram där användaren ska positionera
    sitt ansikte. Ramen är en ren visuell referens - den faktiska
    ansiktsbeskärningen till embedding-modellen använder fortfarande
    BlazeFace:s riktiga, dynamiskt detekterade bounding box.

    En fast position (i motsats till att följa den riktiga bbox:en)
    eliminerar det jitter som annars uppstår i hörnhakarna eftersom
    BlazeFace:s box varierar något mellan frames även vid stillasittande.

    Args:
        frame_width: bildens bredd i pixlar.
        frame_height: bildens höjd i pixlar.

    Returns:
        (x, y, w, h) för guide-ramen.
    """
    w = int(frame_width * 0.35)
    h = int(frame_height * 0.55)
    x = (frame_width - w) // 2
    y = int(frame_height * 0.15)
    return (x, y, w, h)


def draw_hud_text(
    frame: np.ndarray,
    text: str,
    guide_box: tuple[int, int, int, int],
    color: tuple[int, int, int] = COLOR_TEAL,
    scale: float = 0.8,
) -> np.ndarray:
    """
    Ritar en rad HUD-text horisontellt centrerad, strax under guide-ramen.

    Args:
        frame: BGR-bild.
        text: texten att visa. Tom sträng ritar ingenting.
        guide_box: (x, y, w, h) för guide-ramen, texten placeras relativt denna.
        color: BGR-färg för texten.
        scale: cv2.putText fontScale.

    Returns:
        Samma frame, med texten inritad (om text är icke-tom).
    """
    if not text:
        return frame

    x, y, w, h = guide_box
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_width, _), _ = cv2.getTextSize(text, font, scale, thickness)
    text_x = x + (w - text_width) // 2
    text_y = y + h + 40

    cv2.putText(frame, text, (text_x, text_y), font, scale, color, thickness)
    return frame


def draw_full_screen_message(
    frame: np.ndarray,
    text: str,
    color: tuple[int, int, int] = COLOR_TEAL,
    dim_alpha: float = 0.55,
    scale: float = 1.4,
) -> np.ndarray:
    """
    Mörklägger hela framen och ritar en stor, horisontellt/vertikalt
    centrerad textrad ovanpå - används för RESULT_GRANTED, RESULT_DENIED
    och WELCOME, där hela videon tillfälligt ger vika för ett tydligt
    resultatmeddelande.

    Args:
        frame: BGR-bild.
        text: meddelandet att visa, t.ex. "ACCESS GRANTED".
        color: BGR-färg för texten.
        dim_alpha: hur mörk bakgrunden blir (0 = ingen dimning, 1 = helsvart).
        scale: cv2.putText fontScale för huvudtexten.

    Returns:
        Samma frame, mörklagd med texten inritad.
    """
    overlay = np.zeros_like(frame)
    cv2.addWeighted(overlay, dim_alpha, frame, 1 - dim_alpha, 0, dst=frame)

    frame_height, frame_width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 3
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    text_x = (frame_width - text_width) // 2
    text_y = (frame_height + text_height) // 2

    cv2.putText(frame, text, (text_x, text_y), font, scale, color, thickness)
    return frame


def get_denied_blink_color(base_color: tuple[int, int, int] = COLOR_RED) -> tuple[int, int, int]:
    """
    Ger en tidsbaserad, blinkande version av en färg (används för
    RESULT_DENIED, för att ge en tydlig "avvisad"-känsla).

    Blinkar med ~4 Hz genom att helt släcka färgen (svart) under hälften
    av varje cykel.

    Args:
        base_color: BGR-färgen att blinka mellan.

    Returns:
        base_color under halva cykeln, annars svart (0, 0, 0).
    """
    blink_hz = 4.0
    phase = (time.time() * blink_hz) % 1.0
    return base_color if phase < 0.5 else (0, 0, 0)