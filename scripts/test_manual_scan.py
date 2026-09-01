"""
scripts/test_manual_scan.py

Manuellt sanity-check-skript för HUD-overlay-funktionerna i
app/hud_overlay.py. Renderar scanning-linje, hörnhakar och HUD-text
på en dummy-frame och skriver resultatet till disk för visuell
inspektion. Kräver ingen webcam.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from app.hud_overlay import (
    COLOR_TEAL,
    draw_scan_line,
    draw_corner_brackets,
    draw_hud_text,
    draw_full_screen_message,
    get_guide_frame,
)

frame = np.zeros((480, 640, 3), dtype=np.uint8)
guide_box = get_guide_frame(640, 480)

frame = draw_scan_line(frame, guide_box, color=COLOR_TEAL)
frame = draw_corner_brackets(frame, guide_box, color=COLOR_TEAL, glow=False)
frame = draw_hud_text(frame, "Hold still and blink", guide_box, color=COLOR_TEAL)

output_path = project_root / "test_scan_line.png"
cv2.imwrite(str(output_path), frame)
print(f"Klart, se {output_path}")