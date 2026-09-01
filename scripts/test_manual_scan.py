import cv2
import numpy as np
from app.hud_overlay import draw_scan_line, draw_corner_brackets, get_status_color

frame = np.zeros((480, 640, 3), dtype=np.uint8)
face_box = (200, 100, 200, 200)

color = get_status_color("unauthorized")
frame = draw_scan_line(frame, face_box, color=color)
frame = draw_corner_brackets(frame, face_box, color=color, glow=True)

cv2.imwrite("test_scan_line.png", frame)
print("Klart, se test_scan_line.png")