"""
scripts/test_liveness.py

Manuellt testscript för att köra LivenessDetector mot en riktig webcam
och kalibrera trösklar (EAR_THRESHOLD, MOTION_THRESHOLD) empiriskt.

Inte del av produktionspipelinen — ett engångsverktyg för kalibrering,
i samma anda som run_webcam_face_detection() i face_detection.py.

Tryck 'q' för att avsluta, 'r' för att återställa LivenessDetector-state.
"""

import sys
import statistics
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2

from src.face_detection import (
    create_face_detector_for_images,
    create_face_mesh_detector,
    landmarks_to_pixel_array,
    _bgr_frame_to_mp_image,
)
from src.liveness import LivenessDetector


def main(camera_index: int = 0) -> None:
    face_detector = create_face_detector_for_images()
    mesh_detector = create_face_mesh_detector()
    liveness = LivenessDetector()

    ear_min, ear_max = float("inf"), float("-inf")
    motion_min, motion_max = float("inf"), float("-inf")
    motion_values = []

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kunde inte öppna webcam med index {camera_index}.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            mp_image = _bgr_frame_to_mp_image(frame)

            face_result = face_detector.detect(mp_image)
            mesh_result = mesh_detector.detect(mp_image)

            if face_result.detections and mesh_result.face_landmarks:
                bbox = face_result.detections[0].bounding_box
                face_bbox = (bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)

                frame_height, frame_width = frame.shape[:2]
                pixel_landmarks = landmarks_to_pixel_array(
                    mesh_result.face_landmarks[0], frame_width, frame_height
                )

                status = liveness.update(frame, pixel_landmarks, face_bbox)

                if liveness.ear_history:
                    latest_ear = liveness.ear_history[-1]
                    ear_min = min(ear_min, latest_ear)
                    ear_max = max(ear_max, latest_ear)
                else:
                    latest_ear = float("nan")

                latest_motion = liveness.motion_history[-1] if liveness.motion_history else float("nan")
                if liveness.motion_history:
                    motion_min = min(motion_min, latest_motion)
                    motion_max = max(motion_max, latest_motion)
                    motion_values.append(latest_motion)

                cv2.rectangle(
                    frame,
                    (bbox.origin_x, bbox.origin_y),
                    (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                    color=(0, 255, 0),
                    thickness=2,
                )
                overlay_lines = [
                    f"Status: {status}",
                    f"EAR: {latest_ear:.3f}  (min {ear_min:.3f} / max {ear_max:.3f})",
                    f"Motion: {latest_motion:.2f}",
                    f"Blinks i fonster: {len(liveness.blink_timestamps)}",
                ]
            else:
                overlay_lines = ["Inget ansikte/landmarks detekterat"]

            for i, line in enumerate(overlay_lines):
                cv2.putText(
                    frame, line, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0, 255, 0), thickness=2,
                )

            cv2.imshow("Liveness Detection Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                liveness.reset()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nEAR-intervall under körningen: min={ear_min:.3f}, max={ear_max:.3f}")
        if motion_values:
            sorted_motion = sorted(motion_values)
            print(f"Motion-intervall: min={motion_min:.2f}, max={motion_max:.2f}, medel={sum(motion_values)/len(motion_values):.2f}")
            print(f"Motion-percentiler: p10={statistics.quantiles(sorted_motion, n=10)[0]:.2f}, "
                  f"p25={statistics.quantiles(sorted_motion, n=4)[0]:.2f}, "
                  f"median={statistics.median(sorted_motion):.2f}, "
                  f"p75={statistics.quantiles(sorted_motion, n=4)[2]:.2f}, "
                  f"p90={statistics.quantiles(sorted_motion, n=10)[8]:.2f}")


if __name__ == "__main__":
    main()