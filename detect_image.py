"""
detect_image.py — Run helmet detection on a single image.

Usage:
    python detect_image.py --image Media/riders_1.jpg [--save]
"""

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import WEIGHTS_PATH
from utils import draw_detection, ensure_directories, log_violation, save_violation_frame, CLASS_NAMES, CONFIDENCE_THRESHOLD


def detect_image(image_path: str, save_output: bool = False) -> None:
    """
    Run YOLO inference on *image_path* and display detections.

    Args:
        image_path:   Path to the input image.
        save_output:  If True, save the annotated image to disk.
    """
    ensure_directories()

    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    model = YOLO(WEIGHTS_PATH)
    frame = cv2.imread(str(img_path))

    if frame is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)

    results = model(frame)
    violation_count = 0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            draw_detection(frame, x1, y1, x2, y2, cls, conf)

            if cls == 1:            # Without Helmet
                img_name = save_violation_frame(frame, "image")
                log_violation(CLASS_NAMES[cls], conf, image_path, img_name)
                violation_count += 1

    print(f"[INFO] Detections complete. Violations found: {violation_count}")

    if save_output:
        out_path = img_path.stem + "_detected" + img_path.suffix
        cv2.imwrite(out_path, frame)
        print(f"[INFO] Annotated image saved to: {out_path}")

    cv2.imshow("Helmet Detection — Image", frame)
    print("[INFO] Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Helmet detection on a static image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--save",  action="store_true", help="Save the annotated image.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    detect_image(args.image, save_output=args.save)
