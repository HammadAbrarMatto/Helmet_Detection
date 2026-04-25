"""
detect_video.py — Run helmet detection on a video file.

Usage:
    python detect_video.py --video Media/bike_3.mp4 [--save]
"""

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import (
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    FRAME_SKIP,
    INFERENCE_HEIGHT,
    INFERENCE_WIDTH,
    WEIGHTS_PATH,
)
from utils import ensure_directories, process_detections


def detect_video(video_path: str, save_output: bool = False) -> None:
    """
    Run YOLO inference frame-by-frame on *video_path*.

    Args:
        video_path:  Path to the input video file.
        save_output: If True, write the annotated video to disk.
    """
    ensure_directories()

    vid_path = Path(video_path)
    if not vid_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    model = YOLO(WEIGHTS_PATH)
    cap   = cv2.VideoCapture(str(vid_path))

    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        sys.exit(1)

    # Optional writer
    writer = None
    if save_output:
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = vid_path.stem + "_detected.mp4"
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
        print(f"[INFO] Saving annotated video to: {out_path}")

    window_name = "Helmet Detection — Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    frame_id         = 0
    violation_state  = {"count": 0, "last_saved": 0.0}
    total_violations = 0

    print("[INFO] Processing video. Press 'q' or close the window to stop.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("[INFO] End of video.")
            break

        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
        results = model(frame, stream=True, verbose=False)

        new_v = process_detections(frame, results, "video", violation_state)
        total_violations += new_v

        if writer:
            writer.write(frame)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Stopped by user.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Total violations logged: {total_violations}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Helmet detection on a video file.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--save",  action="store_true", help="Save the annotated video.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    detect_video(args.video, save_output=args.save)
