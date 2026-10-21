"""End-to-end check of the real Ultralytics stack, no dataset or downloads needed.

Builds a tiny synthetic dataset (green circles = "With Helmet", red squares = "Without Helmet"),
trains YOLO11n from scratch on CPU, then runs val → image → video with tracking → ONNX export.

    python scripts/e2e_synthetic.py            # ~10–15 min on a laptop CPU
    python scripts/e2e_synthetic.py --epochs 20  # faster smoke run (weaker model)
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def make_dataset(root: Path) -> Path:
    random.seed(0)
    np.random.seed(0)
    for split, n in (("train", 120), ("valid", 30)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(n):
            img = np.random.randint(60, 120, (320, 320, 3), np.uint8)
            lines = []
            for _ in range(random.randint(1, 3)):
                c, s = random.randint(0, 1), random.randint(30, 70)
                x, y = random.randint(s, 320 - s), random.randint(s, 320 - s)
                if c == 0:
                    cv2.circle(img, (x, y), s // 2, (0, 220, 0), -1)
                else:
                    cv2.rectangle(img, (x - s // 2, y - s // 2), (x + s // 2, y + s // 2), (0, 0, 230), -1)
                lines.append(f"{c} {x / 320:.4f} {y / 320:.4f} {s / 320:.4f} {s / 320:.4f}")
            cv2.imwrite(str(root / split / "images" / f"{i}.jpg"), img)
            (root / split / "labels" / f"{i}.txt").write_text("\n".join(lines))
    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root}\ntrain: train/images\nval: valid/images\n"
                    "names:\n  0: With Helmet\n  1: Without Helmet\n")
    writer = cv2.VideoWriter(str(root / "clip.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 320))
    for t in range(30):  # one "rider without helmet" crossing the frame + one with helmet
        img = np.full((320, 320, 3), 90, np.uint8)
        cv2.rectangle(img, (40 + 5 * t, 120), (100 + 5 * t, 180), (0, 0, 230), -1)
        cv2.circle(img, (250, 250), 30, (0, 220, 0), -1)
        writer.write(img)
    writer.release()
    cv2.imwrite(str(root / "test.jpg"), img)
    return yaml


def run(*args: str) -> str:
    print("$ helmet-detect", " ".join(args))
    out = subprocess.run([sys.executable, "-m", "helmet_detection", *args], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    print(out.stdout[-1500:])
    if out.returncode:
        print(out.stderr[-3000:])
        raise SystemExit(f"command failed: {args}")
    return out.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    root = Path(tempfile.mkdtemp(prefix="helmet_e2e_"))
    os.environ.update(HELMET_OUTPUT_DIR=str(root / "out"), HELMET_IMGSZ="320")
    data = make_dataset(root / "ds")
    run("train", "--data", str(data), "--model", "yolo11n.yaml", "--epochs", str(a.epochs), "--imgsz", "320",
        "--batch", "16", "--name", "e2e")
    w = str(Path("runs/helmet/e2e/weights/best.pt").resolve())
    run("val", "--data", str(data), "--weights", w, "--imgsz", "320")
    img = run("image", str(root / "ds" / "test.jpg"), "--weights", w, "--save", "--no-show")
    vid = run("video", str(root / "ds" / "clip.avi"), "--weights", w, "--save", "--no-show")
    run("export", "--weights", w, "--format", "onnx", "--imgsz", "320")
    ok = "1 violation" in img and "Violations logged: 1" in vid
    print(f"\n{'PASS' if ok else 'CHECK'}: image found 1 violation, video logged the crossing rider once "
          f"(outputs in {root / 'out'})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
