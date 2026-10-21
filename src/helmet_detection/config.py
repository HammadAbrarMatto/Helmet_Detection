"""Central configuration.

Every setting has a sensible default and can be overridden with an environment
variable prefixed ``HELMET_`` (e.g. ``HELMET_CONF=0.5``) or a ``.env`` file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"HELMET_{name}", default)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


_load_dotenv()


@dataclass
class Settings:
    # ── Paths ────────────────────────────────────────────────────────────────
    weights: Path = field(default_factory=lambda: Path(_env("WEIGHTS", str(PROJECT_ROOT / "weights" / "best.pt"))))
    output_dir: Path = field(default_factory=lambda: Path(_env("OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))))

    # ── Model ────────────────────────────────────────────────────────────────
    class_names: tuple[str, ...] = ("With Helmet", "Without Helmet")
    violation_class_ids: tuple[int, ...] = (1,)
    conf_threshold: float = float(_env("CONF", "0.40"))
    iou_threshold: float = float(_env("IOU", "0.50"))
    imgsz: int = int(_env("IMGSZ", "640"))
    device: str | None = _env("DEVICE", "") or None  # "cpu", "0", "mps" … None = auto
    half: bool = _env("HALF", "0") == "1"  # FP16 inference on CUDA

    # ── Video / webcam ───────────────────────────────────────────────────────
    frame_skip: int = int(_env("FRAME_SKIP", "1"))  # process every Nth frame
    track: bool = _env("TRACK", "1") == "1"  # ByteTrack → each rider logged once
    tracker: str = _env("TRACKER", "bytetrack.yaml")
    display_width: int = 960
    display_height: int = 540

    # ── Violation logging ────────────────────────────────────────────────────
    violation_cooldown_s: float = float(_env("COOLDOWN", "3"))

    @property
    def violations_dir(self) -> Path:
        return self.output_dir / "violations"

    @property
    def violations_csv(self) -> Path:
        return self.output_dir / "violations.csv"

    def label(self, cls: int) -> str:
        return self.class_names[cls] if 0 <= cls < len(self.class_names) else str(cls)

    def is_violation(self, cls: int) -> bool:
        return cls in self.violation_class_ids


CSV_HEADERS = ["Timestamp", "Detection", "Confidence", "TrackID", "Source", "Image"]
