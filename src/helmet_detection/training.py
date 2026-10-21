"""Training, validation and export helpers (wrapping the Ultralytics API)."""

from __future__ import annotations

from pathlib import Path

DEFAULT_BASE_MODEL = "yolo26s.pt"  # YOLO26 (Ultralytics ≥ 8.4). "yolo11s.pt" also works.


def train(
    data: str,
    model: str = DEFAULT_BASE_MODEL,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = -1,
    device: str | None = None,
    project: str = "runs/helmet",
    name: str = "train",
    patience: int = 25,
) -> Path:
    """Fine-tune a YOLO model on a helmet dataset. Returns the path to best.pt."""
    from ultralytics import YOLO

    yolo = YOLO(model)
    yolo.train(
        data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=device,
        project=project, name=name, patience=patience, exist_ok=True,
        cos_lr=True, close_mosaic=10, plots=True,
    )
    best = Path(yolo.trainer.save_dir) / "weights" / "best.pt"
    print(f"[train] best weights: {best}")
    return best


def validate(weights: str, data: str, imgsz: int = 640, device: str | None = None) -> dict:
    from ultralytics import YOLO

    metrics = YOLO(weights).val(data=data, imgsz=imgsz, device=device, plots=True)
    summary = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    for k, v in summary.items():
        print(f"{k:>10}: {v:.4f}")
    return summary


def export(weights: str, fmt: str = "onnx", imgsz: int = 640, half: bool = False, int8: bool = False) -> str:
    """Export for edge / CPU deployment: onnx, openvino, engine (TensorRT), tflite, coreml…"""
    from ultralytics import YOLO

    out = YOLO(weights).export(format=fmt, imgsz=imgsz, half=half, int8=int8)
    print(f"[export] {out}")
    return str(out)
