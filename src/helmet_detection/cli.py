"""Command-line interface: ``helmet-detect <command> ...``.

Examples
--------
    helmet-detect image media/riders.jpg --save
    helmet-detect video media/bike.mp4 --save --no-show
    helmet-detect webcam --source 0
    helmet-detect webcam --source rtsp://user:pass@cam-ip/stream
    helmet-detect train --data configs/data.example.yaml --model yolo26s.pt --epochs 100
    helmet-detect val --data configs/data.example.yaml
    helmet-detect export --format onnx
    helmet-detect gui
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--weights", help="Path to YOLO weights (default: weights/best.pt)")
    p.add_argument("--conf", type=float, help="Confidence threshold")
    p.add_argument("--device", help="cpu | 0 | mps … (default: auto)")
    p.add_argument("--output-dir", help="Where violation logs/snapshots are written")


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings()
    if getattr(args, "weights", None):
        s.weights = Path(args.weights)
    if getattr(args, "conf", None) is not None:
        s.conf_threshold = args.conf
    if getattr(args, "device", None):
        s.device = args.device
    if getattr(args, "output_dir", None):
        s.output_dir = Path(args.output_dir)
    return s


def cmd_image(args: argparse.Namespace) -> int:
    import cv2

    from .detector import HelmetDetector
    from .pipeline import process_image
    from .violations import ViolationLogger

    s = _settings(args)
    detector = HelmetDetector(s)
    logger = ViolationLogger(s)
    res = process_image(detector, logger, args.path)
    print(f"[INFO] {len(res.detections)} detections, {len(res.new_violations)} violation(s)")
    for d in res.detections:
        print(f"  - {d.label:<16} conf={d.conf:.2f} box={d.box}")
    if args.save:
        out = s.output_dir / f"{Path(args.path).stem}_detected{Path(args.path).suffix}"
        cv2.imwrite(str(out), res.frame)
        print(f"[INFO] Annotated image → {out}")
    if args.show:
        cv2.imshow("Helmet Detection — Image", res.frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def _run_stream(args: argparse.Namespace, source: str) -> int:
    import cv2

    from .detector import HelmetDetector
    from .pipeline import iter_stream, video_properties
    from .violations import ViolationLogger

    s = _settings(args)
    if args.frame_skip:
        s.frame_skip = args.frame_skip
    detector = HelmetDetector(s)
    logger = ViolationLogger(s)

    writer = None
    win = "Helmet Detection"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, s.display_width, s.display_height)

    try:
        for res in iter_stream(detector, logger, source, track=not args.no_track):
            if args.save and writer is None:
                fps, _ = video_properties(source)
                h, w = res.frame.shape[:2]
                out = s.output_dir / f"{Path(str(source)).stem or 'stream'}_detected.mp4"
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps / s.frame_skip, (w, h))
                print(f"[INFO] Writing annotated video → {out}")
            if writer is not None:
                writer.write(res.frame)
            for d in res.new_violations:
                print(f"[VIOLATION] frame {res.frame_index}: {d.label} conf={d.conf:.2f} track={d.track_id}")
            if args.show:
                cv2.imshow(win, res.frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    print(f"[INFO] Done. Violations logged: {logger.count} → {s.violations_csv}")
    return 0


def cmd_video(args: argparse.Namespace) -> int:
    return _run_stream(args, args.path)


def cmd_webcam(args: argparse.Namespace) -> int:
    return _run_stream(args, args.source)


def cmd_train(args: argparse.Namespace) -> int:
    from .training import train

    train(args.data, model=args.model, epochs=args.epochs, imgsz=args.imgsz,
          batch=args.batch, device=args.device, name=args.name)
    return 0


def cmd_val(args: argparse.Namespace) -> int:
    from .training import validate

    validate(args.weights or str(Settings().weights), args.data, imgsz=args.imgsz, device=args.device)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .training import export

    export(args.weights or str(Settings().weights), fmt=args.format, imgsz=args.imgsz,
           half=args.half, int8=args.int8)
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui import launch_gui

    launch_gui(_settings(args))
    return 0


def build_parser() -> argparse.ArgumentParser:
    from .training import DEFAULT_BASE_MODEL

    parser = argparse.ArgumentParser(prog="helmet-detect", description="AI helmet-compliance detection")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("image", help="Detect on a single image")
    p.add_argument("path")
    p.add_argument("--save", action="store_true", help="Save the annotated image")
    p.add_argument("--no-show", dest="show", action="store_false", help="Don't open a window")
    _common(p)
    p.set_defaults(func=cmd_image)

    for name, helptext, func in (("video", "Detect on a video file", cmd_video),
                                 ("webcam", "Detect on a webcam / RTSP stream", cmd_webcam)):
        p = sub.add_parser(name, help=helptext)
        if name == "video":
            p.add_argument("path")
        else:
            p.add_argument("--source", default="0", help="Webcam index or RTSP/HTTP URL")
        p.add_argument("--save", action="store_true", help="Save the annotated video")
        p.add_argument("--no-show", dest="show", action="store_false", help="Run headless")
        p.add_argument("--no-track", action="store_true", help="Disable ByteTrack de-duplication")
        p.add_argument("--frame-skip", type=int, help="Process every Nth frame")
        _common(p)
        p.set_defaults(func=func)

    p = sub.add_parser("train", help="Fine-tune YOLO on a helmet dataset")
    p.add_argument("--data", required=True, help="Ultralytics data.yaml")
    p.add_argument("--model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=-1, help="-1 = auto batch size")
    p.add_argument("--device")
    p.add_argument("--name", default="train")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("val", help="Evaluate weights on the validation split")
    p.add_argument("--data", required=True)
    p.add_argument("--weights")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device")
    p.set_defaults(func=cmd_val)

    p = sub.add_parser("export", help="Export weights (onnx, openvino, engine, tflite …)")
    p.add_argument("--weights")
    p.add_argument("--format", default="onnx")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("gui", help="Launch the Tkinter desktop app")
    _common(p)
    p.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
