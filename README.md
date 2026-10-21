# 🪖 AI Helmet Detection System

[![CI](https://github.com/HammadAbrarMatto/Helmet_Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/HammadAbrarMatto/Helmet_Detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLO26%20%7C%20YOLO11-purple)
![License](https://img.shields.io/badge/license-MIT-green)

Real-time helmet-compliance monitoring for two-wheeler riders. A fine-tuned **YOLO** detector classifies riders as *With Helmet* / *Without Helmet*, **ByteTrack** follows each rider across frames so every offender is logged **exactly once**, and every violation is saved as a timestamped snapshot plus a CSV audit row.

Runs on images, video files, webcams and **RTSP/IP cameras**, through a CLI, a desktop GUI, or a browser app.

---

## ✨ What's new in v2

| Area | v1 | v2 |
|---|---|---|
| Model | YOLOv8-L | **YOLO26 / YOLO11** training presets (Ultralytics ≥ 8.4) |
| Duplicate violations | 3-second global cooldown | **Per-rider tracking (ByteTrack)** – one log per rider; cooldown kept as fallback |
| Interfaces | 2 scripts + Tkinter (OpenCV pop-ups) | Single `helmet-detect` CLI, Tkinter app with **embedded live preview**, new **Streamlit web app** |
| Sources | file / webcam | file / webcam / **RTSP & HTTP streams** |
| Deployment | – | `helmet-detect export` → ONNX, OpenVINO, TensorRT, TFLite |
| Code | flat scripts, duplicated detection loops, `cvzone` | `src/` package, one shared pipeline, pure-OpenCV drawing |
| Quality | – | pytest suite, ruff, GitHub Actions CI, `.env` config |

---

## 📁 Project structure

```
Helmet_Detection/
├── src/helmet_detection/
│   ├── config.py        # Settings (env-overridable, HELMET_*)
│   ├── detector.py      # YOLO wrapper: detect() / track()
│   ├── pipeline.py      # shared image + stream processing
│   ├── violations.py    # de-duplication, snapshots, CSV log
│   ├── drawing.py       # box / label / HUD rendering
│   ├── training.py      # train · val · export helpers
│   ├── gui.py           # Tkinter desktop app
│   └── cli.py           # `helmet-detect` entry point
├── app/streamlit_app.py # browser UI
├── configs/data.example.yaml
├── weights/             # put best.pt here (git-ignored)
├── media/               # sample images / videos (git-ignored)
├── outputs/             # violations.csv + snapshots (auto-created)
├── tests/
├── pyproject.toml
└── requirements.txt
```

---

## 🚀 Quick start

```bash
git clone https://github.com/HammadAbrarMatto/Helmet_Detection.git
cd Helmet_Detection
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on macOS/Linux)
pip install -e ".[web]"           # installs the `helmet-detect` command
```

> **GPU:** install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) *before* the step above.

Copy your trained `best.pt` into `weights/` (or train one — see below).

### Run

```bash
helmet-detect gui                                   # desktop app
streamlit run app/streamlit_app.py                  # web app

helmet-detect image media/riders.jpg --save
helmet-detect video media/bike.mp4 --save           # press q to stop
helmet-detect video media/bike.mp4 --save --no-show # headless / server
helmet-detect webcam --source 0
helmet-detect webcam --source rtsp://user:pass@192.168.1.20:554/stream
```

Without installing, `python -m helmet_detection ...` works from the repo root with `PYTHONPATH=src`.

---

## 🏋️ Training your own model

1. Get a helmet dataset in YOLO format (e.g. a *Bike Helmet Detection* export from Roboflow).
2. Point `configs/data.example.yaml` at it (class 0 = With Helmet, class 1 = Without Helmet).
3. Train, evaluate, and copy the best weights:

```bash
helmet-detect train --data configs/data.example.yaml --model yolo26s.pt --epochs 100
helmet-detect val   --data configs/data.example.yaml --weights runs/helmet/train/weights/best.pt
cp runs/helmet/train/weights/best.pt weights/best.pt   # path is printed at the end of training
```

`--model` accepts any Ultralytics checkpoint (`yolo26n/s/m/l/x.pt`, `yolo11*.pt`, or your old YOLOv8 weights). Training uses auto batch size, cosine LR and early stopping (`patience=25`).

### Export for edge devices

```bash
helmet-detect export --format onnx              # CPU anywhere
helmet-detect export --format openvino --int8   # Intel CPUs / NPUs
helmet-detect export --format engine --half     # NVIDIA Jetson / TensorRT
```

---

## ⚙️ Configuration

Defaults live in `src/helmet_detection/config.py`. Override any of them with environment variables or a `.env` file (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `HELMET_WEIGHTS` | `weights/best.pt` | model checkpoint |
| `HELMET_CONF` | `0.40` | confidence threshold |
| `HELMET_DEVICE` | auto | `cpu`, `0`, `mps` … |
| `HELMET_HALF` | `0` | FP16 inference on CUDA |
| `HELMET_TRACK` | `1` | ByteTrack de-duplication for video |
| `HELMET_FRAME_SKIP` | `1` | process every Nth frame |
| `HELMET_COOLDOWN` | `3` | seconds between snapshots when tracking is off |
| `HELMET_OUTPUT_DIR` | `outputs` | log + snapshot folder |

---

## 📊 Violation log

`outputs/violations.csv`:

| Timestamp | Detection | Confidence | TrackID | Source | Image |
|---|---|---|---|---|---|
| 2026-10-21 14:03:12 | Without Helmet | 0.87 | 12 | traffic.mp4 | violation_traffic_20261021_140312_123456.jpg |

Snapshots are written to `outputs/violations/`. The GUI and web app both include a log viewer, and the web app can export it as CSV.

---

## ✅ End-to-end check

`python scripts/e2e_synthetic.py` trains a small YOLO on a generated dataset and runs train → val → image → video tracking → ONNX export through the real CLI. No dataset or downloads are needed. Last run: mAP50 0.994; the rider who stays in frame for 30 video frames is logged **once**; ONNX output matches PyTorch.

## 🧪 Development

```bash
pip install -e ".[dev]"
pytest -q        # unit tests + a real-Ultralytics smoke test (no downloads)
ruff check .
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).

## 🙏 Acknowledgements

[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) · [ByteTrack](https://github.com/ifzhang/ByteTrack) · Bike-helmet dataset contributors on Roboflow Universe
