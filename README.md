# 🪖 AI Helmet Detection System

A real-time computer-vision pipeline that detects whether motorcycle/bicycle riders are wearing helmets. Built with **YOLOv8** and **OpenCV**, it supports images, video files, and live webcam feeds — and automatically logs every violation with a timestamp and screenshot.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🎯 Real-time detection | YOLOv8-based inference on image / video / webcam |
| 🟢🔴 Two-class classification | *With Helmet* and *Without Helmet* |
| 📸 Violation snapshots | Frames saved automatically on each detection |
| 📋 CSV audit log | Timestamped log with confidence score and source |
| 🖥️ GUI | Clean Tkinter interface — no CLI required |
| ⚙️ CLI scripts | `detect_image.py` and `detect_video.py` for scripting |
| 🔧 Central config | All thresholds and paths in one `config.py` |

---

## 📁 Project Structure

```
helmet-detection/
├── config.py               # ← Edit this to customise paths & thresholds
├── utils.py                # Shared helpers (drawing, logging, saving)
├── detect_image.py         # CLI: detect on a single image
├── detect_video.py         # CLI: detect on a video file
├── gui.py                  # Tkinter GUI (recommended entry point)
│
├── Weights/
│   └── best.pt             # YOLOv8 custom weights (not tracked by git)
│
├── Media/                  # Put your test images / videos here
│   ├── riders_1.jpg
│   └── bike_3.mp4
│
├── violations/             # Auto-created: violation screenshots
├── violations.csv          # Auto-created: violation audit log
│
├── requirements.txt
└── .gitignore
```

---

## 🚀 Getting Started

### 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/helmet-detection.git
cd helmet-detection
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** replace the `torch` line in `requirements.txt` with the correct CUDA wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4 — Add model weights

Download or copy your trained `best.pt` into the `Weights/` folder:

```
Weights/
└── best.pt
```

### 5 — Run

**GUI (recommended)**
```bash
python gui.py
```

**CLI — image**
```bash
python detect_image.py --image Media/riders_1.jpg
python detect_image.py --image Media/riders_1.jpg --save   # also writes annotated image
```

**CLI — video**
```bash
python detect_video.py --video Media/bike_3.mp4
python detect_video.py --video Media/bike_3.mp4 --save     # also writes annotated video
```

---

## ⚙️ Configuration

All tuneable parameters live in **`config.py`** — you should never need to edit any other file:

```python
CONFIDENCE_THRESHOLD = 0.40   # Skip detections below this score
FRAME_SKIP           = 2      # Process every Nth frame (speed vs accuracy trade-off)
VIOLATION_COOLDOWN   = 3      # Seconds between saved violation frames (avoids duplicates)
INFERENCE_WIDTH      = 640    # Frame is resized to this before inference
INFERENCE_HEIGHT     = 480
```

---

## 🏋️ Training Your Own Model

The `helmetdetection.py` notebook (Google Colab) documents the original training run:

```
Dataset : BikeHelmet (Roboflow / custom)
Base    : YOLOv8-Large (yolov8l.pt)
Epochs  : 50
Image   : 640 × 640
```

To retrain, upload your `data.yaml` and run:

```bash
yolo task=detect mode=train \
     model=yolov8l.pt \
     data=path/to/data.yaml \
     epochs=50 imgsz=640
```

Copy the resulting `runs/detect/train/weights/best.pt` to `Weights/`.

---

## 📊 Violation Log

Every no-helmet detection writes a row to `violations.csv`:

| Timestamp | Detection | Confidence | Source | Image |
|---|---|---|---|---|
| 2025-07-09 01:03:12 | Without Helmet | 0.87 | webcam | violation_webcam_20250709_010312_123456.jpg |

Screenshots are stored in `violations/`.

---

## 🔑 Key Controls

| Key / Action | Effect |
|---|---|
| `q` | Quit video / webcam window |
| Close window (✕) | Also exits gracefully |
| GUI — **Stop Webcam** | Cleanly stops the webcam thread |

---

## 🛠️ Requirements

- Python 3.10+
- OpenCV `opencv-python`
- Ultralytics `ultralytics` (YOLOv8)
- `cvzone`
- Tkinter (bundled with standard Python on Windows/macOS; `sudo apt install python3-tk` on Ubuntu)

---

## 📄 License

MIT — see `LICENSE` for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [cvzone](https://github.com/cvzone/cvzone)
- BikeHelmet dataset contributors
