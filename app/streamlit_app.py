"""Browser UI for helmet detection.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helmet_detection.config import Settings  # noqa: E402
from helmet_detection.violations import ViolationLogger, read_log  # noqa: E402

st.set_page_config(page_title="Helmet Detection", page_icon="🪖", layout="wide")


@st.cache_resource(show_spinner="Loading YOLO model…")
def load_detector(weights: str):
    from helmet_detection.detector import HelmetDetector

    return HelmetDetector(Settings(), weights=weights)


settings = Settings()
st.title("🪖 AI Helmet Detection")
st.caption("YOLO + ByteTrack · each rider without a helmet is logged once with a snapshot")

with st.sidebar:
    st.header("Settings")
    weights = st.text_input("Weights", str(settings.weights))
    conf = st.slider("Confidence threshold", 0.10, 0.90, settings.conf_threshold, 0.05)
    track = st.toggle("Track riders (de-duplicate violations)", value=True)
    frame_skip = st.number_input("Process every Nth frame", 1, 10, 2)

try:
    detector = load_detector(weights)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

detector.settings.conf_threshold = conf
logger = ViolationLogger(detector.settings)

tab_img, tab_vid, tab_log = st.tabs(["📷 Image", "🎬 Video", "📋 Violation log"])

with tab_img:
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up:
        from helmet_detection.pipeline import process_frame

        frame = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        logger.reset()
        res = process_frame(detector, logger, frame, up.name)
        c1, c2 = st.columns([3, 1])
        c1.image(cv2.cvtColor(res.frame, cv2.COLOR_BGR2RGB), width="stretch")
        n_bad = sum(d.is_violation for d in res.detections)
        c2.metric("Riders", len(res.detections))
        c2.metric("Without helmet", n_bad)
        if res.detections:
            c2.dataframe(pd.DataFrame([{"label": d.label, "conf": round(d.conf, 2)} for d in res.detections]),
                         hide_index=True)

with tab_vid:
    up = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if up and st.button("▶ Run detection", type="primary"):
        from helmet_detection.pipeline import iter_stream, video_properties

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as tmp:
            tmp.write(up.read())
            path = tmp.name
        _, total = video_properties(path)
        view, prog, stats = st.empty(), st.progress(0.0), st.empty()
        for res in iter_stream(detector, logger, path, frame_skip=int(frame_skip), track=track):
            view.image(cv2.cvtColor(res.frame, cv2.COLOR_BGR2RGB), width="stretch")
            if total:
                prog.progress(min(res.frame_index / total, 1.0))
            stats.markdown(f"**FPS** {res.fps or 0:.1f} · **violations** {logger.count}")
        prog.progress(1.0)
        st.success(f"Done — {logger.count} violation(s) logged.")

with tab_log:
    rows = read_log(settings)
    if len(rows) <= 1:
        st.info("No violations logged yet.")
    else:
        df = pd.DataFrame(rows[1:], columns=rows[0])
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("⬇ Download CSV", df.to_csv(index=False), "violations.csv")
        latest = df.iloc[-1]["Image"]
        snap = settings.violations_dir / latest
        if snap.exists():
            st.image(str(snap), caption=f"Latest snapshot: {latest}", width=480)
