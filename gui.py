"""
gui.py — Tkinter GUI for the Helmet Detection System.

Provides buttons to:
  • Detect on an uploaded image
  • Detect on an uploaded video
  • Start / Stop webcam detection
  • View violation statistics
  • Exit cleanly

Usage:
    python gui.py
"""

import csv
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from ultralytics import YOLO

from config import VIOLATIONS_CSV, VIOLATIONS_DIR, WEIGHTS_PATH
from utils import ensure_directories, process_detections


# ─── Model (loaded once) ─────────────────────────────────────────────────────

_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(WEIGHTS_PATH)
    return _model


# ─── Detection runners ───────────────────────────────────────────────────────

def _run_image(file_path: str) -> None:
    from utils import draw_detection, log_violation, save_violation_frame, CLASS_NAMES, CONFIDENCE_THRESHOLD

    model  = _get_model()
    frame  = cv2.imread(file_path)

    if frame is None:
        messagebox.showerror("Error", f"Could not read image:\n{file_path}")
        return

    results   = model(frame, verbose=False)
    violations = 0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            draw_detection(frame, x1, y1, x2, y2, cls, conf)

            if cls == 1:
                img_name = save_violation_frame(frame, "image")
                log_violation(CLASS_NAMES[cls], conf, file_path, img_name)
                violations += 1

    cv2.imshow("Helmet Detection — Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if violations:
        messagebox.showinfo("Violations Found", f"⚠️  {violations} violation(s) saved to '{VIOLATIONS_DIR}'.")


def _run_video(file_path: str, stop_event: threading.Event) -> None:
    model   = _get_model()
    cap     = cv2.VideoCapture(file_path)
    state   = {"count": 0, "last_saved": 0.0}
    win     = "Helmet Detection — Video"

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    while cap.isOpened() and not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, stream=True, verbose=False)
        process_detections(frame, results, "video", state)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


def _run_webcam(stop_event: threading.Event) -> None:
    model  = _get_model()
    cap    = cv2.VideoCapture(0)
    state  = {"count": 0, "last_saved": 0.0}
    win    = "Helmet Detection — Webcam"

    if not cap.isOpened():
        messagebox.showerror("Webcam Error", "Could not access webcam (index 0).")
        return

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while cap.isOpened() and not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, stream=True, verbose=False)
        process_detections(frame, results, "webcam", state)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()


# ─── Statistics window ───────────────────────────────────────────────────────

def _show_stats(parent: tk.Tk) -> None:
    if not os.path.exists(VIOLATIONS_CSV):
        messagebox.showinfo("Stats", "No violations logged yet.")
        return

    with open(VIOLATIONS_CSV, newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) <= 1:
        messagebox.showinfo("Stats", "No violations logged yet.")
        return

    win = tk.Toplevel(parent)
    win.title("Violation Log")
    win.geometry("700x400")
    win.configure(bg="#1a1a2e")

    headers = rows[0]
    tree    = ttk.Treeview(win, columns=headers, show="headings")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview",
                    background="#16213e", foreground="white",
                    fieldbackground="#16213e", rowheight=28)
    style.configure("Treeview.Heading",
                    background="#0f3460", foreground="white", font=("Courier New", 10, "bold"))

    for col in headers:
        tree.heading(col, text=col)
        tree.column(col, width=130, anchor="center")

    for row in rows[1:]:
        tree.insert("", tk.END, values=row)

    scrollbar = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    scrollbar.pack(side="right", fill="y", pady=10)

    tk.Label(win, text=f"Total violations: {len(rows) - 1}",
             font=("Courier New", 11, "bold"), bg="#1a1a2e", fg="#e94560").pack(pady=5)


# ─── Main GUI ────────────────────────────────────────────────────────────────

def launch_gui() -> None:
    ensure_directories()

    webcam_stop_event: threading.Event | None = None
    webcam_thread:     threading.Thread | None = None
    video_stop_event:  threading.Event | None = None

    # ── Root window ──────────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("AI Helmet Detection System")
    root.geometry("440x520")
    root.resizable(False, False)
    root.configure(bg="#1a1a2e")

    # ── Header ───────────────────────────────────────────────────────────────
    tk.Label(
        root,
        text="🪖  Helmet Detection",
        font=("Courier New", 20, "bold"),
        bg="#1a1a2e",
        fg="#e94560",
    ).pack(pady=(28, 4))

    tk.Label(
        root,
        text="AI-powered safety compliance checker",
        font=("Courier New", 10),
        bg="#1a1a2e",
        fg="#a8a8c0",
    ).pack(pady=(0, 20))

    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=30, pady=4)

    # ── Button factory ────────────────────────────────────────────────────────
    BTN_STYLE = dict(
        font=("Courier New", 12, "bold"),
        bg="#0f3460",
        fg="white",
        activebackground="#e94560",
        activeforeground="white",
        relief="flat",
        cursor="hand2",
        width=28,
        height=2,
    )

    def _btn(text: str, cmd) -> tk.Button:
        b = tk.Button(root, text=text, command=cmd, **BTN_STYLE)
        b.pack(pady=6)
        return b

    # ── Handlers ──────────────────────────────────────────────────────────────
    def upload_image():
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if path:
            threading.Thread(target=_run_image, args=(path,), daemon=True).start()

    def upload_video():
        nonlocal video_stop_event
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")],
        )
        if path:
            video_stop_event = threading.Event()
            threading.Thread(target=_run_video, args=(path, video_stop_event), daemon=True).start()

    def start_webcam():
        nonlocal webcam_stop_event, webcam_thread
        if webcam_thread and webcam_thread.is_alive():
            messagebox.showinfo("Webcam", "Webcam is already running.")
            return
        webcam_stop_event = threading.Event()
        webcam_thread     = threading.Thread(target=_run_webcam, args=(webcam_stop_event,), daemon=True)
        webcam_thread.start()
        status_label.config(text="🟢 Webcam active", fg="#00d26a")

    def stop_webcam():
        nonlocal webcam_stop_event
        if webcam_stop_event:
            webcam_stop_event.set()
        status_label.config(text="🔴 Webcam stopped", fg="#e94560")

    def show_stats():
        _show_stats(root)

    def quit_app():
        stop_webcam()
        cv2.destroyAllWindows()
        root.destroy()

    # ── Buttons ───────────────────────────────────────────────────────────────
    _btn("📷  Detect on Image",    upload_image)
    _btn("🎬  Detect on Video",    upload_video)
    _btn("📹  Start Webcam",       start_webcam)
    _btn("⏹   Stop Webcam",        stop_webcam)
    _btn("📊  View Violations Log", show_stats)

    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=30, pady=10)
    _btn("🚪  Exit", quit_app)

    # ── Status bar ────────────────────────────────────────────────────────────
    status_label = tk.Label(root, text="Ready", font=("Courier New", 9),
                            bg="#1a1a2e", fg="#a8a8c0")
    status_label.pack(pady=(6, 0))

    root.protocol("WM_DELETE_WINDOW", quit_app)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
