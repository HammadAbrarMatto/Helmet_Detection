"""Tkinter desktop app with an embedded video panel.

Inference runs on a worker thread; frames are handed to the Tk main loop through
a queue, so the UI never freezes and OpenCV windows are not needed.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from .config import Settings
from .violations import read_log

BG, PANEL, ACCENT, FG, MUTED = "#1a1a2e", "#16213e", "#e94560", "#ffffff", "#a8a8c0"
PREVIEW_W, PREVIEW_H = 800, 450


class HelmetApp:
    def __init__(self, root: tk.Tk, settings: Settings):
        self.root = root
        self.settings = settings
        self._detector = None
        self._logger = None
        self._frames: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._photo = None
        self._build()
        self.root.after(30, self._poll)

    # ── lazy model loading ───────────────────────────────────────────────────
    def _ensure_model(self) -> bool:
        if self._detector is not None:
            return True
        try:
            from .detector import HelmetDetector
            from .violations import ViolationLogger

            self._set_status("Loading model…")
            self._detector = HelmetDetector(self.settings)
            self._logger = ViolationLogger(self.settings)
            return True
        except Exception as e:  # noqa: BLE001 – surface any load error to the user
            messagebox.showerror("Model error", str(e))
            self._set_status("Model not loaded")
            return False

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build(self) -> None:
        r = self.root
        r.title("AI Helmet Detection System")
        r.configure(bg=BG)
        r.minsize(1100, 560)

        side = tk.Frame(r, bg=BG)
        side.pack(side="left", fill="y", padx=16, pady=16)
        tk.Label(side, text="🪖 Helmet Detection", font=("Segoe UI", 18, "bold"), bg=BG, fg=ACCENT).pack(pady=(4, 2))
        tk.Label(side, text="AI safety-compliance checker", font=("Segoe UI", 10), bg=BG, fg=MUTED).pack(pady=(0, 14))

        btn = dict(font=("Segoe UI", 11, "bold"), bg="#0f3460", fg=FG, activebackground=ACCENT,
                   activeforeground=FG, relief="flat", cursor="hand2", width=24, height=2)
        for text, cmd in (("📷  Detect on Image", self.on_image), ("🎬  Detect on Video", self.on_video),
                          ("📹  Start Webcam", self.on_webcam), ("⏹  Stop", self.on_stop),
                          ("📊  Violations Log", self.on_stats), ("🚪  Exit", self.on_exit)):
            tk.Button(side, text=text, command=cmd, **btn).pack(pady=5)

        tk.Label(side, text="Confidence", bg=BG, fg=MUTED).pack(pady=(14, 0))
        self.conf = tk.DoubleVar(value=self.settings.conf_threshold)
        ttk.Scale(side, from_=0.1, to=0.9, variable=self.conf, command=self._on_conf).pack(fill="x")

        self.status = tk.Label(side, text="Ready", font=("Segoe UI", 9), bg=BG, fg=MUTED, wraplength=220)
        self.status.pack(side="bottom", pady=8)

        self.canvas = tk.Label(r, bg=PANEL, width=PREVIEW_W, height=PREVIEW_H)
        self.canvas.pack(side="right", fill="both", expand=True, padx=16, pady=16)
        r.protocol("WM_DELETE_WINDOW", self.on_exit)

    def _set_status(self, text: str, color: str = MUTED) -> None:
        self.status.config(text=text, fg=color)

    def _on_conf(self, _=None) -> None:
        self.settings.conf_threshold = round(self.conf.get(), 2)

    def _show(self, frame) -> None:
        from PIL import Image, ImageTk

        w = max(self.canvas.winfo_width(), 200)
        h = max(self.canvas.winfo_height(), 150)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((w, h))
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.config(image=self._photo)

    def _poll(self) -> None:
        try:
            while True:
                item = self._frames.get_nowait()
                if isinstance(item, str):
                    self._set_status(item)
                else:
                    self._show(item.frame)
                    msg = f"🟢 Running · FPS {item.fps or 0:.1f} · violations {self._logger.count}"
                    self._set_status(msg, "#00d26a")
        except queue.Empty:
            pass
        self.root.after(30, self._poll)

    # ── actions ──────────────────────────────────────────────────────────────
    def on_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path or not self._ensure_model():
            return
        from .pipeline import process_image

        self.on_stop()
        try:
            res = process_image(self._detector, self._logger, path)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        self._show(res.frame)
        self._set_status(f"{len(res.detections)} detections · {len(res.new_violations)} violation(s)")

    def _start_stream(self, source) -> None:
        if not self._ensure_model():
            return
        self.on_stop()
        self._stop = threading.Event()
        stop = self._stop

        def run() -> None:
            from .pipeline import iter_stream

            try:
                for res in iter_stream(self._detector, self._logger, source):
                    if stop.is_set():
                        break
                    try:
                        self._frames.put_nowait(res)
                    except queue.Full:
                        pass  # drop frames if the UI is behind
                self._frames.put("⏹ Stopped")
            except Exception as e:  # noqa: BLE001
                self._frames.put(f"Error: {e}")

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()

    def on_video(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self._start_stream(path)

    def on_webcam(self) -> None:
        self._start_stream(0)

    def on_stop(self) -> None:
        self._stop.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2)

    def on_stats(self) -> None:
        rows = read_log(self.settings)
        if len(rows) <= 1:
            messagebox.showinfo("Violations", "No violations logged yet.")
            return
        win = tk.Toplevel(self.root)
        win.title("Violation Log")
        win.geometry("820x420")
        win.configure(bg=BG)
        tree = ttk.Treeview(win, columns=rows[0], show="headings")
        for col in rows[0]:
            tree.heading(col, text=col)
            tree.column(col, width=130, anchor="center")
        for row in rows[1:]:
            tree.insert("", tk.END, values=row)
        sb = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        sb.pack(side="right", fill="y", pady=10)

    def on_exit(self) -> None:
        self.on_stop()
        self.root.destroy()


def launch_gui(settings: Settings | None = None) -> None:
    root = tk.Tk()
    HelmetApp(root, settings or Settings())
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
