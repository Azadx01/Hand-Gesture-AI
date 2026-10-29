"""
app.py
------
Desktop GUI for the Hand Gesture AI Virtual Mouse system.
Built with CustomTkinter for a modern, dark-mode interface.

Features
--------
- Live webcam feed with hand landmarks overlay
- Real-time gesture, confidence, and FPS display
- Start / Stop camera, Enable / Pause mouse control
- Launch data collection and model training from the GUI
- Sensitivity slider
- Status log with colour-coded messages
- Keyboard shortcut: ESC to emergency-stop mouse control
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import threading
import subprocess
import sys
import os
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
import logging

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.camera import Camera
from src.hand_tracker import HandTracker
from src.feature_extraction import extract_features
from src.gesture_classifier import GestureClassifier
from src.mouse_controller import MouseController

logger = logging.getLogger(__name__)

# ---- Appearance ------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

FONT_TITLE  = ("Segoe UI", 22, "bold")
FONT_LABEL  = ("Segoe UI", 13)
FONT_VALUE  = ("Segoe UI", 14, "bold")
FONT_LOG    = ("Consolas", 11)

BG_DARK     = "#1a1a2e"
BG_PANEL    = "#16213e"
ACCENT_BLUE = "#0f3460"
ACCENT_CYAN = "#00b4d8"
ACCENT_GREEN= "#2ecc71"
ACCENT_RED  = "#e74c3c"
ACCENT_WARN = "#f39c12"

GESTURE_COLORS = {
    "MOVE":        "#00b4d8",
    "CLICK":       "#2ecc71",
    "RIGHT_CLICK": "#9b59b6",
    "SCROLL":      "#f39c12",
    "PAUSE":       "#95a5a6",
    "DRAG":        "#e74c3c",
    "IDLE":        "#7f8c8d",
}

GESTURE_ICONS = {
    "MOVE":        "👆",
    "CLICK":       "👌",
    "RIGHT_CLICK": "🤏",
    "SCROLL":      "☝️",
    "PAUSE":       "✊",
    "DRAG":        "✌️",
    "IDLE":        "🤚",
}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Hand Gesture AI – Virtual Mouse")
        self.geometry("1200x780")
        self.minsize(1000, 680)
        self.configure(fg_color=BG_DARK)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Escape>", self._emergency_stop)

        # ---- Core components -----------------------------------------------
        self._camera    = Camera(index=0)
        self._tracker   = HandTracker()
        self._classifier = GestureClassifier()
        self._mouse     = MouseController()
        self._mouse_enabled = False

        # ---- Thread control -------------------------------------------------
        self._running    = False
        self._cam_thread: threading.Thread | None = None

        # ---- Load model (non-blocking) --------------------------------------
        self._model_loaded = self._classifier.load()

        # ---- Build UI -------------------------------------------------------
        self._build_ui()
        self._log("Application started.", color=ACCENT_CYAN)
        if self._model_loaded:
            self._log(f"Model loaded. Classes: {self._classifier.classes}", color=ACCENT_GREEN)
        else:
            self._log("No trained model found. Collect data and train first.", color=ACCENT_WARN)

    # ======================================================================
    # UI Construction
    # ======================================================================

    def _build_ui(self):
        # ── Main columns ──────────────────────────────────────────────────
        self.grid_columnconfigure(0, weight=3)   # video
        self.grid_columnconfigure(1, weight=1)   # control panel
        self.grid_rowconfigure(0, weight=1)

        # ── Left: video frame ─────────────────────────────────────────────
        self._video_frame = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=12)
        self._video_frame.grid(row=0, column=0, padx=(12, 6), pady=12, sticky="nsew")
        self._video_frame.grid_rowconfigure(0, weight=1)
        self._video_frame.grid_columnconfigure(0, weight=1)

        self._video_label = ctk.CTkLabel(
            self._video_frame, text="Camera feed will appear here",
            fg_color="#0d0d1a", corner_radius=8, text_color="#555577",
            font=("Segoe UI", 14),
        )
        self._video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # ── Right: control panel ──────────────────────────────────────────
        panel = ctk.CTkScrollableFrame(self, fg_color=BG_PANEL, corner_radius=12)
        panel.grid(row=0, column=1, padx=(6, 12), pady=12, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)

        row = 0

        # Title
        ctk.CTkLabel(panel, text="🖐 Hand Gesture AI", font=FONT_TITLE,
                     text_color=ACCENT_CYAN).grid(row=row, column=0, pady=(16, 4))
        row += 1
        ctk.CTkLabel(panel, text="Virtual Mouse Controller", font=("Segoe UI", 12),
                     text_color="#778899").grid(row=row, column=0, pady=(0, 16))
        row += 1

        # ── Gesture & Stats Display ───────────────────────────────────────
        stats_frame = ctk.CTkFrame(panel, fg_color=ACCENT_BLUE, corner_radius=10)
        stats_frame.grid(row=row, column=0, padx=8, pady=4, sticky="ew"); row += 1
        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(stats_frame, text="Gesture", font=FONT_LABEL, text_color="#aaaacc").grid(
            row=0, column=0, padx=12, pady=(10, 0))
        ctk.CTkLabel(stats_frame, text="Confidence", font=FONT_LABEL, text_color="#aaaacc").grid(
            row=0, column=1, padx=12, pady=(10, 0))

        self._gesture_lbl = ctk.CTkLabel(stats_frame, text="🤚  IDLE",
                                          font=("Segoe UI", 20, "bold"), text_color=ACCENT_CYAN)
        self._gesture_lbl.grid(row=1, column=0, padx=12, pady=(0, 10))

        self._conf_lbl = ctk.CTkLabel(stats_frame, text="—",
                                       font=("Segoe UI", 20, "bold"), text_color=ACCENT_CYAN)
        self._conf_lbl.grid(row=1, column=1, padx=12, pady=(0, 10))

        ctk.CTkLabel(stats_frame, text="FPS", font=FONT_LABEL, text_color="#aaaacc").grid(
            row=2, column=0, padx=12, pady=(0, 0))
        ctk.CTkLabel(stats_frame, text="Mouse", font=FONT_LABEL, text_color="#aaaacc").grid(
            row=2, column=1, padx=12, pady=(0, 0))

        self._fps_lbl = ctk.CTkLabel(stats_frame, text="—",
                                      font=FONT_VALUE, text_color=ACCENT_GREEN)
        self._fps_lbl.grid(row=3, column=0, padx=12, pady=(0, 12))

        self._mouse_status_lbl = ctk.CTkLabel(stats_frame, text="OFF",
                                               font=FONT_VALUE, text_color=ACCENT_RED)
        self._mouse_status_lbl.grid(row=3, column=1, padx=12, pady=(0, 12))

        # ── Sensitivity ───────────────────────────────────────────────────
        ctk.CTkLabel(panel, text="Sensitivity", font=FONT_LABEL,
                     text_color="#aaaacc").grid(row=row, column=0, padx=8, pady=(12, 0), sticky="w")
        row += 1
        self._sensitivity_slider = ctk.CTkSlider(
            panel, from_=0.5, to=5.0, number_of_steps=45,
            command=self._on_sensitivity_change
        )
        self._sensitivity_slider.set(1.8)
        self._sensitivity_slider.grid(row=row, column=0, padx=8, pady=4, sticky="ew"); row += 1
        self._sensitivity_val_lbl = ctk.CTkLabel(panel, text="1.8", font=FONT_LABEL,
                                                  text_color="#aaaacc")
        self._sensitivity_val_lbl.grid(row=row, column=0, pady=(0, 8)); row += 1

        # ── Camera Controls ───────────────────────────────────────────────
        self._sep(panel, row); row += 1
        ctk.CTkLabel(panel, text="Camera", font=("Segoe UI", 13, "bold"),
                     text_color="#aaaacc").grid(row=row, column=0, pady=(6, 2)); row += 1

        self._btn_start_cam = ctk.CTkButton(
            panel, text="▶  Start Camera", fg_color=ACCENT_GREEN,
            hover_color="#27ae60", font=FONT_LABEL, command=self._start_camera
        )
        self._btn_start_cam.grid(row=row, column=0, padx=8, pady=4, sticky="ew"); row += 1

        self._btn_stop_cam = ctk.CTkButton(
            panel, text="⏹  Stop Camera", fg_color=ACCENT_RED,
            hover_color="#c0392b", font=FONT_LABEL, command=self._stop_camera,
            state="disabled"
        )
        self._btn_stop_cam.grid(row=row, column=0, padx=8, pady=4, sticky="ew"); row += 1

        # ── Mouse Controls ────────────────────────────────────────────────
        self._sep(panel, row); row += 1
        ctk.CTkLabel(panel, text="Mouse Control", font=("Segoe UI", 13, "bold"),
                     text_color="#aaaacc").grid(row=row, column=0, pady=(6, 2)); row += 1

        self._btn_enable_mouse = ctk.CTkButton(
            panel, text="🖱  Enable Mouse Control", fg_color=ACCENT_CYAN,
            hover_color="#0090b8", font=FONT_LABEL, command=self._enable_mouse,
            state="disabled"
        )
        self._btn_enable_mouse.grid(row=row, column=0, padx=8, pady=4, sticky="ew"); row += 1

        self._btn_pause_mouse = ctk.CTkButton(
            panel, text="⏸  Pause Mouse Control", fg_color=ACCENT_WARN,
            hover_color="#d68910", font=FONT_LABEL, command=self._pause_mouse,
            state="disabled"
        )
        self._btn_pause_mouse.grid(row=row, column=0, padx=8, pady=4, sticky="ew"); row += 1

        # ── ML Tools ──────────────────────────────────────────────────────
        self._sep(panel, row); row += 1
        ctk.CTkLabel(panel, text="Machine Learning", font=("Segoe UI", 13, "bold"),
                     text_color="#aaaacc").grid(row=row, column=0, pady=(6, 2)); row += 1

        self._btn_collect = ctk.CTkButton(
            panel, text="📊  Collect Data", fg_color="#8e44ad",
            hover_color="#7d3c98", font=FONT_LABEL, command=self._open_collection_window
        )
        self._btn_collect.grid(row=row, column=0, padx=8, pady=4, sticky="ew"); row += 1

        self._btn_train = ctk.CTkButton(
            panel, text="🧠  Train Model", fg_color="#2980b9",
            hover_color="#2471a3", font=FONT_LABEL, command=self._train_model
        )
        self._btn_train.grid(row=row, column=0, padx=8, pady=4, sticky="ew"); row += 1

        self._model_acc_lbl = ctk.CTkLabel(panel, text="Model accuracy: N/A",
                                            font=FONT_LABEL, text_color="#aaaacc")
        self._model_acc_lbl.grid(row=row, column=0, pady=(4, 8)); row += 1

        # ── Emergency Stop ────────────────────────────────────────────────
        self._sep(panel, row); row += 1
        self._btn_emergency = ctk.CTkButton(
            panel, text="🛑  EMERGENCY STOP (ESC)", fg_color="#7b241c",
            hover_color="#922b21", font=("Segoe UI", 12, "bold"),
            command=self._emergency_stop
        )
        self._btn_emergency.grid(row=row, column=0, padx=8, pady=(8, 4), sticky="ew"); row += 1
        ctk.CTkLabel(panel, text="Or move mouse to screen corner",
                     font=("Segoe UI", 10), text_color="#556677").grid(
            row=row, column=0, pady=(0, 8)); row += 1

        # ── Log ───────────────────────────────────────────────────────────
        self._sep(panel, row); row += 1
        ctk.CTkLabel(panel, text="Status Log", font=("Segoe UI", 13, "bold"),
                     text_color="#aaaacc").grid(row=row, column=0, pady=(6, 2)); row += 1

        self._log_box = ctk.CTkTextbox(panel, height=150, font=FONT_LOG,
                                        fg_color="#0d0d1a", text_color="#aaddff",
                                        activate_scrollbars=True, state="disabled")
        self._log_box.grid(row=row, column=0, padx=8, pady=4, sticky="ew"); row += 1

    def _sep(self, parent, row):
        ctk.CTkFrame(parent, height=1, fg_color="#2a2a4a").grid(
            row=row, column=0, padx=8, pady=4, sticky="ew"
        )

    # ======================================================================
    # Camera Loop
    # ======================================================================

    def _start_camera(self):
        if self._running:
            return
        if not self._camera.open():
            self._log("ERROR: Cannot open camera. Check connection.", color=ACCENT_RED)
            messagebox.showerror("Camera Error",
                                 "Could not open webcam.\n"
                                 "Check that no other app is using it.")
            return
        self._running = True
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()
        self._btn_start_cam.configure(state="disabled")
        self._btn_stop_cam.configure(state="normal")
        self._btn_enable_mouse.configure(state="normal")
        self._log("Camera started.", color=ACCENT_GREEN)

    def _stop_camera(self):
        self._mouse_enabled = False
        self._mouse.stop()
        self._running = False
        time.sleep(0.3)
        self._camera.release()
        self._btn_start_cam.configure(state="normal")
        self._btn_stop_cam.configure(state="disabled")
        self._btn_enable_mouse.configure(state="disabled")
        self._btn_pause_mouse.configure(state="disabled")
        self._video_label.configure(image=None, text="Camera stopped")
        self._update_mouse_status_label()
        self._log("Camera stopped.", color=ACCENT_WARN)

    def _camera_loop(self):
        """Runs in a background thread."""
        while self._running:
            ret, frame = self._camera.read()
            if not ret:
                time.sleep(0.02)
                continue

            # Mirror the frame (more intuitive for the user)
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # ---- Hand detection ----
            hand_found = self._tracker.process(frame)
            self._tracker.draw(frame)

            gesture    = "IDLE"
            confidence = 0.0

            if hand_found and self._tracker.landmarks:
                fv = extract_features(self._tracker.landmarks)
                if fv is not None and self._model_loaded:
                    gesture, confidence = self._classifier.predict(fv)

                # ---- Mouse control ----
                if self._mouse_enabled:
                    self._mouse.update(
                        gesture, confidence,
                        self._tracker.landmarks,
                        (h, w),
                    )
            else:
                if self._mouse_enabled:
                    self._mouse.update("IDLE", 0.0, None, (h, w))

            # ---- Overlay ----
            self._draw_overlay(frame, gesture, confidence, self._camera.fps)

            # ---- Send to UI (thread-safe) ----
            self._update_video(frame)
            self.after(0, self._update_stats, gesture, confidence, self._camera.fps)

        logger.info("Camera loop exited.")

    def _draw_overlay(self, frame, gesture, confidence, fps):
        h, w = frame.shape[:2]
        color = tuple(int(c) for c in
                      bytes.fromhex(GESTURE_COLORS.get(gesture, "#7f8c8d").lstrip("#")))
        color_bgr = (color[2], color[1], color[0])

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (10, 10, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"Gesture: {gesture}", (10, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, color_bgr, 2)
        cv2.putText(frame, f"Conf: {confidence:.0%}", (280, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 110, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 200), 1)

        # Mouse active indicator
        if self._mouse_enabled:
            cv2.circle(frame, (w - 20, h - 20), 8, (0, 255, 0), -1)
            cv2.putText(frame, "MOUSE ON", (w - 130, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ESC hint
        cv2.putText(frame, "ESC: emergency stop", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 150), 1)

    def _update_video(self, frame: np.ndarray):
        """Convert BGR frame to CTk-compatible image and display."""
        try:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb)
            # Resize to fit the video label
            lw = self._video_label.winfo_width()
            lh = self._video_label.winfo_height()
            if lw > 10 and lh > 10:
                img = img.resize((lw, lh), Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self.after(0, self._video_label.configure, {"image": ctk_img, "text": ""})
        except Exception as exc:
            logger.debug("Video update error: %s", exc)

    def _update_stats(self, gesture: str, confidence: float, fps: float):
        icon  = GESTURE_ICONS.get(gesture, "🤚")
        color = GESTURE_COLORS.get(gesture, "#7f8c8d")
        self._gesture_lbl.configure(text=f"{icon}  {gesture}", text_color=color)
        self._conf_lbl.configure(text=f"{confidence:.0%}")
        self._fps_lbl.configure(text=f"{fps:.0f}")

    # ======================================================================
    # Mouse Control
    # ======================================================================

    def _enable_mouse(self):
        if not self._model_loaded:
            messagebox.showwarning("No Model",
                                   "No trained model found.\n"
                                   "Please train the model first (🧠 Train Model).")
            return
        if not self._running:
            messagebox.showwarning("Camera Off", "Start the camera first.")
            return
        self._mouse_enabled = True
        self._mouse.resume()
        self._btn_pause_mouse.configure(state="normal")
        self._btn_enable_mouse.configure(state="disabled")
        self._update_mouse_status_label()
        self._log("Mouse control ENABLED. ESC to stop.", color=ACCENT_GREEN)

    def _pause_mouse(self):
        self._mouse_enabled = False
        self._btn_enable_mouse.configure(state="normal")
        self._btn_pause_mouse.configure(state="disabled")
        self._update_mouse_status_label()
        self._log("Mouse control PAUSED.", color=ACCENT_WARN)

    def _update_mouse_status_label(self):
        if self._mouse_enabled:
            self._mouse_status_lbl.configure(text="ON", text_color=ACCENT_GREEN)
        else:
            self._mouse_status_lbl.configure(text="OFF", text_color=ACCENT_RED)

    def _emergency_stop(self, event=None):
        self._mouse_enabled = False
        self._mouse.stop()
        self._btn_enable_mouse.configure(state="normal" if self._running else "disabled")
        self._btn_pause_mouse.configure(state="disabled")
        self._update_mouse_status_label()
        self._log("EMERGENCY STOP activated!", color=ACCENT_RED)

    # ======================================================================
    # ML Tools
    # ======================================================================

    def _open_collection_window(self):
        """Launch data collection in a sub-window."""
        win = ctk.CTkToplevel(self)
        win.title("Data Collection")
        win.geometry("420x320")
        win.configure(fg_color=BG_DARK)
        win.grab_set()

        from src.data_collection import GESTURE_CLASSES

        ctk.CTkLabel(win, text="Collect Gesture Samples",
                     font=FONT_TITLE, text_color=ACCENT_CYAN).pack(pady=(16, 8))

        ctk.CTkLabel(win, text="Select gesture:", font=FONT_LABEL).pack()
        gesture_var = ctk.StringVar(value=GESTURE_CLASSES[0])
        gesture_menu = ctk.CTkOptionMenu(win, values=GESTURE_CLASSES, variable=gesture_var)
        gesture_menu.pack(pady=8, padx=24, fill="x")

        ctk.CTkLabel(win, text="Samples to collect:", font=FONT_LABEL).pack()
        samples_entry = ctk.CTkEntry(win, placeholder_text="200")
        samples_entry.insert(0, "200")
        samples_entry.pack(pady=8, padx=24, fill="x")

        def _start_collection():
            gesture = gesture_var.get()
            try:
                n = int(samples_entry.get())
            except ValueError:
                messagebox.showerror("Input Error", "Samples must be an integer.", parent=win)
                return

            win.destroy()
            # If camera is currently active in the main app, stop it so data_collection can use the webcam
            if self._running:
                self._stop_camera()
                self._log("Paused app camera to allow data collection window to access webcam.", color=ACCENT_WARN)

            self._log(f"Starting collection for: {gesture} ({n} samples)", color="#9b59b6")
            # Run in subprocess so it gets its own window
            script = os.path.join(os.path.dirname(__file__), "data_collection.py")
            subprocess.Popen(
                [sys.executable, script, "--gesture", gesture, "--samples", str(n)],
                cwd=os.path.join(os.path.dirname(__file__), ".."),
            )

        ctk.CTkButton(win, text="▶  Start Collection", fg_color="#8e44ad",
                      command=_start_collection).pack(pady=16, padx=24, fill="x")

    def _train_model(self):
        if messagebox.askyesno(
            "Train Model",
            "This will retrain the gesture classifier.\n"
            "Make sure you have collected enough samples first.\n\n"
            "Continue?"
        ):
            self._log("Training model... (this may take a minute)", color=ACCENT_CYAN)
            self._btn_train.configure(state="disabled", text="⏳  Training...")

            def _do_train():
                script = os.path.join(os.path.dirname(__file__), "train_model.py")
                result = subprocess.run(
                    [sys.executable, script],
                    capture_output=True, text=True,
                    cwd=os.path.join(os.path.dirname(__file__), ".."),
                )
                self.after(0, self._on_train_complete, result)

            threading.Thread(target=_do_train, daemon=True).start()

    def _on_train_complete(self, result: subprocess.CompletedProcess):
        self._btn_train.configure(state="normal", text="🧠  Train Model")
        if result.returncode == 0:
            # Reload model
            self._model_loaded = self._classifier.load()
            self._log("Training complete! Model reloaded.", color=ACCENT_GREEN)
            # Try to parse accuracy from stdout
            for line in result.stdout.splitlines():
                if "accuracy" in line.lower() or "Acc:" in line:
                    self._model_acc_lbl.configure(text=f"Model: {line.strip()[:40]}")
                    break
        else:
            self._log(f"Training failed: {result.stderr[:120]}", color=ACCENT_RED)
            messagebox.showerror("Training Failed",
                                 f"Model training failed.\n\nError:\n{result.stderr[:300]}")

    # ======================================================================
    # Logging
    # ======================================================================

    def _log(self, message: str, color: str = "#aaddff"):
        timestamp = time.strftime("%H:%M:%S")
        self._log_box.configure(state="normal")
        self._log_box.insert("end", f"[{timestamp}] {message}\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    # ======================================================================
    # Sensitivity
    # ======================================================================

    def _on_sensitivity_change(self, value: float):
        self._mouse.set_sensitivity(value)
        self._sensitivity_val_lbl.configure(text=f"{value:.1f}")

    # ======================================================================
    # Lifecycle
    # ======================================================================

    def _on_close(self):
        if messagebox.askyesno("Exit", "Exit the Hand Gesture AI application?"):
            self._running = False
            self._mouse.stop()
            self._camera.release()
            self._tracker.close()
            self.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
