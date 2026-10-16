"""
Drone Reconnaissance Application — Main CustomTkinter UI.
Handles video input, inference threading, and display.
"""

import os
import sys
import time
import queue
import threading
from tkinter import filedialog
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from ui.theme import (
    BG_DARK, BG_PANEL, BG_CARD, NEON_CYAN, NEON_BLUE,
    TACTICAL_RED, TACTICAL_GREEN, TEXT_PRIMARY, TEXT_SECONDARY,
    TEXT_MUTED, BORDER_COLOR, BORDER_ACCENT,
    FONT_TITLE, FONT_HEADING, FONT_BODY, FONT_SMALL, FONT_MONO,
    BTN_PRIMARY, BTN_DANGER, BTN_SECONDARY,
)
from engine.detector import DroneDetector
from engine.tracker import MultiObjectTracker
from hud.overlay import HUDOverlay


class DroneReconApp(ctk.CTk):
    """Main application window for the Drone Reconnaissance System."""

    def __init__(self):
        super().__init__()

        # ── Window setup ──
        self.title(cfg.WINDOW_TITLE)
        self.geometry(cfg.WINDOW_SIZE)
        self.minsize(1024, 600)
        self.configure(fg_color=BG_DARK)

        # ── State ──
        self.running = False          # Is processing active?
        self.cap: Optional[cv2.VideoCapture] = None
        self.process_thread: Optional[threading.Thread] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=3)
        self.stop_event = threading.Event()

        # ── Timeline / Seek state ──
        self._total_frames = 0
        self._total_duration = 0.0
        self._current_frame_pos = 0
        self._video_fps_source = 30.0
        self._seek_to = -1            # Frame to seek to (-1 = none)
        self._is_video_file = False
        self._is_paused = False
        self._user_seeking = False

        # ── AI modules (lazy-loaded) ──
        self._detector: Optional[DroneDetector] = None
        self._tracker: Optional[MultiObjectTracker] = None
        self._hud: Optional[HUDOverlay] = None

        # ── Metrics ──
        self._fps = 0.0
        self._inference_ms = 0.0
        self._n_targets = 0
        self._model_status = "NOT LOADED"

        # ── Build UI ──
        self._build_layout()

        # ── Window close handler ──
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # ── Start display polling ──
        self.after(30, self._poll_display)

    # ──────────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ──────────────────────────────────────────────────────────────────

    def _build_layout(self):
        """Build the four-section layout: sidebar + canvas + timeline + status bar."""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_canvas()
        self._build_timeline()
        self._build_statusbar()

    def _build_sidebar(self):
        """Build the left sidebar with controls."""
        self.sidebar = ctk.CTkFrame(
            self, width=cfg.SIDEBAR_WIDTH, corner_radius=0,
            fg_color=BG_PANEL, border_width=1, border_color=BORDER_COLOR,
        )
        self.sidebar.grid(row=0, column=0, rowspan=3, sticky="nsw")
        self.sidebar.grid_propagate(False)



        # ── Logo / Title ──
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.pack(fill="x", padx=16, pady=(18, 4))

        ctk.CTkLabel(
            logo_frame, text="⬡  DRONE RECON",
            font=FONT_TITLE, text_color=NEON_CYAN, anchor="w",
        ).pack(fill="x")

        ctk.CTkLabel(
            logo_frame, text="Tactical Reconnaissance System",
            font=FONT_SMALL, text_color=TEXT_MUTED, anchor="w",
        ).pack(fill="x")

        self._separator(self.sidebar)

        # ── Input buttons ──
        self.btn_upload = ctk.CTkButton(
            self.sidebar, text="📁  UPLOAD MEDIA",
            command=self._on_upload_media, **BTN_PRIMARY,
        )
        self.btn_upload.pack(fill="x", padx=16, pady=(0, 8))

        self.btn_stream = ctk.CTkButton(
            self.sidebar, text="📡  LIVE STREAM",
            command=self._on_live_stream, **BTN_PRIMARY,
        )
        self.btn_stream.pack(fill="x", padx=16, pady=(0, 8))

        # RTSP URL entry
        ctk.CTkLabel(
            self.sidebar, text="RTSP / URL:", font=FONT_SMALL,
            text_color=TEXT_SECONDARY, anchor="w",
        ).pack(fill="x", padx=18, pady=(4, 2))

        self.rtsp_entry = ctk.CTkEntry(
            self.sidebar, placeholder_text="rtsp://192.168.1.1:554/stream",
            font=FONT_MONO, fg_color=BG_CARD, border_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY, height=32,
        )
        self.rtsp_entry.pack(fill="x", padx=16, pady=(0, 4))

        self._separator(self.sidebar)

        # ── Confidence slider ──
        ctk.CTkLabel(
            self.sidebar, text="CONFIDENCE THRESHOLD", font=FONT_SMALL,
            text_color=TEXT_SECONDARY, anchor="w",
        ).pack(fill="x", padx=18, pady=(4, 2))

        self.conf_var = ctk.DoubleVar(value=cfg.CONFIDENCE_THRESHOLD)
        self.conf_label = ctk.CTkLabel(
            self.sidebar,
            text=f"{cfg.CONFIDENCE_THRESHOLD:.2f}",
            font=FONT_BODY, text_color=NEON_CYAN,
        )
        self.conf_label.pack(padx=16)

        self.conf_slider = ctk.CTkSlider(
            self.sidebar, from_=0.10, to=0.90, variable=self.conf_var,
            command=self._on_conf_change,
            button_color=NEON_CYAN, button_hover_color=NEON_BLUE,
            progress_color=NEON_CYAN, fg_color=BG_CARD, height=16,
        )
        self.conf_slider.pack(fill="x", padx=16, pady=(0, 4))

        self._separator(self.sidebar)

        # ── Model info ──
        info_frame = ctk.CTkFrame(self.sidebar, fg_color=BG_CARD,
                                  corner_radius=6, border_width=1,
                                  border_color=BORDER_COLOR)
        info_frame.pack(fill="x", padx=16, pady=(0, 8))

        self.lbl_model = ctk.CTkLabel(
            info_frame, text="MODEL: YOLOv11n", font=FONT_SMALL,
            text_color=TEXT_SECONDARY, anchor="w",
        )
        self.lbl_model.pack(fill="x", padx=10, pady=(6, 0))

        self.lbl_engine = ctk.CTkLabel(
            info_frame, text="ENGINE: NOT LOADED", font=FONT_SMALL,
            text_color=TEXT_MUTED, anchor="w",
        )
        self.lbl_engine.pack(fill="x", padx=10, pady=(0, 6))

        self.btn_export = ctk.CTkButton(
            self.sidebar, text="⚡  EXPORT TensorRT",
            command=self._on_export_trt, **BTN_SECONDARY,
        )
        self.btn_export.pack(fill="x", padx=16, pady=(0, 8))

        self._separator(self.sidebar)

        # ── Live metrics ──
        metrics_frame = ctk.CTkFrame(self.sidebar, fg_color=BG_CARD,
                                     corner_radius=6, border_width=1,
                                     border_color=BORDER_COLOR)
        metrics_frame.pack(fill="x", padx=16, pady=(0, 8))

        self.lbl_fps = ctk.CTkLabel(
            metrics_frame, text="FPS: —", font=FONT_BODY,
            text_color=NEON_CYAN, anchor="w",
        )
        self.lbl_fps.pack(fill="x", padx=10, pady=(6, 0))

        self.lbl_inf = ctk.CTkLabel(
            metrics_frame, text="INF: — ms", font=FONT_BODY,
            text_color=NEON_CYAN, anchor="w",
        )
        self.lbl_inf.pack(fill="x", padx=10)

        self.lbl_targets = ctk.CTkLabel(
            metrics_frame, text="TARGETS: 0", font=FONT_BODY,
            text_color=TEXT_PRIMARY, anchor="w",
        )
        self.lbl_targets.pack(fill="x", padx=10, pady=(0, 6))

        # ── Stop button (at bottom) ──
        spacer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

        self.btn_stop = ctk.CTkButton(
            self.sidebar, text="⏹  STOP",
            command=self._on_stop, **BTN_DANGER, state="disabled",
        )
        self.btn_stop.pack(fill="x", padx=16, pady=(0, 16))

    def _build_canvas(self):
        """Build the main video display area."""
        self.canvas_frame = ctk.CTkFrame(
            self, fg_color=BG_DARK, corner_radius=0,
        )
        self.canvas_frame.grid(row=0, column=1, sticky="nsew")
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=1)

        self.display_label = ctk.CTkLabel(
            self.canvas_frame, text="", fg_color=BG_DARK,
        )
        self.display_label.grid(row=0, column=0, sticky="nsew")

        # Show splash screen
        self._show_splash()

    def _build_statusbar(self):
        """Build the bottom status bar."""
        self.statusbar = ctk.CTkFrame(
            self, height=28, corner_radius=0,
            fg_color=BG_PANEL, border_width=1, border_color=BORDER_COLOR,
        )
        self.statusbar.grid(row=2, column=1, sticky="ew")
        self.statusbar.grid_propagate(False)

        self.lbl_status = ctk.CTkLabel(
            self.statusbar,
            text="  ● SYSTEM READY — Select input source to begin.",
            font=FONT_MONO, text_color=TEXT_SECONDARY, anchor="w",
        )
        self.lbl_status.pack(fill="x", padx=8, pady=2)

    def _build_timeline(self):
        """Build the video timeline / seek bar between canvas and status bar."""
        self.timeline_frame = ctk.CTkFrame(
            self, height=44, corner_radius=0,
            fg_color=BG_PANEL, border_width=1, border_color=BORDER_COLOR,
        )
        self.timeline_frame.grid(row=1, column=1, sticky="ew")
        self.timeline_frame.grid_propagate(False)
        self.timeline_frame.grid_columnconfigure(2, weight=1)

        # Play / Pause button
        self.btn_play_pause = ctk.CTkButton(
            self.timeline_frame, text="⏸", width=36, height=28,
            font=("Consolas", 14), corner_radius=4,
            fg_color=BG_CARD, hover_color="#1a1a35",
            border_width=1, border_color=BORDER_COLOR,
            text_color=NEON_CYAN, command=self._on_play_pause,
        )
        self.btn_play_pause.grid(row=0, column=0, padx=(8, 4), pady=6)

        # Current time label
        self.lbl_time_current = ctk.CTkLabel(
            self.timeline_frame, text="00:00",
            font=("Consolas", 10), text_color=NEON_CYAN, width=48,
        )
        self.lbl_time_current.grid(row=0, column=1, padx=(4, 4), pady=6)

        # Timeline slider
        self.timeline_var = ctk.DoubleVar(value=0.0)
        self.timeline_slider = ctk.CTkSlider(
            self.timeline_frame, from_=0.0, to=1.0,
            variable=self.timeline_var,
            command=self._on_timeline_drag,
            button_color=NEON_CYAN, button_hover_color=NEON_BLUE,
            progress_color="#004466", fg_color=BG_CARD, height=14,
        )
        self.timeline_slider.grid(row=0, column=2, sticky="ew", padx=4, pady=6)

        # Bind press/release to track user seeking
        self.timeline_slider.bind("<ButtonPress-1>", self._on_seek_start)
        self.timeline_slider.bind("<ButtonRelease-1>", self._on_seek_end)

        # Total time label
        self.lbl_time_total = ctk.CTkLabel(
            self.timeline_frame, text="00:00",
            font=("Consolas", 10), text_color=TEXT_SECONDARY, width=48,
        )
        self.lbl_time_total.grid(row=0, column=3, padx=(4, 4), pady=6)

        # Frame counter label
        self.lbl_frame_num = ctk.CTkLabel(
            self.timeline_frame, text="F: 0 / 0",
            font=("Consolas", 9), text_color=TEXT_MUTED, width=90,
        )
        self.lbl_frame_num.grid(row=0, column=4, padx=(4, 8), pady=6)

        # Start hidden (only show for video files)
        self.timeline_frame.grid_remove()

    # ──────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────

    def _separator(self, parent):
        """Add a horizontal separator line."""
        ctk.CTkFrame(
            parent, height=1, fg_color=BORDER_COLOR,
        ).pack(fill="x", padx=16, pady=10)

    def _show_splash(self):
        """Show tactical splash / idle screen."""
        w = max(800, self.canvas_frame.winfo_width())
        h = max(500, self.canvas_frame.winfo_height())

        splash = np.zeros((h, w, 3), dtype=np.uint8)
        splash[:] = (12, 12, 18)

        cx, cy = w // 2, h // 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Center text
        lines = [
            ("DRONE RECONNAISSANCE SYSTEM", 0.8, (0, 255, 255), -60),
            ("TACTICAL HUD v1.0", 0.5, (100, 100, 120), -20),
            ("Awaiting input source...", 0.45, (80, 80, 100), 30),
            ("[UPLOAD MEDIA]  or  [LIVE STREAM]", 0.4, (0, 180, 180), 65),
        ]
        for text, scale, color, dy in lines:
            text_size = cv2.getTextSize(text, font, scale, 1)[0]
            tx = cx - text_size[0] // 2
            ty = cy + dy
            cv2.putText(splash, text, (tx, ty), font, scale, color, 1,
                        cv2.LINE_AA)

        # Decorative corners
        corner_len = 40
        margin = 50
        for (x, y), (dx, dy) in [
            ((margin, margin), (1, 1)),
            ((w - margin, margin), (-1, 1)),
            ((margin, h - margin), (1, -1)),
            ((w - margin, h - margin), (-1, -1)),
        ]:
            cv2.line(splash, (x, y), (x + dx * corner_len, y),
                     (0, 255, 255), 1)
            cv2.line(splash, (x, y), (x, y + dy * corner_len),
                     (0, 255, 255), 1)

        self._display_frame(splash)

    def _display_frame(self, frame: np.ndarray):
        """Convert an OpenCV BGR frame to CTkImage and display it."""
        try:
            # Get display area size
            disp_w = max(400, self.canvas_frame.winfo_width() - 4)
            disp_h = max(300, self.canvas_frame.winfo_height() - 4)

            fh, fw = frame.shape[:2]
            # Fit frame into display area maintaining aspect ratio
            scale = min(disp_w / fw, disp_h / fh)
            new_w = int(fw * scale)
            new_h = int(fh * scale)

            if new_w <= 0 or new_h <= 0:
                return

            resized = cv2.resize(frame, (new_w, new_h),
                                 interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            ctk_img = ctk.CTkImage(
                light_image=pil_img, dark_image=pil_img,
                size=(new_w, new_h),
            )
            self.display_label.configure(image=ctk_img, text="")
            self.display_label._ctk_image = ctk_img  # prevent GC
        except Exception:
            pass  # Silently handle display errors during shutdown

    def _set_status(self, text: str, color: str = TEXT_SECONDARY):
        """Update status bar text."""
        self.lbl_status.configure(text=f"  ● {text}", text_color=color)

    def _set_controls_running(self, running: bool):
        """Toggle button states based on processing state."""
        state_input = "disabled" if running else "normal"
        state_stop = "normal" if running else "disabled"
        self.btn_upload.configure(state=state_input)
        self.btn_stream.configure(state=state_input)
        self.btn_stop.configure(state=state_stop)
        self.btn_export.configure(state=state_input)

    # ──────────────────────────────────────────────────────────────────
    # MODEL LOADING
    # ──────────────────────────────────────────────────────────────────

    def _ensure_model(self) -> bool:
        """Lazy-load the detection model. Returns True if ready."""
        if self._detector is not None:
            return True

        self._set_status("Loading YOLOv11 model — please wait...", NEON_CYAN)
        self.update_idletasks()

        try:
            self._detector = DroneDetector()
            self._tracker = MultiObjectTracker()
            self._model_status = self._detector.engine_type
            self.lbl_engine.configure(
                text=f"ENGINE: {self._model_status}",
                text_color=TACTICAL_GREEN,
            )
            self._set_status("Model loaded — ready.", TACTICAL_GREEN)
            return True
        except Exception as e:
            self._set_status(f"Model load FAILED: {e}", TACTICAL_RED)
            self._detector = None
            return False

    # ──────────────────────────────────────────────────────────────────
    # EVENT HANDLERS
    # ──────────────────────────────────────────────────────────────────

    def _on_upload_media(self):
        """Handle Upload Media button click."""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(
            title="Select Media File", filetypes=filetypes,
        )
        if not path:
            return

        if not self._ensure_model():
            return

        self._start_source(path)

    def _on_live_stream(self):
        """Handle Live Stream button click."""
        if not self._ensure_model():
            return

        url = self.rtsp_entry.get().strip()
        source = url if url else 0  # Default to webcam
        self._start_source(source)

    def _on_stop(self):
        """Stop all processing."""
        self.stop_event.set()
        self.running = False
        self._is_paused = False
        self._set_controls_running(False)
        self._set_status("Processing stopped.", TEXT_SECONDARY)

        # Wait for thread to finish
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=3)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self._tracker:
            self._tracker.reset()

        # Hide timeline and reset
        self.timeline_frame.grid_remove()
        self.timeline_var.set(0.0)
        self._is_video_file = False

        # Clear display
        self.after(200, self._show_splash)

    def _on_conf_change(self, value):
        """Confidence slider changed."""
        val = round(value, 2)
        self.conf_label.configure(text=f"{val:.2f}")
        if self._detector:
            self._detector.update_confidence(val)

    def _on_export_trt(self):
        """Export model to TensorRT."""
        if not self._ensure_model():
            return

        self._set_status("Exporting TensorRT engine — this may take minutes...",
                         TACTICAL_YELLOW)
        self.btn_export.configure(state="disabled")
        self.update_idletasks()

        def _do_export():
            result = self._detector.export_tensorrt()
            self.after(0, lambda: self._on_export_done(result))

        threading.Thread(target=_do_export, daemon=True).start()

    def _on_export_done(self, engine_path):
        """Called when TRT export finishes."""
        if engine_path:
            self._set_status(f"TensorRT saved: {engine_path}", TACTICAL_GREEN)
        else:
            self._set_status("TensorRT export failed.", TACTICAL_RED)
        self.btn_export.configure(state="normal")

    def _on_closing(self):
        """Clean up on window close."""
        self.stop_event.set()
        self.running = False

        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2)

        if self.cap is not None:
            self.cap.release()

        self.destroy()

    # ──────────────────────────────────────────────────────────────────
    # VIDEO PROCESSING
    # ──────────────────────────────────────────────────────────────────

    def _start_source(self, source):
        """Open a video source and start the processing thread."""
        # Stop any existing processing
        if self.running:
            self._on_stop()
            time.sleep(0.3)

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self._set_status(f"Cannot open source: {source}", TACTICAL_RED)
            return

        # Read one frame to get dimensions
        ret, test_frame = self.cap.read()
        if not ret:
            self._set_status("Cannot read from source.", TACTICAL_RED)
            self.cap.release()
            self.cap = None
            return

        h, w = test_frame.shape[:2]
        self._hud = HUDOverlay(w, h)

        # Reset tracker
        self._tracker.reset()

        # Detect if this is a seekable video file
        is_image = isinstance(source, str) and source.lower().split('.')[-1] in (
            'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp',
        )
        self._is_video_file = isinstance(source, str) and not is_image

        # Get video metadata for timeline
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_fps_source = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_duration = (
            self._total_frames / self._video_fps_source
            if self._video_fps_source > 0 else 0.0
        )
        self._current_frame_pos = 0
        self._seek_to = -1
        self._is_paused = False

        # Show/hide timeline
        if self._is_video_file and self._total_frames > 1:
            self.timeline_frame.grid()
            self.lbl_time_total.configure(
                text=self._format_time(self._total_duration),
            )
            self.lbl_time_current.configure(text="00:00")
            self.lbl_frame_num.configure(
                text=f"F: 0 / {self._total_frames}",
            )
            self.timeline_var.set(0.0)
            self.btn_play_pause.configure(text="⏸")
        else:
            self.timeline_frame.grid_remove()

        # Reset source position to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Start processing
        self.running = True
        self.stop_event.clear()
        self._set_controls_running(True)

        src_name = source if isinstance(source, str) else "Webcam"
        self._set_status(f"ACTIVE — {src_name} [{w}×{h}]", TACTICAL_GREEN)

        self.process_thread = threading.Thread(
            target=self._processing_loop, args=(source,), daemon=True,
        )
        self.process_thread.start()

    def _processing_loop(self, source):
        """Background thread: read frames, detect, track, render HUD."""
        fps_counter = 0
        fps_timer = time.time()
        detect_frame_counter = 0  # Frame skip counter
        is_image = isinstance(source, str) and source.lower().split('.')[-1] in (
            'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp',
        )

        # For images, read the single frame once
        static_frame = None
        if is_image:
            ret, static_frame = self.cap.read()
            if not ret:
                self.running = False
                return

        while self.running and not self.stop_event.is_set():
            t_start = time.time()

            # ── Handle pause ──
            if self._is_paused and not is_image:
                # While paused, only process seek requests
                if self._seek_to >= 0:
                    seek_frame = self._seek_to
                    self._seek_to = -1
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
                    self._current_frame_pos = seek_frame
                    self._tracker.reset()

                    ret, frame = self.cap.read()
                    if ret:
                        h, w = frame.shape[:2]
                        detections = self._detector.detect(frame)
                        targets = self._tracker.update(detections, frame_height=h)
                        rendered = self._hud.render(
                            frame, targets,
                            inference_ms=self._detector.inference_ms,
                            fps=self._fps,
                            n_targets=len(targets),
                        )
                        try:
                            self.frame_queue.put_nowait(rendered)
                        except queue.Full:
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                self.frame_queue.put_nowait(rendered)
                            except queue.Full:
                                pass
                else:
                    time.sleep(0.05)
                continue

            # ── Handle seek request ──
            if self._seek_to >= 0:
                seek_frame = self._seek_to
                self._seek_to = -1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
                self._current_frame_pos = seek_frame
                self._tracker.reset()

            # Read frame
            if is_image:
                frame = static_frame.copy()
                ret = True
            else:
                ret, frame = self.cap.read()
                if ret:
                    self._current_frame_pos = int(
                        self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    )

            if not ret:
                # Video ended — loop or stop
                if isinstance(source, str) and not is_image:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._current_frame_pos = 0
                    self._tracker.reset()
                    continue
                else:
                    break

            # Resize for consistent processing
            h, w = frame.shape[:2]
            detect_frame_counter += 1

            # ── Frame skipping: detect every Nth, Kalman-predict others ──
            if detect_frame_counter % cfg.FRAME_SKIP == 0:
                detections = self._detector.detect(frame)
                inf_ms = self._detector.inference_ms
                targets = self._tracker.update(detections, frame_height=h)
            else:
                inf_ms = self._inference_ms  # Reuse last inference time
                targets = self._tracker.predict_only(frame_height=h)

            # ── HUD Rendering ──
            rendered = self._hud.render(
                frame, targets,
                inference_ms=inf_ms,
                fps=self._fps,
                n_targets=len(targets),
            )

            # ── FPS calculation ──
            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 0.5:
                self._fps = fps_counter / elapsed
                self._inference_ms = inf_ms
                self._n_targets = len(targets)
                fps_counter = 0
                fps_timer = time.time()

            # ── Put frame in queue (drop if full) ──
            try:
                self.frame_queue.put_nowait(rendered)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait(rendered)
                except queue.Full:
                    pass

            # Throttle to avoid burning CPU if processing is very fast
            proc_time = time.time() - t_start
            target_time = 1.0 / 60.0  # Cap at 60 fps
            if proc_time < target_time:
                time.sleep(target_time - proc_time)

        self.running = False

    def _poll_display(self):
        """Main-thread polling loop to display frames from the queue."""
        try:
            frame = self.frame_queue.get_nowait()
            self._display_frame(frame)

            # Update sidebar metrics
            self.lbl_fps.configure(text=f"FPS: {self._fps:.1f}")
            self.lbl_inf.configure(text=f"INF: {self._inference_ms:.1f} ms")

            t_color = TACTICAL_RED if self._n_targets > 0 else TEXT_PRIMARY
            self.lbl_targets.configure(
                text=f"TARGETS: {self._n_targets}", text_color=t_color,
            )

            # Update timeline slider (only if user is NOT dragging)
            if (self._is_video_file and self._total_frames > 0
                    and not self._user_seeking):
                progress = self._current_frame_pos / self._total_frames
                self.timeline_var.set(min(1.0, max(0.0, progress)))
                cur_time = self._current_frame_pos / self._video_fps_source
                self.lbl_time_current.configure(
                    text=self._format_time(cur_time),
                )
                self.lbl_frame_num.configure(
                    text=f"F: {self._current_frame_pos} / {self._total_frames}",
                )
        except queue.Empty:
            pass

        # Re-schedule
        self.after(16, self._poll_display)

    # ──────────────────────────────────────────────────────────────────
    # TIMELINE CONTROLS
    # ──────────────────────────────────────────────────────────────────

    def _on_play_pause(self):
        """Toggle play/pause state."""
        if not self.running or not self._is_video_file:
            return
        self._is_paused = not self._is_paused
        if self._is_paused:
            self.btn_play_pause.configure(text="▶")
            self._set_status("PAUSED", NEON_CYAN)
        else:
            self.btn_play_pause.configure(text="⏸")
            self._set_status("PLAYING", TACTICAL_GREEN)

    def _on_timeline_drag(self, value):
        """Called continuously while user drags the timeline slider."""
        if not self._is_video_file or self._total_frames <= 0:
            return
        target_frame = int(float(value) * self._total_frames)
        target_frame = max(0, min(target_frame, self._total_frames - 1))

        # Update the time label live during drag
        cur_time = target_frame / self._video_fps_source
        self.lbl_time_current.configure(text=self._format_time(cur_time))
        self.lbl_frame_num.configure(
            text=f"F: {target_frame} / {self._total_frames}",
        )

    def _on_seek_start(self, event):
        """User started dragging the timeline slider."""
        self._user_seeking = True

    def _on_seek_end(self, event):
        """User released the timeline slider — commit seek."""
        self._user_seeking = False
        if not self._is_video_file or self._total_frames <= 0:
            return
        value = self.timeline_var.get()
        target_frame = int(value * self._total_frames)
        target_frame = max(0, min(target_frame, self._total_frames - 1))
        self._seek_to = target_frame

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into MM:SS or HH:MM:SS."""
        seconds = max(0.0, seconds)
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
