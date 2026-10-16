"""
YOLOv11 Detection Engine — CUDA / TensorRT accelerated inference.
"""

import os
import time
from typing import List, Callable, Optional

import cv2
import numpy as np
from ultralytics import YOLO

import config as cfg
from engine.utils import Detection


class DroneDetector:
    """High-performance YOLOv11 detector with auto CUDA/TensorRT selection."""

    def __init__(
        self,
        model_path: str = cfg.MODEL_PATH,
        device: str = cfg.DEVICE,
        conf: float = cfg.CONFIDENCE_THRESHOLD,
        iou: float = cfg.IOU_THRESHOLD,
    ):
        self.device = device
        self.conf = conf
        self.iou = iou
        self.model_path = model_path
        self.inference_ms: float = 0.0

        # Load model (supports .pt and .engine)
        print(f"[DETECTOR] Loading model: {model_path} on {device}")
        self.model = YOLO(model_path)

        # Warm-up with a dummy frame
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(
            dummy, device=device, verbose=False,
            conf=conf, iou=iou, half=cfg.USE_HALF,
        )
        print("[DETECTOR] Warm-up complete — engine ready.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame, return list of Detection.

        Low-res frames (max dimension < 640px) are automatically upscaled
        before inference for better detection, then bboxes are mapped back
        to the original resolution.
        """
        t0 = time.perf_counter()

        h_orig, w_orig = frame.shape[:2]
        detect_frame = frame
        scale_factor = 1.0

        # Aggressively upscale low-res frames for better detection
        max_dim = max(h_orig, w_orig)
        if max_dim < cfg.UPSCALE_TARGET:
            scale_factor = float(cfg.UPSCALE_TARGET) / max_dim
            new_w = int(w_orig * scale_factor)
            new_h = int(h_orig * scale_factor)
            detect_frame = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC,
            )

        # Build prediction kwargs — classes=None means detect ALL
        predict_args = dict(
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            half=cfg.USE_HALF,
            verbose=False,
            max_det=300,
            agnostic_nms=True,
        )
        if cfg.TARGET_CLASSES is not None:
            predict_args["classes"] = cfg.TARGET_CLASSES

        results = self.model.predict(detect_frame, **predict_args)

        self.inference_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Detection] = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0].item())
                    conf_val = float(box.conf[0].item())

                    # Map bboxes back to original resolution
                    if scale_factor != 1.0:
                        x1 = int(x1 / scale_factor)
                        y1 = int(y1 / scale_factor)
                        x2 = int(x2 / scale_factor)
                        y2 = int(y2 / scale_factor)

                    # All 80 COCO classes are mapped to military labels
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=cls_id,
                        confidence=conf_val,
                    ))

        return detections

    def update_confidence(self, conf: float):
        """Update confidence threshold live."""
        self.conf = max(0.05, min(0.95, conf))

    def export_tensorrt(self, callback: Optional[Callable] = None):
        """Export current .pt model to TensorRT .engine (FP16)."""
        if not self.model_path.endswith(".pt"):
            print("[DETECTOR] Model is not a .pt file, skipping TRT export.")
            return None
        try:
            print("[DETECTOR] Exporting to TensorRT FP16 — this may take minutes...")
            engine_path = self.model.export(
                format="engine", device=0, half=True, imgsz=640,
            )
            print(f"[DETECTOR] TensorRT engine saved: {engine_path}")
            if callback:
                callback(engine_path)
            return engine_path
        except Exception as e:
            print(f"[DETECTOR] TensorRT export failed: {e}")
            if callback:
                callback(None)
            return None

    @property
    def engine_type(self) -> str:
        """Return the current inference backend name."""
        if self.model_path.endswith(".engine"):
            return "TensorRT FP16"
        return f"PyTorch {'FP16' if cfg.USE_HALF else 'FP32'}"
