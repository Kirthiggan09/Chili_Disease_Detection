"""
ChiliRover AI — Deployment Utilities
======================================
Helper functions for the onboard inference script:
  • Bounding-box drawing with class-specific colours
  • Smoothed FPS counter (rolling average)
  • YAML config loader

Classes (4-class optimized):
  0: Cercospora_Leaf_Spot
  1: Healthy
  2: Chlorosis
  3: Powdery_Mildew
"""

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

# ── Class colour palette (visually distinct, high contrast on foliage) ──
CLASS_COLOURS = [
    (55, 100, 235),   # red    — Cercospora_Leaf_Spot
    (72, 209, 55),    # green  — Healthy
    (55, 235, 235),   # yellow — Chlorosis
    (200, 80, 200),   # purple — Powdery_Mildew
    (235, 150, 55),   # blue   — extra class slot
    (100, 235, 200),  # teal   — extra class slot
    (180, 55, 235),   # magenta
]


def load_config(config_path: str | Path) -> dict:
    """Load pipeline_config.yaml and return as dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


# =====================================================================
#  FPS Counter
# =====================================================================

class FPSCounter:
    """Rolling-average FPS counter for smooth display."""

    def __init__(self, window: int = 30):
        self._times: deque[float] = deque(maxlen=window)
        self._last = time.perf_counter()

    def tick(self) -> float:
        """Call once per frame. Returns the smoothed FPS."""
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        avg = sum(self._times) / len(self._times)
        return 1.0 / avg if avg > 0 else 0.0


# =====================================================================
#  Drawing
# =====================================================================

def get_class_colour(class_id: int) -> tuple[int, int, int]:
    """Return a BGR colour tuple for the given class index."""
    return CLASS_COLOURS[class_id % len(CLASS_COLOURS)]


def draw_detections(
    frame: np.ndarray,
    boxes,
    names: dict[int, str],
    fps: float | None = None,
) -> np.ndarray:
    """
    Draw bounding boxes, class labels, and optional FPS overlay.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (will be modified in-place).
    boxes : ultralytics.engine.results.Boxes
        Detection boxes from YOLO result.
    names : dict
        Class-index → class-name mapping.
    fps : float, optional
        Current FPS to overlay in the top-left corner.

    Returns
    -------
    np.ndarray
        The annotated frame.
    """
    for box in boxes:
        # Extract box coordinates, confidence, class
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = names.get(cls_id, f"cls_{cls_id}")

        colour = get_class_colour(cls_id)
        label = f"{cls_name} {conf:.0%}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(
            frame, label, (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # FPS overlay
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, fps_text, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2, cv2.LINE_AA,
        )

    return frame
