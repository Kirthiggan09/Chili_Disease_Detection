#!/usr/bin/env python3
"""
ChiliRover AI — Raspberry Pi 5 Onboard Inference
===================================================
Production-ready real-time chili disease detection.

• Loads an OpenVINO-optimized YOLOv8n model
• Captures frames from Pi Camera / USB camera via OpenCV
• Runs inference frame-by-frame at 320×320
• Renders low-latency bounding boxes with class labels
• Displays smoothed FPS counter

Usage:
    python pi_inference.py
    python pi_inference.py --model models/best_openvino_model
    python pi_inference.py --camera 0 --confidence 0.45
    python pi_inference.py --config ../config/pipeline_config.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Resolve project root for imports and default config
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import FPSCounter, draw_detections, load_config


# ===================================================================
#  Camera Initialisation
# ===================================================================

def init_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """
    Open camera with fallback strategies for Raspberry Pi compatibility.

    Tries multiple backends: V4L2 → default → libcamera index.
    """
    # Attempt 1: V4L2 backend (most reliable on Pi OS)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if cap.isOpened():
        print(f"[OK] Camera {camera_index} opened via V4L2")
        return cap

    # Attempt 2: Default backend
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"[OK] Camera {camera_index} opened via default backend")
        return cap

    # Attempt 3: Try /dev/video0 path directly
    cap = cv2.VideoCapture(f"/dev/video{camera_index}")
    if cap.isOpened():
        print(f"[OK] Camera opened via /dev/video{camera_index}")
        return cap

    print(f"[ERROR] Cannot open camera index {camera_index}")
    print("  Troubleshooting:")
    print("    • Check camera connection:  libcamera-hello --timeout 2000")
    print("    • List devices:             v4l2-ctl --list-devices")
    print("    • Try a different index:    --camera 1")
    sys.exit(1)


# ===================================================================
#  Model Initialisation
# ===================================================================

def init_model(model_path: str, warmup_runs: int = 3, imgsz: int = 320):
    """
    Load the OpenVINO YOLO model and run warm-up inferences.

    Parameters
    ----------
    model_path : str
        Path to the OpenVINO model directory (contains .xml + .bin).
    warmup_runs : int
        Number of dummy inferences to prime OpenVINO kernels.
    imgsz : int
        Input resolution for warm-up frames.

    Returns
    -------
    YOLO model instance ready for prediction.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"[ERROR] Model not found at: {model_path}")
        print("  Run training first or provide the correct --model path.")
        sys.exit(1)

    print(f"[INFO] Loading model from: {model_path}")
    model = YOLO(str(model_path), task="detect")

    # Warm-up: prime OpenVINO execution kernels for stable latency
    print(f"[INFO] Running {warmup_runs} warm-up inferences...")
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for i in range(warmup_runs):
        model.predict(dummy, imgsz=imgsz, verbose=False)
        print(f"  Warm-up {i+1}/{warmup_runs} done")

    print("[OK] Model ready.\n")
    return model


# ===================================================================
#  Inference Loop
# ===================================================================

def run_inference_loop(
    model,
    cap: cv2.VideoCapture,
    imgsz: int = 320,
    confidence: float = 0.40,
    iou_threshold: float = 0.45,
    show_display: bool = True,
    save_output: bool = False,
    output_dir: str = "output",
):
    """
    Main real-time inference loop.

    Captures frames, runs YOLO inference, draws results,
    and optionally saves annotated frames.
    """
    fps_counter = FPSCounter(window=30)
    frame_count = 0
    names = model.names  # class-index → name mapping

    # Optional: video writer for saving output
    writer = None
    if save_output:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / "detection_output.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(out_file), fourcc, 15.0, (imgsz, imgsz))
        print(f"[INFO] Saving output to: {out_file}")

    print("=" * 50)
    print("  [*] LIVE INFERENCE -- Press 'q' to quit")
    print("=" * 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame grab failed, retrying...")
                time.sleep(0.1)
                continue

            # Resize to model input resolution
            frame_resized = cv2.resize(frame, (imgsz, imgsz))

            # Run inference (single frame, no verbose logging)
            results = model.predict(
                frame_resized,
                imgsz=imgsz,
                conf=confidence,
                iou=iou_threshold,
                verbose=False,
            )

            # Update FPS
            fps = fps_counter.tick()
            frame_count += 1

            # Draw detections onto the frame
            result = results[0]
            annotated = draw_detections(
                frame_resized, result.boxes, names, fps=fps
            )

            # Display
            if show_display:
                cv2.imshow("ChiliRover AI", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # q or ESC
                    break

            # Save frame
            if writer is not None:
                writer.write(annotated)

            # Periodic stats to console
            if frame_count % 60 == 0:
                n_detections = len(result.boxes)
                print(
                    f"  Frame {frame_count:>5d} | "
                    f"FPS: {fps:5.1f} | "
                    f"Detections: {n_detections}"
                )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show_display:
            cv2.destroyAllWindows()
        print(f"\n[OK] Stopped after {frame_count} frames.")


# ===================================================================
#  CLI Entry Point
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChiliRover AI — Raspberry Pi 5 Real-Time Inference"
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "config" / "pipeline_config.yaml"),
        help="Path to pipeline_config.yaml",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to OpenVINO model directory",
    )
    parser.add_argument("--camera", type=int, default=None, help="Camera index")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable OpenCV window (headless mode)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated frames to output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config, then override with CLI flags
    config = load_config(args.config)
    deploy = config.get("deployment", {})

    model_path = args.model or deploy.get("model_path", "models/best_openvino_model")
    camera_idx = args.camera if args.camera is not None else deploy.get("camera_index", 0)
    imgsz = args.imgsz or deploy.get("imgsz", 320)
    confidence = args.confidence or deploy.get("confidence", 0.40)
    iou_threshold = deploy.get("iou_threshold", 0.45)
    warmup = deploy.get("warmup_runs", 3)
    show = not args.no_display and deploy.get("show_display", True)
    save = args.save or deploy.get("save_output", False)
    output_dir = deploy.get("output_dir", "output")

    print("\n[*] ChiliRover AI -- Onboard Inference\n")

    # Initialise
    model = init_model(model_path, warmup_runs=warmup, imgsz=imgsz)
    cap = init_camera(camera_idx)

    # Run
    run_inference_loop(
        model=model,
        cap=cap,
        imgsz=imgsz,
        confidence=confidence,
        iou_threshold=iou_threshold,
        show_display=show,
        save_output=save,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
