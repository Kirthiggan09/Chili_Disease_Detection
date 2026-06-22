#!/usr/bin/env python3
"""
ChiliRover AI — Combined Inference with Multi-Modal Sensor Telemetry
=====================================================================
Production-ready real-time chili disease detection with live sensor HUD.

• Loads an OpenVINO-optimized YOLOv8n model (640×640, 4-class)
• Captures frames from Pi Camera / USB camera via OpenCV
• Reads serial JSON telemetry from MCU (DHT11 + MQ3 sensors)
• Overlays a live sensor HUD alongside YOLO bounding boxes
• Supports both display and headless (MJPEG stream-ready) modes

MCU JSON format expected on serial:
    {"temp": 28.5, "humidity": 65.2, "ethylene_ppm": 12.4, "voc_ppm": 45.7}

Wiring:
    DHT11       → MCU digital pin   → temp / humidity
    MQ3 (Vout)  → MCU analog pin    → ethylene / VOC in PPM

Usage:
    python pi_combined_inference.py
    python pi_combined_inference.py --serial-port /dev/ttyUSB0
    python pi_combined_inference.py --serial-port COM3 --baud 115200
    python pi_combined_inference.py --no-display --save
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# Resolve project root for imports and default config
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "deployment"))

from utils import FPSCounter, draw_detections, load_config


# ===================================================================
#  Serial Telemetry Reader (threaded)
# ===================================================================

class TelemetryReader:
    """
    Non-blocking serial reader for MCU sensor data.

    Reads JSON strings from the microcontroller in a background thread.
    Thread-safe access to the latest sensor readings.
    """

    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.1):
        self.port = port
        self.baud = baud
        self.timeout = timeout

        self._lock = threading.Lock()
        self._latest: dict = {
            "temp": None,
            "humidity": None,
            "ethylene_ppm": None,
            "voc_ppm": None,
        }
        self._connected = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._serial = None

    def start(self):
        """Open serial port and start background reader thread."""
        try:
            import serial
        except ImportError:
            print("[ERROR] pyserial not installed. Run: pip install pyserial")
            print("[WARN] Continuing without sensor telemetry...")
            return

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout,
            )
            self._connected = True
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            print(f"[OK] Telemetry connected: {self.port} @ {self.baud} baud")
        except Exception as e:
            print(f"[WARN] Cannot open serial port {self.port}: {e}")
            print("[WARN] Continuing without sensor telemetry...")
            self._connected = False

    def _read_loop(self):
        """Background thread: continuously read and parse serial JSON."""
        while self._running and self._serial and self._serial.is_open:
            try:
                line = self._serial.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                data = json.loads(line)
                with self._lock:
                    if "temp" in data:
                        self._latest["temp"] = float(data["temp"])
                    if "humidity" in data:
                        self._latest["humidity"] = float(data["humidity"])
                    if "ethylene_ppm" in data:
                        self._latest["ethylene_ppm"] = float(data["ethylene_ppm"])
                    if "voc_ppm" in data:
                        self._latest["voc_ppm"] = float(data["voc_ppm"])

            except json.JSONDecodeError:
                pass  # Skip malformed lines
            except Exception:
                time.sleep(0.05)

    @property
    def latest(self) -> dict:
        """Return the latest sensor readings (thread-safe copy)."""
        with self._lock:
            return dict(self._latest)

    @property
    def connected(self) -> bool:
        return self._connected

    def stop(self):
        """Stop the reader thread and close the serial port."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
        print("[OK] Telemetry reader stopped.")


# ===================================================================
#  Sensor HUD Overlay
# ===================================================================

def draw_sensor_hud(
    frame: np.ndarray,
    telemetry: dict,
    connected: bool,
) -> np.ndarray:
    """
    Overlay a semi-transparent sensor HUD panel on the frame.

    Displays DHT11 temperature/humidity and MQ3 ethylene/VOC readings
    in the bottom-left corner of the frame.
    """
    h, w = frame.shape[:2]

    # HUD dimensions
    hud_w, hud_h = 280, 130
    hud_x, hud_y = 8, h - hud_h - 8

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (hud_x, hud_y),
        (hud_x + hud_w, hud_y + hud_h),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Border
    cv2.rectangle(
        frame,
        (hud_x, hud_y),
        (hud_x + hud_w, hud_y + hud_h),
        (0, 200, 200),
        1,
    )

    # Title
    title = "SENSOR TELEMETRY" if connected else "SENSORS OFFLINE"
    title_colour = (0, 255, 200) if connected else (0, 0, 200)
    cv2.putText(
        frame, title,
        (hud_x + 10, hud_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, title_colour, 1, cv2.LINE_AA,
    )

    # Horizontal divider
    cv2.line(
        frame,
        (hud_x + 8, hud_y + 28),
        (hud_x + hud_w - 8, hud_y + 28),
        (80, 80, 80), 1,
    )

    # Sensor values
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    text_colour = (220, 220, 220)
    y_offset = hud_y + 48

    temp = telemetry.get("temp")
    humidity = telemetry.get("humidity")
    ethylene = telemetry.get("ethylene_ppm")
    voc = telemetry.get("voc_ppm")

    lines = [
        (f"Temp:     {temp:.1f} C" if temp is not None else "Temp:     --", (100, 200, 255)),
        (f"Humidity: {humidity:.1f} %" if humidity is not None else "Humidity: --", (100, 255, 200)),
        (f"Ethylene: {ethylene:.1f} PPM" if ethylene is not None else "Ethylene: --", (200, 200, 100)),
        (f"VOC:      {voc:.1f} PPM" if voc is not None else "VOC:      --", (200, 150, 255)),
    ]

    for text, colour in lines:
        cv2.putText(
            frame, text,
            (hud_x + 14, y_offset),
            font, font_scale, colour, 1, cv2.LINE_AA,
        )
        y_offset += 22

    return frame


# ===================================================================
#  Camera Initialisation (reused from deployment)
# ===================================================================

def init_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """Open camera with fallback strategies for Raspberry Pi compatibility."""
    # Attempt 1: V4L2 backend
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if cap.isOpened():
        print(f"[OK] Camera {camera_index} opened via V4L2")
        return cap

    # Attempt 2: Default backend
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"[OK] Camera {camera_index} opened via default backend")
        return cap

    # Attempt 3: /dev/video path
    cap = cv2.VideoCapture(f"/dev/video{camera_index}")
    if cap.isOpened():
        print(f"[OK] Camera opened via /dev/video{camera_index}")
        return cap

    print(f"[ERROR] Cannot open camera index {camera_index}")
    sys.exit(1)


# ===================================================================
#  Model Initialisation
# ===================================================================

def init_model(model_path: str, warmup_runs: int = 3, imgsz: int = 640):
    """Load the OpenVINO YOLO model and run warm-up inferences."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"[ERROR] Model not found at: {model_path}")
        sys.exit(1)

    print(f"[INFO] Loading model from: {model_path}")
    model = YOLO(str(model_path), task="detect")

    # Warm-up: prime OpenVINO execution kernels
    print(f"[INFO] Running {warmup_runs} warm-up inferences...")
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for i in range(warmup_runs):
        model.predict(dummy, imgsz=imgsz, verbose=False)
        print(f"  Warm-up {i+1}/{warmup_runs} done")

    print("[OK] Model ready.\n")
    return model


def filter_detections(boxes, frame_height: int):
    """
    Post-processing filter to suppress false positives.

    Filters applied (in order):
      1. Confidence threshold:  drop if conf < 0.65
      2. Area bounds:           drop if box < 30x30 or > 500x500 px
      3. ROI mask:              drop if box center falls in top 30% of Y-axis

    Parameters
    ----------
    boxes : ultralytics.engine.results.Boxes
        Raw detection boxes from YOLO result.
    frame_height : int
        Height of the inference frame (pixels).

    Returns
    -------
    list[int]
        Indices of boxes that passed all filters.
    """
    MIN_CONF = 0.65
    MIN_SIDE = 30
    MAX_SIDE = 500
    SKY_RATIO = 0.30

    sky_limit = int(frame_height * SKY_RATIO)
    kept = []

    for i, box in enumerate(boxes):
        conf = float(box.conf[0])
        if conf < MIN_CONF:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        bw = x2 - x1
        bh = y2 - y1

        if bw < MIN_SIDE or bh < MIN_SIDE:
            continue
        if bw > MAX_SIDE or bh > MAX_SIDE:
            continue

        cy = (y1 + y2) // 2
        if cy < sky_limit:
            continue

        kept.append(i)

    return kept


# ===================================================================
#  Combined Inference Loop
# ===================================================================

def run_combined_loop(
    model,
    cap: cv2.VideoCapture,
    telemetry_reader: TelemetryReader,
    imgsz: int = 640,
    confidence: float = 0.40,
    iou_threshold: float = 0.45,
    show_display: bool = True,
    save_output: bool = False,
    output_dir: str = "output",
):
    """
    Main inference loop with integrated sensor HUD.

    Captures frames, runs YOLO inference, reads sensor telemetry,
    overlays both detection boxes and sensor data, and optionally
    saves annotated frames.
    """
    fps_counter = FPSCounter(window=30)
    frame_count = 0
    names = model.names

    # Optional video writer
    writer = None
    if save_output:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / "combined_output.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(out_file), fourcc, 15.0, (imgsz, imgsz))
        print(f"[INFO] Saving output to: {out_file}")

    print("=" * 60)
    print("  [*] COMBINED INFERENCE + TELEMETRY — Press 'q' to quit")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame grab failed, retrying...")
                time.sleep(0.1)
                continue

            # Resize to model input resolution
            frame_resized = cv2.resize(frame, (imgsz, imgsz))

            # Run YOLO inference
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

            result = results[0]

            # ── Post-processing: filter false positives ──
            kept_idx = filter_detections(result.boxes, imgsz)
            filtered_boxes = result.boxes[kept_idx] if kept_idx else result.boxes[:0]

            # Draw only filtered detections
            annotated = draw_detections(
                frame_resized, filtered_boxes, names, fps=fps
            )

            # Overlay sensor HUD
            sensor_data = telemetry_reader.latest
            annotated = draw_sensor_hud(
                annotated,
                sensor_data,
                telemetry_reader.connected,
            )

            # Display
            if show_display:
                cv2.imshow("ChiliRover AI + Telemetry", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            # Save frame
            if writer is not None:
                writer.write(annotated)

            # Periodic console stats
            if frame_count % 60 == 0:
                n_det = len(filtered_boxes)
                t = sensor_data.get("temp")
                h = sensor_data.get("humidity")
                e = sensor_data.get("ethylene_ppm")
                t_str = f"{t:.1f}°C" if t else "--"
                h_str = f"{h:.1f}%" if h else "--"
                e_str = f"{e:.1f}ppm" if e else "--"
                print(
                    f"  Frame {frame_count:>5d} | "
                    f"FPS: {fps:5.1f} | "
                    f"Det: {n_det} | "
                    f"T: {t_str} H: {h_str} Gas: {e_str}"
                )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show_display:
            cv2.destroyAllWindows()
        telemetry_reader.stop()
        print(f"\n[OK] Stopped after {frame_count} frames.")


# ===================================================================
#  CLI Entry Point
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChiliRover AI — Combined Inference + Sensor Telemetry"
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "config" / "pipeline_config.yaml"),
        help="Path to pipeline_config.yaml",
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to OpenVINO model directory")
    parser.add_argument("--camera", type=int, default=None, help="Camera index")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument(
        "--serial-port", type=str, default=None,
        help="Serial port for MCU sensor data (e.g. /dev/ttyUSB0 or COM3)",
    )
    parser.add_argument("--baud", type=int, default=None, help="Serial baud rate")
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

    # Load config
    config = load_config(args.config)
    deploy = config.get("deployment", {})
    telem_cfg = config.get("telemetry", {})

    # Resolve settings (CLI overrides config)
    model_path = args.model or deploy.get("model_path", "models/best_openvino_model")
    camera_idx = args.camera if args.camera is not None else deploy.get("camera_index", 0)
    imgsz = args.imgsz or deploy.get("imgsz", 640)
    confidence = args.confidence or deploy.get("confidence", 0.40)
    iou_threshold = deploy.get("iou_threshold", 0.45)
    warmup = deploy.get("warmup_runs", 3)
    show = not args.no_display and deploy.get("show_display", True)
    save = args.save or deploy.get("save_output", False)
    output_dir = deploy.get("output_dir", "output")

    serial_port = args.serial_port or telem_cfg.get("serial_port", "/dev/ttyUSB0")
    baud_rate = args.baud or telem_cfg.get("baud_rate", 115200)
    serial_timeout = telem_cfg.get("timeout", 0.1)

    print("\n[*] ChiliRover AI — Combined Inference + Telemetry\n")

    # Initialise model
    model = init_model(model_path, warmup_runs=warmup, imgsz=imgsz)

    # Initialise camera
    cap = init_camera(camera_idx)

    # Initialise telemetry reader
    telemetry = TelemetryReader(
        port=serial_port,
        baud=baud_rate,
        timeout=serial_timeout,
    )
    telemetry.start()

    # Run combined loop
    run_combined_loop(
        model=model,
        cap=cap,
        telemetry_reader=telemetry,
        imgsz=imgsz,
        confidence=confidence,
        iou_threshold=iou_threshold,
        show_display=show,
        save_output=save,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
