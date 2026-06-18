#!/usr/bin/env python3
"""
ChiliRover AI — Raspberry Pi 5 Live Video Streamer
======================================================
Production-ready HTTP video streamer.
Runs the OpenVINO AI model and serves a low-latency MJPEG stream 
so your friend's web dashboard can embed the live AI camera feed.

Usage:
    python pi_stream.py
    python pi_stream.py --port 5000 --camera 0
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
from flask import Flask, Response, render_template_string

# Resolve project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import FPSCounter, draw_detections, load_config

# Import camera and model initialisation from our existing script
try:
    from pi_inference import init_camera, init_model
except ImportError:
    print("[ERROR] Could not import pi_inference.py")
    sys.exit(1)

app = Flask(__name__)

# Global variables to hold model and camera
global_model = None
global_cap = None
global_config = {}

# Simple fallback HTML to test the stream directly in a browser
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>ChiliRover AI Live Stream</title>
    <style>
        body { background-color: #121212; color: #fff; text-align: center; font-family: sans-serif; }
        h1 { margin-top: 20px; }
        img { border: 3px solid #333; border-radius: 8px; max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <h1>🌶️ ChiliRover AI Feed</h1>
    <img src="/video_feed" alt="Live AI Video Stream">
</body>
</html>
"""

@app.route('/')
def index():
    """Return a simple webpage to test the video stream directly."""
    return render_template_string(INDEX_HTML)


def generate_frames():
    """
    Generator function that grabs frames from the camera,
    runs YOLO inference, draws bounding boxes, and yields JPEG bytes.
    """
    global global_model, global_cap, global_config

    deploy_cfg = global_config.get("deployment", {})
    imgsz = deploy_cfg.get("imgsz", 320)
    confidence = deploy_cfg.get("confidence", 0.40)
    iou_threshold = deploy_cfg.get("iou_threshold", 0.45)
    names = global_model.names

    fps_counter = FPSCounter(window=30)

    while True:
        success, frame = global_cap.read()
        if not success:
            time.sleep(0.1)
            continue

        # Resize to match our trained 320x320 resolution
        frame_resized = cv2.resize(frame, (imgsz, imgsz))

        # Run OpenVINO inference
        results = global_model.predict(
            frame_resized,
            imgsz=imgsz,
            conf=confidence,
            iou=iou_threshold,
            verbose=False,
        )

        # Update FPS
        fps = fps_counter.tick()

        # Draw detections
        annotated_frame = draw_detections(
            frame_resized, results[0].boxes, names, fps=fps
        )

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # Yield the frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """
    HTTP route returning a multipart/x-mixed-replace stream.
    Your friend's dashboard will use <img src="http://<pi-ip>:5000/video_feed">
    """
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def parse_args():
    parser = argparse.ArgumentParser(description="ChiliRover AI Flask Streamer")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config" / "pipeline_config.yaml"))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind to")
    parser.add_argument("--port", type=int, default=5000, help="HTTP Port")
    return parser.parse_args()


def main():
    global global_model, global_cap, global_config

    args = parse_args()
    global_config = load_config(args.config)
    deploy = global_config.get("deployment", {})

    model_path = args.model or deploy.get("model_path", "models/best_openvino_model")
    camera_idx = args.camera if args.camera is not None else deploy.get("camera_index", 0)
    warmup = deploy.get("warmup_runs", 3)
    imgsz = deploy.get("imgsz", 320)

    print("\n🌶️  ChiliRover AI — Starting Video Streamer\n")

    # Initialize hardware and models
    global_model = init_model(model_path, warmup_runs=warmup, imgsz=imgsz)
    global_cap = init_camera(camera_idx)

    # Start Flask server
    print(f"\n[INFO] Starting stream server on http://{args.host}:{args.port}")
    print("[INFO] Dashboard integration tag:")
    print(f'       <img src="http://<RASPBERRY_PI_IP>:{args.port}/video_feed">')
    print("=" * 60)
    
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
    finally:
        if global_cap:
            global_cap.release()


if __name__ == "__main__":
    main()
