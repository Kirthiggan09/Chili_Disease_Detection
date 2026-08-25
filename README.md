# 🌶️ ChiliRover AI

Real-time chili plant disease detection for a Raspberry Pi 5 rover with multi-modal sensor telemetry.

**Pipeline:** Roboflow Dataset → YOLOv8m Training (800×800) → Hailo INT8 Export (.hef) → Pi 5 (AI HAT+) Live Inference + Sensor HUD

---

## Disease Classes (4-class optimized)

| ID | Class | Description |
|---|---|---|
| 0 | `Cercospora_Leaf_Spot` | Fungal spots with light gray centers and dark margins |
| 1 | `Healthy` | Normal, disease-free leaves |
| 2 | `Chlorosis` | Yellowing / nutrient deficiency symptoms |
| 3 | `Powdery_Mildew` | White/gray powdery patches on leaf surfaces |

---

## Quick Start

### 1. Training (Colab or Local GPU)

```bash
# Install dependencies
pip install -r requirements/train_requirements.txt

# Set your Roboflow API key
export ROBOFLOW_API_KEY="your_key_here"

# Edit config/pipeline_config.yaml with your Roboflow workspace/project details

# Run the full pipeline (download → train → export)
cd training
python train.py

# Or override via CLI
python train.py --epochs 50 --imgsz 800 --batch 8
```

### 2. Deploy to Raspberry Pi 5 (with AI HAT+)

```bash
# On the Pi — install dependencies
pip install -r requirements/pi_requirements.txt
# Ensure Hailo RT is installed on your Raspberry Pi!

# Copy the exported model to the Pi
# scp weights/hailo_model/best_hailo_model.hef pi@raspberrypi:~/chili-rover-ai/weights/hailo_model/

# Run live inference (vision only)
cd deployment
python pi_inference.py
python pi_inference.py --model ../weights/hailo_model/best_hailo_model.hef --camera 0 --confidence 0.45
python pi_inference.py --no-display --save   # Headless mode with output saving
```

### 3. Combined Inference + Sensor Telemetry

```bash
# Run with MCU sensor data (DHT11 + MQ3)
cd scripts
python pi_combined_inference.py --serial-port /dev/ttyUSB0

# Custom options
python pi_combined_inference.py --serial-port COM3 --baud 115200
python pi_combined_inference.py --no-display --save   # Headless mode
```

**MCU JSON format expected on serial:**
```json
{"temp": 28.5, "humidity": 65.2, "ethylene_ppm": 12.4, "voc_ppm": 45.7}
```

---

## Project Structure

```
chili-rover-ai/
├── config/
│   └── pipeline_config.yaml       # All hyperparams, paths, and telemetry config
├── training/
│   ├── train.py                    # Train + export pipeline to Hailo
│   └── augmentation.py             # Augmentation presets
├── deployment/
│   ├── pi_inference.py             # Onboard real-time inference (vision only)
│   ├── pi_stream.py                # MJPEG video streaming server
│   └── utils.py                    # Drawing, FPS counter, helpers
├── scripts/
│   └── pi_combined_inference.py    # Vision + sensor telemetry HUD
├── datasets/                       # Roboflow dataset (4 classes)
├── weights/                        # Exported Hailo models (.hef)
├── requirements/
│   ├── train_requirements.txt
│   └── pi_requirements.txt
└── README.md
```

---

## Configuration

All settings live in `config/pipeline_config.yaml`:

| Section | Key Settings |
|---|---|
| `dataset` | Roboflow workspace, project, version, API key env var |
| `training` | Model (yolov8m), epochs, batch size, imgsz=800 |
| `augmentation` | HSV jitter, rotation, scale, mosaic, mixup |
| `export` | Format (hef), INT8 flag, imgsz=800, output directory |
| `deployment` | Model path, camera index, confidence, warm-up runs |
| `telemetry` | Serial port, baud rate, timeout for MCU sensor data |

---

## Sensor Integration

The rover reads data from two sensors via an MCU (ESP32/Arduino) over serial:

| Sensor | Measurements | Purpose |
|---|---|---|
| **DHT11** | Temperature (°C), Humidity (%) | Environmental monitoring for disease risk |
| **MQ3** | Ethylene (PPM), VOC (PPM) | Plant stress / fruit ripening indicators |

These values are overlaid as a live HUD panel on the camera feed.

---

## Performance Expectations

| Metric | Raspberry Pi 5 + AI HAT+ (Hailo-8) |
|---|---|
| FPS | 30+ (at 800×800) |
| Latency | <30 ms/frame |
| Model size (INT8) | ~25 MB |
| RAM usage | ~400 MB |

---

## Roboflow Setup

1. Create a free account at [roboflow.com](https://app.roboflow.com)
2. Find a chili/capsicum disease dataset on [Roboflow Universe](https://universe.roboflow.com)
3. Copy your **API key** from Settings → API Keys
4. Update `pipeline_config.yaml` with your workspace slug, project slug, and version number

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Camera not detected | Run `v4l2-ctl --list-devices` or `libcamera-hello` |
| Import errors on Pi | Ensure `pip install ultralytics` and Hailo dependencies are met |
| ROBOFLOW_API_KEY missing | `export ROBOFLOW_API_KEY="your_key"` before running |
| Serial port not found | Check `ls /dev/ttyUSB*` or `ls /dev/ttyACM*` |
| No sensor data | Verify MCU is sending JSON at the configured baud rate |

---

## License

MIT
