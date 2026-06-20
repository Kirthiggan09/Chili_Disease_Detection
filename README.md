# 🌶️ ChiliRover AI

Real-time chili plant disease detection for a Raspberry Pi 5 rover with multi-modal sensor telemetry.

**Pipeline:** Roboflow Dataset → YOLOv8n Training (640×640) → OpenVINO FP16 Export → Pi 5 Live Inference + Sensor HUD

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
python train.py --epochs 50 --imgsz 640 --batch 16
```

### 2. Deploy to Raspberry Pi 5

```bash
# On the Pi — install dependencies
pip install -r requirements/pi_requirements.txt

# Copy the exported model directory to the Pi
# scp -r models/best_openvino_model/ pi@raspberrypi:~/chili-rover-ai/models/

# Run live inference (vision only)
cd deployment
python pi_inference.py
python pi_inference.py --model ../models/best_openvino_model --camera 0 --confidence 0.45
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
│   ├── train.py                    # Train + export pipeline (box=8.5, cls=2.0)
│   └── augmentation.py             # Augmentation presets
├── deployment/
│   ├── pi_inference.py             # Onboard real-time inference (vision only)
│   ├── pi_stream.py                # MJPEG video streaming server
│   └── utils.py                    # Drawing, FPS counter, helpers
├── scripts/
│   └── pi_combined_inference.py    # Vision + sensor telemetry HUD
├── datasets/                       # Roboflow dataset (4 classes)
├── models/                         # Exported OpenVINO weights
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
| `training` | Model, epochs, batch size, imgsz=640, box=8.5, cls=2.0 |
| `augmentation` | HSV jitter, rotation, scale, mosaic, mixup |
| `export` | Format (OpenVINO), FP16 flag, imgsz=640, output directory |
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

| Metric | Raspberry Pi 5 |
|---|---|
| FPS | 5–10 (at 640×640) |
| Latency | 100–200 ms/frame |
| Model size (FP16) | ~6 MB |
| RAM usage | ~350 MB |

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
| Low FPS | Reduce `imgsz` to 320 in config (trades accuracy for speed) |
| Import errors on Pi | Ensure `pip install ultralytics openvino` completed |
| ROBOFLOW_API_KEY missing | `export ROBOFLOW_API_KEY="your_key"` before running |
| Serial port not found | Check `ls /dev/ttyUSB*` or `ls /dev/ttyACM*` |
| No sensor data | Verify MCU is sending JSON at the configured baud rate |

---

## License

MIT
