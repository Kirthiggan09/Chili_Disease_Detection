# 🌶️ ChiliRover AI

Real-time chili plant disease detection for a Raspberry Pi 5 rover.

**Pipeline:** Roboflow Dataset → YOLOv8n Training → OpenVINO FP16 Export → Pi 5 Live Inference

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
python train.py --epochs 50 --imgsz 320 --batch 16
```

### 2. Deploy to Raspberry Pi 5

```bash
# On the Pi — install dependencies
pip install -r requirements/pi_requirements.txt

# Copy the exported model directory to the Pi
# scp -r models/best_openvino_model/ pi@raspberrypi:~/chili-rover-ai/models/

# Run live inference
cd deployment
python pi_inference.py

# Custom options
python pi_inference.py --model ../models/best_openvino_model --camera 0 --confidence 0.45
python pi_inference.py --no-display --save   # Headless mode with output saving
```

---

## Project Structure

```
chili-rover-ai/
├── config/
│   └── pipeline_config.yaml       # All hyperparams and paths
├── training/
│   ├── train.py                    # Train + export pipeline
│   └── augmentation.py             # Augmentation presets
├── deployment/
│   ├── pi_inference.py             # Onboard real-time inference
│   └── utils.py                    # Drawing, FPS counter, helpers
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
| `training` | Model, epochs, batch size, image size, patience |
| `augmentation` | HSV jitter, rotation, scale, mosaic, mixup |
| `export` | Format (OpenVINO), FP16 flag, output directory |
| `deployment` | Model path, camera index, confidence, warm-up runs |

---

## Roboflow Setup

1. Create a free account at [roboflow.com](https://app.roboflow.com)
2. Find a chili/capsicum disease dataset on [Roboflow Universe](https://universe.roboflow.com)
3. Copy your **API key** from Settings → API Keys
4. Update `pipeline_config.yaml` with your workspace slug, project slug, and version number

---

## Performance Expectations

| Metric | Raspberry Pi 5 |
|---|---|
| FPS | 8–15 |
| Latency | 70–130 ms/frame |
| Model size (FP16) | ~3.5 MB |
| RAM usage | ~250 MB |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Camera not detected | Run `v4l2-ctl --list-devices` or `libcamera-hello` |
| Low FPS | Reduce `imgsz` to 256 in config |
| Import errors on Pi | Ensure `pip install ultralytics openvino` completed |
| ROBOFLOW_API_KEY missing | `export ROBOFLOW_API_KEY="your_key"` before running |

---

## License

MIT
