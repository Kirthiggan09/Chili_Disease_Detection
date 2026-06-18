#!/usr/bin/env python3
"""
ChiliRover AI — Training & Export Script
==========================================
End-to-end pipeline:
  1. Downloads a chili/capsicum disease dataset from Roboflow
  2. Trains YOLOv8n with aggressive spatial + colour augmentations
  3. Exports the best checkpoint to OpenVINO FP16 @ 320×320

Usage (Colab / local):
    python train.py
    python train.py --epochs 50 --imgsz 320 --api-key YOUR_KEY
    python train.py --config ../config/pipeline_config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Resolve project root so imports work regardless of cwd
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from augmentation import get_augmentation_kwargs


# ===================================================================
#  1. Configuration
# ===================================================================

def load_config(config_path: str | Path) -> dict:
    """Load YAML config file and return as dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"[WARN] Config not found at {config_path}, using defaults.")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Override config values with any CLI arguments that were explicitly set."""
    train_cfg = config.setdefault("training", {})
    export_cfg = config.setdefault("export", {})
    ds_cfg = config.setdefault("dataset", {})

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.imgsz is not None:
        train_cfg["imgsz"] = args.imgsz
        export_cfg["imgsz"] = args.imgsz
    if args.batch is not None:
        train_cfg["batch"] = args.batch
    if args.api_key is not None:
        ds_cfg["_api_key_override"] = args.api_key
    if args.model is not None:
        train_cfg["model"] = args.model

    return config


# ===================================================================
#  2. Dataset Download
# ===================================================================

def download_dataset(config: dict) -> str:
    """
    Download dataset from Roboflow and return the path to data.yaml.

    Falls back to a well-known public chili disease dataset if no
    workspace/project is configured.
    """
    ds_cfg = config.get("dataset", {})

    # Resolve API key: CLI override > env var > None
    api_key = ds_cfg.get("_api_key_override") or os.getenv(
        ds_cfg.get("api_key_env", "ROBOFLOW_API_KEY")
    )

    if not api_key:
        print(
            "\n╔══════════════════════════════════════════════════════════╗\n"
            "║  ROBOFLOW API KEY NOT FOUND                             ║\n"
            "║                                                         ║\n"
            "║  Set the env var before running:                        ║\n"
            "║    export ROBOFLOW_API_KEY='your_key_here'              ║\n"
            "║                                                         ║\n"
            "║  Or pass via CLI:                                       ║\n"
            "║    python train.py --api-key YOUR_KEY                   ║\n"
            "║                                                         ║\n"
            "║  Get a free key at https://app.roboflow.com/settings    ║\n"
            "╚══════════════════════════════════════════════════════════╝\n"
        )
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow package not installed. Run: pip install roboflow")
        sys.exit(1)

    workspace = ds_cfg.get("workspace", "YOUR_WORKSPACE")
    project_name = ds_cfg.get("project", "YOUR_PROJECT")
    version = ds_cfg.get("version", 1)
    fmt = ds_cfg.get("format", "yolov8")

    print(f"\n[INFO] Connecting to Roboflow workspace: {workspace}")
    print(f"[INFO] Project: {project_name}, Version: {version}")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download(fmt, location=str(PROJECT_ROOT / "datasets"))

    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        print(f"[ERROR] data.yaml not found at {data_yaml}")
        sys.exit(1)

    print(f"[OK] Dataset ready at: {dataset.location}")
    return str(data_yaml)


# ===================================================================
#  3. Training
# ===================================================================

def train_model(config: dict, data_yaml: str) -> str:
    """
    Train YOLOv8n and return the path to the best checkpoint.
    """
    from ultralytics import YOLO

    train_cfg = config.get("training", {})
    aug_kwargs = get_augmentation_kwargs(config)

    model_name = train_cfg.get("model", "yolov8n.pt")
    imgsz = train_cfg.get("imgsz", 320)
    epochs = train_cfg.get("epochs", 100)
    batch = train_cfg.get("batch", 16)
    patience = train_cfg.get("patience", 20)
    workers = train_cfg.get("workers", 4)
    device = train_cfg.get("device", "")
    project = train_cfg.get("project", "runs/train")
    name = train_cfg.get("name", "chili_disease")

    print(f"\n{'='*60}")
    print(f"  TRAINING CONFIG")
    print(f"  Model:      {model_name}")
    print(f"  Resolution: {imgsz}×{imgsz}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch:      {batch}")
    print(f"  Device:     {device or 'auto'}")
    print(f"{'='*60}\n")

    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=patience,
        workers=workers,
        device=device if device else None,
        project=str(PROJECT_ROOT / project),
        name=name,
        exist_ok=True,
        verbose=True,
        # Unpack augmentation kwargs
        **aug_kwargs,
    )

    # Locate best weights
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = Path(results.save_dir) / "weights" / "last.pt"

    print(f"\n[OK] Training complete. Best weights: {best_pt}")
    return str(best_pt)


# ===================================================================
#  4. OpenVINO Export
# ===================================================================

def export_to_openvino(config: dict, weights_path: str) -> str:
    """
    Export trained YOLO weights to OpenVINO IR format (FP16).

    Returns the path to the exported model directory.
    """
    from ultralytics import YOLO

    export_cfg = config.get("export", {})
    imgsz = export_cfg.get("imgsz", 320)
    half = export_cfg.get("half", True)
    output_dir = PROJECT_ROOT / export_cfg.get("output_dir", "models")

    print(f"\n{'='*60}")
    print(f"  EXPORTING TO OPENVINO")
    print(f"  Weights:    {weights_path}")
    print(f"  Format:     OpenVINO IR ({'FP16' if half else 'FP32'})")
    print(f"  Resolution: {imgsz}×{imgsz}")
    print(f"{'='*60}\n")

    model = YOLO(weights_path)
    export_path = model.export(format="openvino", imgsz=imgsz, half=half)

    print(f"\n[OK] OpenVINO model exported to: {export_path}")
    print(f"[INFO] Copy this directory to your Raspberry Pi 5 for deployment.")

    return str(export_path)


# ===================================================================
#  5. CLI Entry Point
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChiliRover AI — Train & Export Pipeline"
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "config" / "pipeline_config.yaml"),
        help="Path to pipeline_config.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="Base model name (e.g. yolov8n.pt, yolo11n.pt)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Roboflow API key (overrides env var)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, only export existing weights")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to existing .pt weights (use with --skip-train)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    print("\n[*] ChiliRover AI -- Training Pipeline\n")

    if args.skip_train:
        # Export-only mode
        if not args.weights:
            print("[ERROR] --skip-train requires --weights path")
            sys.exit(1)
        weights_path = args.weights
    else:
        # Full pipeline: download → train → export
        data_yaml = download_dataset(config)
        weights_path = train_model(config, data_yaml)

    export_to_openvino(config, weights_path)

    print("\n[OK] Pipeline complete!\n")


if __name__ == "__main__":
    main()
