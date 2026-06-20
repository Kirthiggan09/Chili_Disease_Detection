#!/usr/bin/env python3
"""One-shot training script for 4-class 640px chili disease model."""
import multiprocessing
from pathlib import Path


def main():
    from ultralytics import YOLO

    ROOT = Path(__file__).resolve().parent

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(ROOT / "datasets" / "data.yaml"),
        imgsz=640,
        epochs=100,
        batch=8,
        patience=20,
        workers=2,
        device=0,
        project=str(ROOT / "runs" / "train"),
        name="chili_disease_v2",
        exist_ok=True,
        verbose=True,
        # Loss tuning for small-object class imbalance
        box=8.5,
        cls=2.0,
        # Augmentation
        hsv_h=0.02,
        hsv_s=0.70,
        hsv_v=0.40,
        degrees=15.0,
        translate=0.15,
        scale=0.50,
        shear=5.0,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print("=" * 60)
    print(f"  Training complete!")
    print(f"  Save dir:     {results.save_dir}")
    print(f"  Best weights: {best_pt}")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
