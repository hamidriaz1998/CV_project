import os
import sys
from pathlib import Path

from ultralytics import YOLO


def train():
    dataset_yaml = str(Path(__file__).parent / "dataset.yaml")
    project_dir = str(Path(__file__).parent / "results")

    print("Loading YOLOv8n pre-trained model...")
    model = YOLO("yolov8n.pt")

    print("Starting training...")
    print("  Model: YOLOv8n (3.2M params)")
    print("  Dataset: chest X-ray (50 annotated images)")
    print(f"  Device: {'cuda' if os.path.exists('/usr/local/cuda') or os.path.exists('/dev/nvidia0') else 'cpu'}")
    print("  Note: CPU training may take 10–20 minutes")

    results = model.train(
        data=dataset_yaml,
        epochs=100,
        patience=20,
        batch=8,
        imgsz=224,
        project=project_dir,
        name="yolov8n_chest_xray",
        exist_ok=True,
        pretrained=True,
        optimizer="Adam",
        lr0=0.001,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.1,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.1,
        verbose=True,
    )

    print("\nTraining complete!")
    print(f"Best model saved to: {project_dir}/yolov8n_chest_xray/weights/best.pt")
    print(f"Results saved to: {project_dir}/yolov8n_chest_xray/")


if __name__ == "__main__":
    train()
