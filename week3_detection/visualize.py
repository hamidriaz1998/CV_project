import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

RESULTS_DIR = Path(__file__).parent / "results" / "yolov8n_chest_xray"
MODEL_PATH = RESULTS_DIR / "weights" / "best.pt"
VAL_IMG_DIR = Path(__file__).parent / "data" / "images" / "val"
VAL_LBL_DIR = Path(__file__).parent / "data" / "labels" / "val"
OUTPUT_DIR = Path(__file__).parent / "predictions"

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
}
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def read_yolo_label(txt_path, img_w, img_h):
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def draw_boxes(img, boxes, color_map, label_names, prefix=""):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
    for cls_id, x1, y1, x2, y2 in boxes:
        color = color_map.get(cls_id, (255, 255, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{prefix}{label_names[cls_id]}"
        bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1), label, fill=(0, 0, 0), font=font)
    return img


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(str(MODEL_PATH))

    img_paths = sorted(VAL_IMG_DIR.iterdir())
    if not img_paths:
        print("No validation images found.")
        return

    print(f"Running inference on {len(img_paths)} validation images...")
    results = model(list(img_paths), imgsz=224, conf=0.25, iou=0.5)

    for img_path, result in zip(img_paths, results):
        pil_img = Image.open(img_path).convert("RGB")
        img_w, img_h = pil_img.size

        gt_boxes = read_yolo_label(
            VAL_LBL_DIR / (img_path.stem + ".txt"),
            img_w,
            img_h,
        )

        pred_boxes = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                pred_boxes.append((cls_id, x1, y1, x2, y2))

        gt_img = pil_img.copy()
        gt_img = draw_boxes(gt_img, gt_boxes, CLASS_COLORS, CLASS_NAMES, prefix="GT: ")

        pred_img = pil_img.copy()
        pred_img = draw_boxes(pred_img, pred_boxes, CLASS_COLORS, CLASS_NAMES, prefix="")

        composite = Image.new("RGB", (img_w * 2, img_h))
        composite.paste(gt_img, (0, 0))
        composite.paste(pred_img, (img_w, 0))

        out_path = OUTPUT_DIR / f"pred_{img_path.stem}.png"
        composite.save(out_path)
        print(f"  Saved: {out_path.name}")

    print(f"\nDone! {len(img_paths)} prediction images saved to {OUTPUT_DIR}")
    print("Left: Ground Truth | Right: Model Prediction")


if __name__ == "__main__":
    main()
