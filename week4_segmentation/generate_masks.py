import json
import os
import shutil
import random
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COCO_PATH = PROJECT_ROOT / "annotations_coco.json"
IMG_SOURCE_DIR = PROJECT_ROOT / "chest_xray" / "train"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)


def load_coco(path):
    with open(path) as f:
        return json.load(f)


def make_dirs():
    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "masks" / split).mkdir(parents=True, exist_ok=True)


def merge_boxes_into_mask(boxes, img_w, img_h):
    """Merge all bounding boxes into a single binary mask."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def main():
    make_dirs()

    coco = load_coco(COCO_PATH)
    image_info = {img["id"]: img for img in coco["images"]}
    annotations_by_image = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    annotated_ids = list(annotations_by_image.keys())
    random.shuffle(annotated_ids)
    split_idx = int(len(annotated_ids) * TRAIN_RATIO)
    train_ids = set(annotated_ids[:split_idx])
    val_ids = set(annotated_ids[split_idx:])

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
        for img_id in split_ids:
            img = image_info[img_id]
            fname = img["file_name"]
            img_w, img_h = img["width"], img["height"]

            src_path = IMG_SOURCE_DIR / fname
            if not src_path.exists():
                print(f"WARNING: {src_path} not found, skipping")
                continue

            dst_name = Path(fname).name
            dst_img = OUTPUT_DIR / "images" / split_name / dst_name
            shutil.copy2(src_path, dst_img)

            boxes = [
                (int(ann["bbox"][0]),
                 int(ann["bbox"][1]),
                 int(ann["bbox"][0] + ann["bbox"][2]),
                 int(ann["bbox"][1] + ann["bbox"][3]))
                for ann in annotations_by_image[img_id]
            ]
            mask = merge_boxes_into_mask(boxes, img_w, img_h)
            mask_path = OUTPUT_DIR / "masks" / split_name / (Path(fname).stem + ".png")
            Image.fromarray(mask).save(mask_path)

    print(f"\nDone. Rectangular masks saved to: {OUTPUT_DIR}")
    print("\nNOTE: These are simple box-derived masks (rectangles).")
    print("For precise organically-shaped masks, run SAM in the Colab notebook (Step 1b).")


if __name__ == "__main__":
    main()
