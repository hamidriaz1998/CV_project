import json
import os
import shutil
import random
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COCO_PATH = PROJECT_ROOT / "annotations_coco.json"
IMG_SOURCE_DIR = PROJECT_ROOT / "chest_xray" / "train"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.2


def load_coco(path):
    with open(path) as f:
        return json.load(f)


def make_dirs():
    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def get_class_map(categories):
    return {cat["id"]: idx for idx, cat in enumerate(categories)}


def convert_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))
    return x_center, y_center, w_norm, h_norm


def get_class_name(categories):
    return [cat["name"] for cat in sorted(categories, key=lambda c: c["id"])]


def main():
    coco = load_coco(COCO_PATH)
    cat_map = get_class_map(coco["categories"])
    class_names = get_class_name(coco["categories"])
    print(f"Classes: {class_names}")
    print(f"Class map (COCO id -> YOLO idx): {cat_map}")

    image_info = {}
    for img in coco["images"]:
        image_info[img["id"]] = img

    annotations_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        annotations_by_image.setdefault(img_id, []).append(ann)

    annotated_ids = list(annotations_by_image.keys())
    random.shuffle(annotated_ids)

    split_idx = int(len(annotated_ids) * TRAIN_RATIO)
    train_ids = set(annotated_ids[:split_idx])
    val_ids = set(annotated_ids[split_idx:])
    print(f"Train images: {len(train_ids)}, Val images: {len(val_ids)}")

    make_dirs()

    for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
        for img_id in split_ids:
            img = image_info[img_id]
            fname = img["file_name"]
            img_w = img["width"]
            img_h = img["height"]

            src_path = IMG_SOURCE_DIR / fname
            if not src_path.exists():
                print(f"  WARNING: {src_path} not found, skipping")
                continue

            dst_name = Path(fname).name
            dst_img = OUTPUT_DIR / "images" / split_name / dst_name
            shutil.copy2(src_path, dst_img)

            yolo_lines = []
            for ann in annotations_by_image[img_id]:
                cat_id = ann["category_id"]
                class_idx = cat_map[cat_id]
                xc, yc, w, h = convert_bbox(ann["bbox"], img_w, img_h)
                yolo_lines.append(f"{class_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            dst_label = OUTPUT_DIR / "labels" / split_name / (Path(fname).stem + ".txt")
            with open(dst_label, "w") as f:
                f.write("\n".join(yolo_lines))

    print("Done. Data prepared at:", OUTPUT_DIR)

    class_list = ", ".join(class_names)
    split_info = f"train: {len(train_ids)}, val: {len(val_ids)}"
    print(f"  Classes: [{class_list}]")
    print(f"  Split: {split_info}")


if __name__ == "__main__":
    main()
