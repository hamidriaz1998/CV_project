# Patient Health Monitoring — Chest X-Ray Multi-Task Deep Learning Pipeline

Computer Vision project for detecting pneumonia from chest X-ray images using a complete deep learning pipeline: **classification** (image-level), **object detection** (bounding-box level), and **semantic segmentation** (pixel-level).

## Project Structure

```
.
├── chest_xray/              # Dataset — see download instructions below
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── docs/
│   ├── week1_dataset_proposal/
│   │   └── dataset_proposal.md
│   └── week2/
│       └── report.md
├── week2_classification/    # ResNet50 classification (Week 2)
│   ├── classification.ipynb
│   ├── best_model.pth
│   └── metrics.json
├── week3_detection/         # YOLOv8n object detection (Week 3)
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── colab_week3.ipynb
│   └── data/
├── week4_segmentation/      # U-Net + SAM segmentation (Week 4)
│   ├── generate_masks.py
│   ├── train_unet.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── colab_week4.ipynb
│   ├── data/
│   └── paper/
│       ├── paper.typ        # IEEE-format Typst source
│       └── paper.pdf        # Compiled paper
├── annotator.py             # FastAPI web-based bounding-box annotation tool
├── clean_dataset.py         # Dataset cleaning pipeline
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

```bash
uv sync
```

## Dataset Download

**Dataset:** Chest X-Ray Images (Pneumonia) — Wang et al. 2017
**Source:** [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2)
**Paper:** [Cell 2018](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)
**Citation:** Wang, X. et al. "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks." arXiv:1705.02315, 2017. *(Dataset hosted on Mendeley by Kermany et al.)*

Download `ChestXRay2017.zip` (1.15 GB) from the Mendeley page, place it in the project root, then:

```bash
unzip ChestXRay2017.zip
# Cleanup any nested dirs or macOS junk if present
find chest_xray -mindepth 2 -maxdepth 2 -type d | head -20
ls chest_xray/       # should show: train/  test/
```

The dataset contains `train/` and `test/` splits only (no `val/` split).

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Dataset Cleaning

Removes corrupt images, deduplicates by MD5 hash, resizes all images to 224×224 RGB, and overwrites originals in place.

```bash
uv run python clean_dataset.py
```

Output after running on the downloaded dataset:
```
Split     Initial  Corrupt Duplicates    Final
------   --------  ------- ----------   ------
train       5232        0         26     5206
test         624        0          6      618
------   --------  ------- ----------   ------
TOTAL       5856        0         32     5824
```

## Bounding-Box Annotation Tool

Interactive pygame tool for annotating lung-field regions in chest X-rays.

```bash
uv run python annotator.py
```

**Controls:**

| Key | Action |
|-----|--------|
| Click + drag | Draw bounding box |
| N | Label as NORMAL |
| P | Label as PNEUMONIA |
| S | Save and go to next image |
| R | Clear box and redraw |
| Q | Quit and save progress |

Annotations are exported to `annotations.json` and `annotations_coco.json` in COCO format (filename, label, x1, y1, x2, y2). The same 50 annotated images are reused for both object detection (Week 3) and segmentation (Week 4).

## Results Summary

| Task | Model | Key Metric | Value |
|------|-------|-----------|-------|
| **Classification** (Week 2) | ResNet50 | Test Accuracy | 81.07% |
| **Classification** (Week 2) | ResNet50 | ROC-AUC | 0.9301 |
| **Object Detection** (Week 3) | YOLOv8n | mAP@0.5 | 0.626 |
| **Object Detection** (Week 3) | YOLOv8n | mAP@0.5:0.95 | 0.297 |
| **Segmentation** (Week 4) | U-Net + SAM | mIoU | 0.74 |
| **Segmentation** (Week 4) | U-Net + SAM | Dice | 0.82 |

## Week 3 — Object Detection

```bash
cd week3_detection
uv run python prepare_data.py  # Convert COCO → YOLO format, split train/val
# Then upload colab_package.zip and colab_week3.ipynb to Google Colab
# Or for local training:
uv run python train.py
uv run python evaluate.py
uv run python visualize.py
```

See `week3_detection/week3_report.md` for full details.

## Week 4 — Segmentation

```bash
cd week4_segmentation
uv run python generate_masks.py  # Box-derived masks (initial)
# Then upload colab_package.zip and colab_week4.ipynb to Google Colab
# (Notebook uses SAM to refine masks and train U-Net)
```

The IEEE-format research paper is in `week4_segmentation/paper/paper.pdf`. To recompile:

```bash
typst compile paper/paper.typ
```

See `week4_segmentation/week4_report.md` for full details.