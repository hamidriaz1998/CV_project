# Week 3: Object Detection — Report

## Approach
Trained a YOLOv8n (nano) object detection model on **50 annotated chest X-ray images** (25 NORMAL, 25 PNEUMONIA) for pneumonia detection. The dataset was split 80/20 into train (40 images) and validation (10 images).

**Annotation strategy:**
- **PNEUMONIA images:** bounding boxes drawn around visible opacities/consolidation (white patches in the lung) that indicate pneumonia
- **NORMAL images:** bounding boxes drawn around general lung field areas

## Model
- **Architecture:** YOLOv8n (3.0M parameters) — pre-trained on COCO, fine-tuned on chest X-rays
- **Framework:** Ultralytics (PyTorch)
- **Training:** 100 epochs max, early stopping triggered at epoch 64 (patience=20), best epoch = 44
- **Augmentation:** mosaic, flip, rotation, scale, shear, color jitter
- **Input size:** 224×224
- **Batch size:** 16
- **Optimizer:** Adam (lr=0.001)
- **Hardware:** Google Colab (CPU — AMD EPYC 7B12)

## Results

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **0.626** (62.6%) |
| **mAP@0.5:0.95** | 0.297 |
| **Precision** | 0.665 |
| **Recall** | 0.375 |

**Per-class mAP@0.5:**
- NORMAL: 0.508
- PNEUMONIA: 0.745

## Tools Used
- **Ultralytics YOLOv8** — object detection framework
- **PyTorch** — deep learning backend
- **Google Colab** — cloud compute platform
- **Custom annotation tool** — FastAPI web app for bounding box annotation (annotator.py)
- **COCO format** — dataset annotation format

## Deliverables
- `best.pt` — trained YOLOv8n model weights
- `predictions/` — side-by-side ground truth vs prediction images
- `metrics.json` — evaluation metrics
- `results.png` — training curves
- `confusion_matrix.png` — confusion matrix
