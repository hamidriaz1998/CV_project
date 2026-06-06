# Week 4: Segmentation — Report

## Approach
Trained a U-Net (with ResNet34 encoder pre-trained on ImageNet) for semantic segmentation of pneumonia-affected regions in chest X-rays. Ground truth masks were generated using **Meta's Segment Anything Model (SAM)** with each bounding box from the week 3 annotations used as a prompt. This produces organically-shaped masks that follow the actual boundary of the opacity, rather than simple rectangular masks.

**Pipeline:**
1. **Mask generation:** SAM (ViT-B) prompted with bounding box → precise binary mask
2. **Training:** U-Net with ResNet34 encoder, 50 images split 80/20 (40 train / 10 val)
3. **Loss:** Combined Dice loss + binary cross-entropy
4. **Augmentation:** horizontal flip, brightness/contrast jitter

## Model
- **Architecture:** U-Net (encoder: ResNet34, pre-trained on ImageNet)
- **Framework:** PyTorch + segmentation_models_pytorch
- **Input size:** 256×256
- **Training:** 100 epochs max, early stopping (patience=20)
- **Optimizer:** Adam (lr=1e-4)
- **Hardware:** Google Colab (CPU)

## Results

| Metric | Value |
|--------|-------|
| **mIoU** | **0.74** |
| **Dice Coefficient** | **0.82** |
| Pixel Accuracy | 0.96 |
| Precision | 0.85 |
| Recall | 0.80 |

## Tools Used
- **Meta Segment Anything Model (SAM)** — mask generation from bounding box prompts
- **U-Net** — semantic segmentation architecture
- **segmentation_models_pytorch** — pre-built U-Net with ResNet34 encoder
- **PyTorch** — deep learning backend
- **Google Colab** — cloud compute platform
- **Custom FastAPI annotation tool** (annotator.py) — source of bounding boxes

## Deliverables
- `best_model.pth` — trained U-Net model weights
- `predictions/` — side-by-side ground truth vs predicted mask overlays
- `metrics.json` — mIoU, Dice, accuracy, precision, recall
- `paper/paper.pdf` — IEEE-format research paper
- `paper/paper.typ` — Typst source for the research paper
