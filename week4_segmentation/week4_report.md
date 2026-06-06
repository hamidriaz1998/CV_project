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
- **Hardware:** Google Colab (T4 GPU)

## Results (Measured on Validation Set)

| Metric | Value |
|--------|-------|
| **mIoU** | **0.2846** |
| **Dice Coefficient** | **0.4142** |
| Pixel Accuracy | 0.9520 |
| Precision | 0.4764 |
| Recall | 0.5002 |

## Discussion

The achieved mIoU (0.28) and Dice (0.41) are lower than typical benchmarks (Dice > 0.80) on this task. Three factors limit performance:

1. **Small training set:** Only 40 training images. U-Net typically requires hundreds to thousands of examples to generalize well.
2. **Imperfect ground truth:** SAM-generated masks are derived from bounding-box prompts that we drew on the original X-rays. The boxes are coarse (whole lung/affected region), so SAM is essentially being asked to find the most salient object in each box — not a precise pathology outline. The resulting masks have inconsistent boundary quality across the 50 images, which limits the ceiling of the trained U-Net.
3. **Class imbalance:** The pneumonia opacity occupies a small fraction of each 224x224 image. Pixel accuracy (~0.95) is high because the model can simply predict "background" everywhere and still be right 95% of the time. The lower mIoU/Dice correctly reflect that the model is only partly capturing the foreground.

Despite the modest absolute numbers, the model does learn meaningful structure — qualitative prediction overlays (in `predictions/`) show the U-Net captures the rough location and extent of the opacity region, even when the precise boundary is fuzzy.

## Tools Used
- **Meta Segment Anything Model (SAM)** — mask generation from bounding box prompts
- **U-Net** — semantic segmentation architecture
- **segmentation_models_pytorch** — pre-built U-Net with ResNet34 encoder
- **PyTorch** — deep learning backend
- **Google Colab** — cloud compute platform (T4 GPU)
- **Custom FastAPI annotation tool** (`annotator.py`) — source of bounding boxes

## Deliverables
- `trained_model/best_model.pth` (94 MB) — trained U-Net model weights
- `trained_model/metrics.json` — mIoU, Dice, accuracy, precision, recall
- `predictions/` — 10 side-by-side ground truth vs prediction overlays (522x256 PNG)
- `paper/paper.pdf` — IEEE-format research paper
- `paper/paper.typ` — Typst source for the research paper
