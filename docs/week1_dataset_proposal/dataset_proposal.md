# Dataset Proposal

## 1. Project Title

Patient Health Monitoring using Chest X-Ray Classification

## 2. Selected Dataset

**Chest X-Ray Images (Pneumonia)** — Kermany et al.

**Source:** Kaggle — [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Original citation:** Kermany, D.S. et al. "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." *Cell*, 172(5), 2018. DOI: [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)

## 3. Dataset Description

The dataset consists of **5,863 chest X-ray JPEG images** classified into two classes — **NORMAL** and **PNEUMONIA** — provided pre-split into `train/`, `val/`, and `test/` directories.

All images are anterior-posterior (AP) chest radiographs from **pediatric patients aged 1–5**, collected retrospectively from Guangzhou Women and Children's Medical Center. Image quality was validated and grades assigned by two expert physicians, with a third expert adjudicating the evaluation set.

## 4. Why This Dataset

- **Real clinical data** with expert-verified labels, making it directly applicable to patient health monitoring.
- Contains sufficient samples to support deep learning training without requiring synthetic data augmentation as a prerequisite.
- **Widely used benchmark** in the medical imaging ML community, enabling verifiable comparison against published results.

## 5. Preprocessing Steps

`clean_dataset.py` applies a three-phase pipeline to every image across all splits and classes:

| Phase | Method | Result |
|-------|--------|--------|
| **Corruption detection** | Open each file with Pillow, call `.verify()` | Corrupt/unreadable files silently deleted |
| **Deduplication** | Compute MD5 hash of raw bytes; first occurrence kept | Duplicate images removed |
| **Standardization** | Resize to 224×224 px using LANCZOS resampling; convert grayscale to RGB | All images normalized to `(224, 224, 3)` format compatible with standard CNN architectures |

Surviving images are saved back in place, overwriting originals.

## 6. Annotation Plan

A custom OpenCV-based bounding-box annotator (`annotator.py`) is used to delineate the lung field region in chest X-rays:

- Images are loaded one at a time from `chest_xray/train/`, shuffled randomly.
- The user draws a single bounding box per image via click-and-drag.
- After drawing, the label is assigned by pressing **N** (NORMAL) or **P** (PNEUMONIA).
- **S** saves the annotation; **R** redoes the box; **Q** quits and persists progress.
- Annotations are saved to `annotations.csv` with columns `filename, label, x1, y1, x2, y2`.
- Already-annotated images are skipped on resume.

**Target for Week 1:** 20 annotated sample images, to be expanded in subsequent weeks.

## 7. Class Distribution

Actual counts from `clean_dataset.py` after preprocessing:

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,340 | 3,850 | 5,190 |
| Val | 8 | 8 | 16 |
| Test | 231 | 387 | 618 |
| **Total** | **1,579** | **4,245** | **5,824** |
