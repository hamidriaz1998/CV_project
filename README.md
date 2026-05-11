# Patient Health Monitoring — Chest X-Ray Classification

Computer Vision project for detecting pneumonia from chest X-ray images using deep learning.

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
│   │   └── dataset_proposal.md   # Week 1 dataset proposal document
│   └── lab-cover-template.typ
├── annotator.py             # Pygame-based bounding-box annotation tool
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

Annotations are written to `annotations.csv` (filename, label, x1, y1, x2, y2). Already-annotated images are skipped on resume. The tool automatically stops after saving 20 annotations.