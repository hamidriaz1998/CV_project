# Patient Health Monitoring — Chest X-Ray Classification

Computer Vision project for detecting pneumonia from chest X-ray images using deep learning.

## Project Structure

```
.
├── chest_xray/              # Dataset (auto-downloaded on first run)
│   ├── train/               #   ├── NORMAL/
│   │   └── PNEUMONIA/       #   └── ...
│   ├── val/
│   └── test/
├── docs/
│   ├── week1_dataset_proposal/
│   │   └── dataset_proposal.md   # Week 1 dataset proposal document
│   └── lab-cover-template.typ
├── annotator.py             # Custom OpenCV bounding-box annotation tool
├── clean_dataset.py         # Dataset cleaning pipeline
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

Dependencies are managed with `uv`. Install once:

```bash
uv sync
```

## Dataset Cleaning

Removes corrupt images and duplicates, resizes all images to 224×224 RGB, overwrites originals.

```bash
uv run python clean_dataset.py
```

## Bounding-Box Annotation Tool

Interactive OpenCV tool for annotating lung-field regions in chest X-rays.

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

Annotations are written to `annotations.csv` (filename, label, x1, y1, x2, y2). Already-annotated images are skipped on resume.
