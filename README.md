# Patient Health Monitoring — Chest X-Ray Classification

Computer Vision project for detecting pneumonia from chest X-ray images using deep learning.

## Project Structure

```
.
├── chest_xray/              # Dataset — see download instructions below
│   ├── train/               #   ├── NORMAL/
│   │   └── PNEUMONIA/       #   └── ...
│   ├── val/
│   └── test/
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

Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Kermany et al.

### Option 1 — kagglehub (recommended)

```bash
uv add kagglehub
uv run python -c "
import kagglehub
path = kagglehub.dataset_download('paultimothymooney/chest-xray-pneumonia')
print(f'Downloaded to: {path}')
"
```

Then copy or symlink the `chest_xray` folder to the project root.

### Option 2 — Kaggle CLI

```bash
kaggle datasets download paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip
```

### Option 3 — Manual

Download the zip from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place `chest-xray-pneumonia.zip` in the project root.

### Post-extraction cleanup

The zip contains an extra nested `chest_xray/chest_xray/` directory and a `__MACOSX/` junk folder. Run these commands to clean up:

```bash
unzip chest-xray-pneumonia.zip
rm -rf chest_xray/chest_xray chest_xray/__MACOSX
ls chest_xray/       # should show: train/  val/  test/
```

Expected structure after cleanup:

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
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

Output:
```
Split     Initial  Corrupt Duplicates    Final
------   --------  ------- ----------   ------
train       5216        0         26     5190
val           16        0          0       16
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
