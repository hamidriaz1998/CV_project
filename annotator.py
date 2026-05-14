import csv
import io
import json
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="Image Annotator")

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
ANNOTATIONS_FILE = BASE_DIR / "annotations.json"
CLASSES_FILE = BASE_DIR / "classes.json"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# --- Models ---

class AnnotationIn(BaseModel):
    filename: str
    label: str
    x1: int
    y1: int
    x2: int
    y2: int

# --- In-memory state ---

annotations: List[dict] = []
next_ann_id = 1
classes: List[str] = []

# --- Persistence ---

def _load_data():
    global annotations, next_ann_id, classes

    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE) as f:
            data = json.load(f)
        annotations = []
        for i, item in enumerate(data, 1):
            for key in ("x1", "y1", "x2", "y2"):
                if key in item:
                    item[key] = int(item[key])
            item.setdefault("id", str(i))
            item.setdefault("created_at", "")
            for key in ("filename", "label", "x1", "y1", "x2", "y2"):
                item.setdefault(key, "" if key in ("filename", "label") else 0)
            annotations.append(item)
        next_ann_id = max(int(a["id"]) for a in annotations) + 1 if annotations else 1
    else:
        annotations = []
        next_ann_id = 1

    if CLASSES_FILE.exists():
        with open(CLASSES_FILE) as f:
            classes = json.load(f)
    else:
        classes = ["NORMAL", "PNEUMONIA"]
        _save_classes()

def _save_annotations():
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations, f, indent=2)

def _save_classes():
    with open(CLASSES_FILE, "w") as f:
        json.dump(classes, f, indent=2)

def _get_image_size(filename):
    path = UPLOAD_DIR / filename
    try:
        with Image.open(path) as img:
            return img.size
    except (FileNotFoundError, OSError):
        return (0, 0)

# --- Static files ---

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- Routes ---

@app.get("/")
async def root():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return {"message": "Annotator API is running. Create static/index.html for the UI."}

@app.get("/images")
async def list_images():
    files = []
    for f in sorted(UPLOAD_DIR.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            files.append({"filename": f.name, "size": f.stat().st_size})
    return files

@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    uploaded = []
    for file in files:
        content = await file.read()
        (UPLOAD_DIR / file.filename).write_bytes(content)
        uploaded.append(file.filename)
    return {"uploaded": uploaded}

@app.get("/images/{name:path}")
async def get_image(name: str):
    path = UPLOAD_DIR / name
    if not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(path)

@app.get("/annotations")
async def get_annotations():
    return annotations

@app.post("/annotate")
async def create_annotation(ann: AnnotationIn):
    global next_ann_id

    x1, x2 = sorted((ann.x1, ann.x2))
    y1, y2 = sorted((ann.y1, ann.y2))

    img_path = UPLOAD_DIR / ann.filename
    if not img_path.exists():
        raise HTTPException(400, f"Image '{ann.filename}' not found in uploads")

    entry = {
        "filename": ann.filename,
        "label": ann.label,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "id": str(next_ann_id),
        "created_at": datetime.now().isoformat(),
    }
    next_ann_id += 1
    annotations.append(entry)
    _save_annotations()
    return entry

@app.delete("/annotate/{ann_id}")
async def delete_annotation(ann_id: str):
    global annotations
    annotations = [a for a in annotations if a["id"] != ann_id]
    _save_annotations()
    return {"ok": True}

@app.get("/classes")
async def get_classes():
    return classes

@app.post("/classes")
async def add_class(name: str = Query(...)):
    name = name.upper().strip()
    if not name:
        raise HTTPException(400, "Class name cannot be empty")
    if name in classes:
        raise HTTPException(400, f"Class '{name}' already exists")
    classes.append(name)
    _save_classes()
    return classes

@app.delete("/classes/{name}")
async def remove_class(name: str):
    name = name.upper().strip()
    if name not in classes:
        raise HTTPException(404, f"Class '{name}' not found")
    classes.remove(name)
    _save_classes()
    return classes

@app.get("/export")
async def export_annotations(format: str = Query("json")):
    if format == "json":
        return _export_json()
    elif format == "csv":
        return _export_csv()
    elif format == "coco":
        return _export_coco()
    raise HTTPException(400, f"Unsupported format: {format}")

def _export_json():
    data = []
    for a in annotations:
        data.append({
            "filename": a["filename"],
            "label": a["label"],
            "x1": a["x1"],
            "y1": a["y1"],
            "x2": a["x2"],
            "y2": a["y2"],
        })
    return StreamingResponse(
        io.StringIO(json.dumps(data, indent=2)),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=annotations.json"},
    )

def _export_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["filename", "label", "x1", "y1", "x2", "y2"])
    for a in annotations:
        writer.writerow([a["filename"], a["label"], a["x1"], a["y1"], a["x2"], a["y2"]])
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=annotations.csv"},
    )

def _export_coco():
    cat_names = sorted(set(a["label"] for a in annotations))
    cat_map = {name: i + 1 for i, name in enumerate(cat_names)}

    img_ids = {}
    images = []
    coco_annotations = []

    for a in annotations:
        fname = a["filename"]
        if fname not in img_ids:
            img_id = len(img_ids) + 1
            img_ids[fname] = img_id
            w, h = _get_image_size(fname)
            images.append({
                "id": img_id,
                "file_name": fname,
                "width": w,
                "height": h,
            })

        img_id = img_ids[fname]
        bw = a["x2"] - a["x1"]
        bh = a["y2"] - a["y1"]
        ann_id = len(coco_annotations) + 1
        coco_annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": cat_map[a["label"]],
            "bbox": [a["x1"], a["y1"], bw, bh],
            "area": bw * bh,
            "iscrowd": 0,
        })

    categories = [
        {"id": cid, "name": name}
        for name, cid in sorted(cat_map.items(), key=lambda x: x[1])
    ]

    coco = {
        "images": images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    return StreamingResponse(
        io.StringIO(json.dumps(coco, indent=2)),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=annotations_coco.json"},
    )

# --- Startup ---

_load_data()
