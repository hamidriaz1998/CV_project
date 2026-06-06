import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COCO_PATH = PROJECT_ROOT / "annotations_coco.json"
IMG_SOURCE_DIR = PROJECT_ROOT / "chest_xray" / "train"
OUTPUT_DIR = Path(__file__).parent / "colab_package"


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    img_dir = OUTPUT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    import json
    coco = json.loads(COCO_PATH.read_text())
    copied = 0
    for img in coco["images"]:
        src = IMG_SOURCE_DIR / img["file_name"]
        if src.exists():
            dst = img_dir / Path(img["file_name"]).name
            shutil.copy2(src, dst)
            img["file_name"] = Path(img["file_name"]).name
            copied += 1
        else:
            print(f"WARNING: {src} not found")

    output_coco = OUTPUT_DIR / "annotations_coco.json"
    with open(output_coco, "w") as f:
        json.dump(coco, f, indent=2)

    zip_path = Path(__file__).parent / "colab_package.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in OUTPUT_DIR.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(OUTPUT_DIR))

    shutil.rmtree(OUTPUT_DIR)
    print(f"Colab package created: {zip_path}")
    print(f"  Images: {copied}")
    print(f"  Size: {zip_path.stat().st_size / 1024:.1f} KB")
    print("\nTo use:")
    print("  1. Go to https://colab.research.google.com")
    print("  2. Upload colab_week4.ipynb")
    print(f"  3. Upload {zip_path.name} when the notebook prompts")


if __name__ == "__main__":
    main()
