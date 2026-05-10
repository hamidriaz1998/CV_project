import os
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm


BASE_DIR = Path("chest_xray")
SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]
EXTENSIONS = {".jpeg", ".jpg"}


def walk_images():
    """Collect all image paths organized by (split, class)."""
    images = defaultdict(list)
    for split in SPLITS:
        for cls in CLASSES:
            dirpath = BASE_DIR / split / cls
            if not dirpath.exists():
                continue
            for fname in sorted(os.listdir(dirpath)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in EXTENSIONS:
                    images[(split, cls)].append(dirpath / fname)
    return images


def phase1_remove_corrupt(images_by_key):
    """Remove images that can't be opened by Pillow. Returns corrupt count per key."""
    corrupt = defaultdict(int)
    for key in list(images_by_key.keys()):
        paths = images_by_key[key]
        surviving = []
        for p in tqdm(paths, desc=f"  Corrupt check {key[0]}/{key[1]}", leave=False):
            try:
                img = Image.open(p)
                img.verify()
                surviving.append(p)
            except Exception:
                p.unlink()
                corrupt[key] += 1
        images_by_key[key] = surviving
    return corrupt


def phase2_remove_duplicates(images_by_key):
    """Remove duplicates by MD5 hash. Returns dup count per key."""
    dups = defaultdict(int)
    for key in list(images_by_key.keys()):
        paths = images_by_key[key]
        seen_hashes = {}
        surviving = []
        for p in tqdm(paths, desc=f"  Dedup check {key[0]}/{key[1]}", leave=False):
            data = p.read_bytes()
            md5 = hashlib.md5(data).hexdigest()
            if md5 in seen_hashes:
                p.unlink()
                dups[key] += 1
            else:
                seen_hashes[md5] = p
                surviving.append(p)
        images_by_key[key] = surviving
    return dups


def phase3_resize_rgb(images_by_key):
    """Resize to 224x224 and convert to RGB."""
    for key in list(images_by_key.keys()):
        paths = images_by_key[key]
        for p in tqdm(paths, desc=f"  Resize {key[0]}/{key[1]}", leave=False):
            img = Image.open(p)
            img = img.resize((224, 224), Image.LANCZOS)
            img = img.convert("RGB")
            img.save(p)


def print_summary(initial, corrupt, dups, final):
    """Print summary table with totals per split."""
    print()
    header = f"{'Split':<8} {'Initial':>8} {'Corrupt':>8} {'Duplicates':>10} {'Final':>8}"
    print(header)
    print("-" * len(header))
    for split in SPLITS:
        init_total = sum(initial.get((split, c), 0) for c in CLASSES)
        corr_total = sum(corrupt.get((split, c), 0) for c in CLASSES)
        dup_total = sum(dups.get((split, c), 0) for c in CLASSES)
        final_total = sum(final.get((split, c), 0) for c in CLASSES)
        print(f"{split:<8} {init_total:>8} {corr_total:>8} {dup_total:>10} {final_total:>8}")
    print("-" * len(header))
    grand_init = sum(initial.values())
    grand_corr = sum(corrupt.values())
    grand_dup = sum(dups.values())
    grand_final = sum(final.values())
    print(f"{'TOTAL':<8} {grand_init:>8} {grand_corr:>8} {grand_dup:>10} {grand_final:>8}")
    print()


def main():
    if not BASE_DIR.exists():
        print(f"Error: {BASE_DIR} not found. Download the dataset first.")
        return

    print("Walking dataset...")
    images = walk_images()
    initial = {key: len(paths) for key, paths in images.items()}
    total_init = sum(initial.values())
    print(f"Found {total_init} images\n")

    print("Phase 1/3: Removing corrupt images...")
    corrupt = phase1_remove_corrupt(images)

    print("\nPhase 2/3: Removing duplicates...")
    dups = phase2_remove_duplicates(images)

    print("\nPhase 3/3: Resizing to 224x224 and converting to RGB...")
    phase3_resize_rgb(images)

    final = {key: len(paths) for key, paths in images.items()}
    print_summary(initial, corrupt, dups, final)

    total_removed = sum(corrupt.values()) + sum(dups.values())
    final_total = sum(final.values())
    print(f"Done. Removed {total_removed} images, resized {final_total} images.")


if __name__ == "__main__":
    main()
