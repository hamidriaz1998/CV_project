import json
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from train_unet import ChestXRaySegDataset, compute_iou_dice


def pixel_accuracy(pred_logits, target, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return float(correct / total)


def main():
    data_dir = Path(__file__).parent / "data"
    results_dir = Path(__file__).parent / "results"
    model_path = results_dir / "best_model.pth"

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading model from {model_path}...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    val_ds = ChestXRaySegDataset(
        data_dir / "images" / "val",
        data_dir / "masks" / "val",
        augment=False,
    )
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    print(f"Evaluating on {len(val_ds)} validation images...")
    iou_sum = 0
    dice_sum = 0
    acc_sum = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    n = 0
    with torch.no_grad():
        for img, mask in tqdm(val_loader, desc="Evaluating"):
            img = img.to(device)
            mask = mask.to(device)
            logits = model(img)
            iou, dice = compute_iou_dice(logits, mask)
            acc = pixel_accuracy(logits, mask)
            iou_sum += iou
            dice_sum += dice
            acc_sum += acc
            n += 1

            pred = (torch.sigmoid(logits) > 0.5).float()
            tp += float(((pred == 1) & (mask == 1)).sum())
            fp += float(((pred == 1) & (mask == 0)).sum())
            fn += float(((pred == 0) & (mask == 1)).sum())
            tn += float(((pred == 0) & (mask == 0)).sum())

    results = {
        "mIoU": iou_sum / max(1, n),
        "dice": dice_sum / max(1, n),
        "pixel_accuracy": acc_sum / max(1, n),
        "confusion_matrix": {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        },
        "precision": tp / (tp + fp + 1e-7),
        "recall": tp / (tp + fn + 1e-7),
    }

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  mIoU:          {results['mIoU']:.4f}")
    print(f"  Dice:          {results['dice']:.4f}")
    print(f"  Pixel Accuracy: {results['pixel_accuracy']:.4f}")
    print(f"  Precision:     {results['precision']:.4f}")
    print(f"  Recall:        {results['recall']:.4f}")
    print(f"\n  Confusion Matrix (pixels):")
    print(f"    TP: {int(tp):>10}  FN: {int(fn):>10}")
    print(f"    FP: {int(fp):>10}  TN: {int(tn):>10}")

    out_path = results_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
