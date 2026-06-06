import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image
import torchvision.transforms.functional as TF

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class ChestXRaySegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, augment=True):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.augment = augment
        self.images = sorted(
            f for f in images_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_dir / (img_path.stem + ".png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = TF.resize(img, [256, 256])
        mask = TF.resize(mask, [256, 256], interpolation=TF.InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)

        if mask.max() > 0:
            mask = mask / mask.max()

        if self.augment and random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if self.augment and random.random() < 0.5:
            angle = random.uniform(-10, 10)
            img = TF.affine(img, angle=angle, translate=(0, 0), scale=1.0, shear=0)
            mask = TF.affine(mask, angle=angle, translate=(0, 0), scale=1.0, shear=0)
        if self.augment and random.random() < 0.3:
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))

        return img, mask


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    d_loss = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * d_loss


def compute_iou_dice(pred_logits, target, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2 * intersection + 1e-7) / (pred.sum() + target.sum() + 1e-7)
    return float(iou), float(dice)


def train():
    data_dir = Path(__file__).parent / "data"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = ChestXRaySegDataset(
        data_dir / "images" / "train",
        data_dir / "masks" / "train",
        augment=True,
    )
    val_ds = ChestXRaySegDataset(
        data_dir / "images" / "val",
        data_dir / "masks" / "val",
        augment=False,
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    print("Loading U-Net with ResNet34 encoder (pre-trained on ImageNet)...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    use_amp = device == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_val_dice = -1
    best_path = results_dir / "best_model.pth"
    patience = 20
    patience_counter = 0

    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for img, mask in train_loader:
            img = img.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(img)
                    loss = combined_loss(logits, mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(img)
                loss = combined_loss(logits, mask)
                loss.backward()
                optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        val_iou_sum = 0
        val_dice_sum = 0
        n = 0
        with torch.no_grad():
            for img, mask in val_loader:
                img = img.to(device)
                mask = mask.to(device)
                logits = model(img)
                loss = combined_loss(logits, mask)
                val_loss += loss.item()
                iou, dice = compute_iou_dice(logits, mask)
                val_iou_sum += iou
                val_dice_sum += dice
                n += 1

        avg_train = train_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))
        avg_iou = val_iou_sum / max(1, n)
        avg_dice = val_dice_sum / max(1, n)

        scheduler.step(avg_val)
        cur_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss: {avg_train:.4f} | val_loss: {avg_val:.4f} | "
            f"mIoU: {avg_iou:.4f} | Dice: {avg_dice:.4f} | lr: {cur_lr:.6f}"
        )

        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    print(f"\nTraining complete. Best Dice: {best_val_dice:.4f}")
    print(f"Best model: {best_path}")


if __name__ == "__main__":
    train()
