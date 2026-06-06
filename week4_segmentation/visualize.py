from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor, to_pil_image

CLASS_COLORS = {
    "GT": (0, 255, 0),       # Green for ground truth
    "Pred": (255, 0, 0),     # Red for predicted
}


def overlay_mask(pil_img, mask, color, alpha=0.4):
    """Overlay a binary mask on the image with given color and transparency."""
    img = pil_img.convert("RGB").copy()
    overlay = Image.new("RGB", img.size, color)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8).astype(np.uint8))
    if mask_img.size != img.size:
        mask_img = mask_img.resize(img.size, Image.NEAREST)
    blended = Image.blend(img, overlay, alpha)
    img_arr = np.array(img)
    mask_arr = np.array(mask_img) > 127
    img_arr[mask_arr] = (alpha * np.array(color) + (1 - alpha) * img_arr[mask_arr]).astype(np.uint8)
    return Image.fromarray(img_arr)


def add_text(img, text, position=(5, 5)):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text(position, text, fill=(255, 255, 255), font=font)
    return img


def main():
    data_dir = Path(__file__).parent / "data"
    results_dir = Path(__file__).parent / "results"
    out_dir = results_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = results_dir / "best_model.pth"

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    val_img_dir = data_dir / "images" / "val"
    val_mask_dir = data_dir / "masks" / "val"
    img_paths = sorted(val_img_dir.iterdir())

    print(f"Visualizing {len(img_paths)} validation images...")
    for img_path in img_paths:
        pil = Image.open(img_path).convert("RGB")
        gt_mask = np.array(Image.open(val_mask_dir / (img_path.stem + ".png")).convert("L"))
        gt_mask = (gt_mask > 127).astype(np.uint8)

        img_resized = pil.resize((256, 256))
        img_tensor = to_tensor(img_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            pred = (torch.sigmoid(logits) > 0.5).cpu().squeeze().numpy().astype(np.uint8)

        pil_resized = pil.resize((256, 256))
        gt_overlay = overlay_mask(pil_resized, gt_mask, CLASS_COLORS["GT"])
        pred_overlay = overlay_mask(pil_resized, pred, CLASS_COLORS["Pred"])

        gt_overlay = add_text(gt_overlay, "Ground Truth")
        pred_overlay = add_text(pred_overlay, "Prediction")

        composite = Image.new("RGB", (256 * 2 + 10, 256), (255, 255, 255))
        composite.paste(gt_overlay, (0, 0))
        composite.paste(pred_overlay, (256 + 10, 0))
        composite.save(out_dir / f"pred_{img_path.stem}.png")

    print(f"Saved {len(img_paths)} prediction images to {out_dir}/")


if __name__ == "__main__":
    main()
