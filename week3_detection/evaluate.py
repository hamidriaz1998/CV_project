import json
from pathlib import Path

from ultralytics import YOLO


def evaluate():
    results_dir = Path(__file__).parent / "results" / "yolov8n_chest_xray"
    dataset_yaml = str(Path(__file__).parent / "dataset.yaml")
    model_path = results_dir / "weights" / "best.pt"

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run train.py first.")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(str(model_path))

    print("Running validation...")
    metrics = model.val(
        data=dataset_yaml,
        imgsz=224,
        batch=8,
        save_json=True,
        save_conf=True,
        plots=True,
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "f1": float(metrics.box.f1),
        "class_maps": {},
    }

    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        for cls_idx, cls_name in enumerate(metrics.names.values()):
            cls_map = float(metrics.box.maps[cls_idx]) if cls_idx < len(metrics.box.maps) else 0.0
            results["class_maps"][cls_name] = cls_map

    print(f"  mAP@0.5:      {results['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {results['mAP50_95']:.4f}")
    print(f"  Precision:    {results['precision']:.4f}")
    print(f"  Recall:       {results['recall']:.4f}")
    print(f"  F1-score:     {results['f1']:.4f}")
    print()
    if results["class_maps"]:
        print("  Per-class mAP@0.5:")
        for cls, m in results["class_maps"].items():
            print(f"    {cls}: {m:.4f}")

    out_path = results_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    evaluate()
