"""Fine-tune a small YOLO on SKU-110K for dense retail shelf product detection.

Usage (smoke test, ~2% of train data, also triggers the ~13.6 GB dataset
download on first run):
    python src/train.py --fraction 0.02 --epochs 1

Usage (full run):
    python src/train.py --epochs 15

SKU-110K averages ~147 boxes per image (up to ~700), so the validation
detection cap is raised above the Ultralytics default of 300.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--fraction", type=float, default=1.0,
                   help="fraction of the training split to use (smoke tests)")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data="SKU-110K.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        max_det=1000,
        project=str(ROOT / "outputs"),
        name="yolo-sku110k",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
