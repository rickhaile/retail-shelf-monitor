"""Evaluate detection quality (mAP) plus the business metric: counting accuracy.

mAP is the computer-vision score, but an inventory system consumes "how many
facings of each product are on this shelf". So in addition to mAP we compare
predicted vs ground-truth product counts per test image and report MAE, MAPE
and Pearson r, with a scatter plot saved to outputs/.

Usage:
    python src/evaluate.py --checkpoint outputs/yolo-sku110k/weights/best.pt
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


def list_images(spec):
    """Resolve a dataset split spec (image dir or .txt list) to image paths."""
    p = Path(spec)
    if p.is_file() and p.suffix == ".txt":
        lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        return [Path(ln) if Path(ln).is_absolute() else p.parent / ln for ln in lines]
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"})


def label_path(img_path):
    """Ultralytics convention: .../images/... -> .../labels/....txt"""
    swapped = str(img_path).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
    return Path(swapped).with_suffix(".txt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/yolo-sku110k/weights/best.pt")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35,
                   help="confidence threshold for the counting metric")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--skip-map", action="store_true",
                   help="skip the mAP pass and only run the counting metric")
    p.add_argument("--chunk", type=int, default=None,
                   help="count at most N new images then exit (rerun to resume)")
    args = p.parse_args()

    model = YOLO(args.checkpoint)

    # 1) standard detection metrics
    det_metrics = {}
    if not args.skip_map:
        val = model.val(data="SKU-110K.yaml", split=args.split,
                        imgsz=args.imgsz, max_det=1000)
        det_metrics = {"mAP50": round(val.box.map50, 4),
                       "mAP50-95": round(val.box.map, 4)}
        print(det_metrics)

    # 2) counting validation
    data = check_det_dataset("SKU-110K.yaml")
    images = list_images(data[args.split])
    if args.max_images:
        images = images[: args.max_images]

    # Crash-safe counting: append per-image counts to a partial CSV so an
    # interrupted run resumes where it stopped, and release cached GPU memory
    # periodically so it cannot creep past the card's dedicated VRAM.
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    partial = out / f"counts_partial_{args.split}.csv"
    done = {}
    if partial.exists():
        for ln in partial.read_text().splitlines():
            name, g, pr = ln.rsplit(",", 2)
            done[name] = (int(g), int(pr))
    todo = [im for im in images if im.name not in done]
    print(f"{len(done)} images already counted, {len(todo)} to go")
    remaining_after_run = 0
    if args.chunk and len(todo) > args.chunk:
        remaining_after_run = len(todo) - args.chunk
        todo = todo[: args.chunk]

    if todo:
        with partial.open("a") as fh:
            results = model.predict(todo, conf=args.conf, max_det=1000,
                                    imgsz=args.imgsz, stream=True, verbose=False)
            for i, (img, res) in enumerate(zip(todo, results)):
                lp = label_path(img)
                gt = len(lp.read_text().splitlines()) if lp.exists() else 0
                done[img.name] = (gt, len(res.boxes))
                fh.write(f"{img.name},{gt},{len(res.boxes)}\n")
                fh.flush()
                if (i + 1) % 100 == 0:
                    torch.cuda.empty_cache()

    if remaining_after_run:
        print(f"chunk finished, {remaining_after_run} images remaining — rerun to resume")
        return

    gt = np.array([done[im.name][0] for im in images], dtype=float)
    pred = np.array([done[im.name][1] for im in images], dtype=float)
    err = pred - gt
    count_metrics = {
        "n_images": len(gt),
        "conf_threshold": args.conf,
        "count_mae": round(float(np.abs(err).mean()), 2),
        "count_mape": round(float((np.abs(err)[gt > 0] / gt[gt > 0]).mean()) * 100, 2),
        "count_pearson_r": round(float(np.corrcoef(gt, pred)[0, 1]), 4),
        "mean_gt_count": round(float(gt.mean()), 1),
    }
    print(count_metrics)

    (out / f"metrics_{args.split}.json").write_text(
        json.dumps({**det_metrics, **count_metrics}, indent=2))

    lim = max(gt.max(), pred.max()) * 1.05
    plt.figure(figsize=(6, 6))
    plt.scatter(gt, pred, s=8, alpha=0.3)
    plt.plot([0, lim], [0, lim], "r--", lw=1, label="perfect count")
    plt.xlabel("ground-truth products per image")
    plt.ylabel("predicted products per image")
    plt.title(f"Counting validation ({args.split}), r={count_metrics['count_pearson_r']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"count_validation_{args.split}.png", dpi=150)
    print(f"saved outputs/metrics_{args.split}.json and count scatter plot")


if __name__ == "__main__":
    main()
