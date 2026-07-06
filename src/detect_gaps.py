"""Out-of-stock gap detection and OSA scoring on shelf images.

SKU-110K has no "gap" labels, so gaps are derived from detection geometry:
  1. detect all products on the shelf,
  2. cluster boxes into shelf rows by vertical-center proximity,
  3. inside each row, merge box x-intervals and flag empty stretches wider
     than --min-gap-ratio x the row's median product width,
  4. OSA (on-shelf availability) = occupied row width / total row width.

Usage (random test-split shelves):
    python src/detect_gaps.py --checkpoint outputs/yolo-sku110k/weights/best.pt --n 6

Usage (your own photo):
    python src/detect_gaps.py --checkpoint ... --source my_shelf.jpg

Annotated images (green = product, red = gap) go to outputs/gaps/.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

from evaluate import list_images


def group_rows(boxes):
    """Cluster boxes (xyxy) into shelf rows by vertical-center proximity."""
    centers = (boxes[:, 1] + boxes[:, 3]) / 2
    heights = boxes[:, 3] - boxes[:, 1]
    rows = []  # list of lists of box indices
    for i in np.argsort(centers):
        placed = False
        for row in rows:
            row_yc = np.mean([centers[j] for j in row])
            row_h = np.median([heights[j] for j in row])
            if abs(centers[i] - row_yc) < 0.5 * row_h:
                row.append(i)
                placed = True
                break
        if not placed:
            rows.append([i])
    return rows


def find_gaps(boxes, row, min_gap_ratio):
    """Return (gaps, occupied_width, row_extent) for one shelf row.

    Only interior gaps are flagged: space before the first / after the last
    product could simply be the edge of the photo.
    """
    xs = sorted((boxes[i][0], boxes[i][2]) for i in row)
    med_w = np.median([x2 - x1 for x1, x2 in xs])
    merged = [list(xs[0])]
    for x1, x2 in xs[1:]:
        if x1 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x2)
        else:
            merged.append([x1, x2])
    gaps = [(a[1], b[0]) for a, b in zip(merged, merged[1:])
            if b[0] - a[1] >= min_gap_ratio * med_w]
    occupied = sum(x2 - x1 for x1, x2 in merged)
    extent = merged[-1][1] - merged[0][0]
    return gaps, occupied, extent


def analyze_image(model, img_path, out_dir, conf, min_gap_ratio, min_row_boxes=3):
    res = model.predict(img_path, conf=conf, max_det=1000, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    img = cv2.imread(str(img_path))

    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)

    occupied_total = extent_total = 0.0
    n_gaps = 0
    for row in group_rows(boxes) if len(boxes) else []:
        if len(row) < min_row_boxes:
            continue
        gaps, occupied, extent = find_gaps(boxes, row, min_gap_ratio)
        occupied_total += occupied
        extent_total += extent
        n_gaps += len(gaps)
        y1 = int(np.median([boxes[i][1] for i in row]))
        y2 = int(np.median([boxes[i][3] for i in row]))
        for gx1, gx2 in gaps:
            cv2.rectangle(img, (int(gx1), y1), (int(gx2), y2), (0, 0, 255), 3)
            cv2.putText(img, "GAP", (int(gx1) + 4, (y1 + y2) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    osa = occupied_total / extent_total if extent_total else 1.0
    label = f"products: {len(boxes)}  gaps: {n_gaps}  OSA: {osa:.0%}"
    cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
    cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    out_path = out_dir / f"gaps_{Path(img_path).stem}.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"{Path(img_path).name}: {label} -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/yolo-sku110k/weights/best.pt")
    p.add_argument("--source", default=None, help="image file; default samples the test split")
    p.add_argument("--n", type=int, default=6, help="test images to sample when no --source")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--min-gap-ratio", type=float, default=0.8,
                   help="gap = empty stretch wider than this x median product width")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.source:
        images = [Path(args.source)]
    else:
        data = check_det_dataset("SKU-110K.yaml")
        images = list_images(data["test"])
        random.seed(args.seed)
        images = random.sample(images, min(args.n, len(images)))

    out_dir = Path("outputs/gaps")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.checkpoint)
    for img_path in images:
        analyze_image(model, img_path, out_dir, args.conf, args.min_gap_ratio)


if __name__ == "__main__":
    main()
