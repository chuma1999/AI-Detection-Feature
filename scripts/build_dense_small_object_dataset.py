from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

root = Path("C:/Users/info/OneDrive/Desktop/AI Detection")
source_dir = root / "input" / "test_frames"
train_dir = root / "dataset" / "images" / "train_dense_small"
val_dir = root / "dataset" / "images" / "val_dense_small"
train_label_dir = root / "dataset" / "labels" / "train_dense_small"
val_label_dir = root / "dataset" / "labels" / "val_dense_small"

for folder in [train_dir, val_dir, train_label_dir, val_label_dir]:
    folder.mkdir(parents=True, exist_ok=True)

manual_label_root = root / "dataset" / "labels" / "train_balanced"

# Keep a few real frames as validation while expanding the training set with
# zoomed variants of the same real-world examples to emphasize small objects.
val_names = {"test-image-4.jpg", "test-image-6.jpg"}


def load_boxes(image_name: str) -> list[tuple[int, float, float, float, float]]:
    label_path = manual_label_root / (Path(image_name).stem + ".txt")
    if not label_path.exists():
        return []

    boxes: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue
        boxes.append((cls_id, cx, cy, w, h))
    return boxes


def write_label_file(label_path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    lines = [f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls_id, cx, cy, w, h in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def scale_label_box(cls_id: int, cx: float, cy: float, w: float, h: float, scale: float) -> tuple[int, float, float, float, float]:
    # Simulate a zoomed crop by expanding the normalized box and shifting the center toward the middle.
    offset = (scale - 1.0) / 2.0
    new_cx = cx * scale - offset
    new_cy = cy * scale - offset
    new_w = min(0.99, w * scale)
    new_h = min(0.99, h * scale)
    return cls_id, float(np.clip(new_cx, 0.0, 1.0)), float(np.clip(new_cy, 0.0, 1.0)), float(np.clip(new_w, 0.01, 1.0)), float(np.clip(new_h, 0.01, 1.0))


# Build training variants of each real frame.
for src in sorted(source_dir.glob("*.jpg")):
    image_name = src.name
    boxes = load_boxes(image_name)
    if not boxes:
        continue

    target_dir = val_dir if image_name in val_names else train_dir
    target_label_dir = val_label_dir if image_name in val_names else train_label_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_label_dir.mkdir(parents=True, exist_ok=True)

    # Keep the original frame as a baseline sample.
    dst_path = target_dir / image_name
    dst_path.write_bytes(src.read_bytes())
    write_label_file(target_label_dir / (src.stem + ".txt"), boxes)

    for scale in [1.2, 1.35, 1.5]:
        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        scaled_w = int(round(w * scale))
        scaled_h = int(round(h * scale))
        resized = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        left = (scaled_w - w) // 2
        top = (scaled_h - h) // 2
        cropped = resized[top:top + h, left:left + w]

        variant_name = f"{src.stem}_zoom_{scale:.2f}.jpg"
        cv2.imwrite(str(target_dir / variant_name), cropped)

        shifted_boxes = [scale_label_box(cls_id, cx, cy, bw, bh, scale) for cls_id, cx, cy, bw, bh in boxes]
        write_label_file(target_label_dir / (Path(variant_name).stem + ".txt"), shifted_boxes)

# Write dataset config for the expanded dense small-object set.
(root / "dataset" / "data_small_clustered_dense.yaml").write_text(
    "train: ./images/train_dense_small\n"
    "val: ./images/val_dense_small\n"
    "test: ./images/test\n\n"
    "nc: 5\n"
    "names: ['person', 'car', 'motorcycle', 'bus', 'truck']\n",
    encoding="utf-8",
)

print(f"Created training split: {train_dir}")
print(f"Created validation split: {val_dir}")
print(f"YAML: {root / 'dataset' / 'data_small_clustered_dense.yaml'}")
print(f"Train images: {len(list(train_dir.glob('*.jpg')))}")
print(f"Val images: {len(list(val_dir.glob('*.jpg')))}")
