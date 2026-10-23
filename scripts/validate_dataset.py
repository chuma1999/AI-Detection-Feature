#!/usr/bin/env python3
"""Validate a YOLO dataset structure.

This script checks the prepared dataset for the most common problems before
training begins. It does not train anything.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

VALID_CLASSES = {0, 1, 2, 3, 4}
NAMES = ["person", "car", "motorcycle", "bus", "truck"]


def validate_split(split_dir: Path, split_name: str, labels_dir: Path | None = None) -> dict[str, int]:
    image_files = sorted(split_dir.glob("**/*"))
    image_files = [path for path in image_files if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    if labels_dir is None:
        labels_dir = split_dir.parent.parent / "labels" / split_name

    missing_labels = 0
    invalid_boxes = 0
    class_counts = Counter()
    duplicates = set()

    seen_names: set[str] = set()
    for image_path in image_files:
        if image_path.name in seen_names:
            duplicates.add(image_path.name)
        seen_names.add(image_path.name)

        label_path = labels_dir / (image_path.stem + ".txt")
        if not label_path.exists():
            missing_labels += 1
            continue

        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                invalid_boxes += 1
                continue
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                invalid_boxes += 1
                continue

            if class_id not in VALID_CLASSES:
                invalid_boxes += 1
                continue
            if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                invalid_boxes += 1
                continue

            class_counts[class_id] += 1

    return {
        "images": len(image_files),
        "missing_labels": missing_labels,
        "invalid_boxes": invalid_boxes,
        "duplicates": len(duplicates),
        "class_counts": dict(class_counts),
        "split_name": split_name,
    }


def validate_dataset(dataset_dir: Path) -> dict[str, dict[str, int]]:
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    split_names = ["train", "val", "test"]
    summary: dict[str, dict[str, int]] = {}
    for split_name in split_names:
        split_dir = dataset_dir / "images" / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Image split directory not found: {split_dir}")

        labels_dir = dataset_dir / "labels" / split_name
        if not labels_dir.exists():
            raise FileNotFoundError(f"Label split directory not found: {labels_dir}")

        summary[split_name] = validate_split(split_dir, split_name, labels_dir=labels_dir)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a prepared YOLO dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"), help="Dataset root folder to validate.")
    args = parser.parse_args()

    summary = validate_dataset(args.dataset_dir)
    for split_name, result in summary.items():
        print(f"{split_name}: images={result['images']}, missing_labels={result['missing_labels']}, invalid_boxes={result['invalid_boxes']}, duplicates={result['duplicates']}")
        if result["class_counts"]:
            print("  class_counts:")
            for class_id, count in sorted(result["class_counts"].items()):
                print(f"    {class_id} ({NAMES[class_id]}): {count}")


if __name__ == "__main__":
    main()
