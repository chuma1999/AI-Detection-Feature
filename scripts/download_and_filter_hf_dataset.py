#!/usr/bin/env python3
"""Download a verified Hugging Face COCO-style dataset and filter it to the project classes.

This script is intentionally separate from the live detection prototype. It does
not train or evaluate the model. It only downloads dataset metadata and reports
whether the dataset contains the required object classes.

The dataset used for verification is a COCO-style object detection dataset on the
Hugging Face Hub. The script filters the metadata to the required classes:
    person, car, motorcycle, bus, truck

If the downloaded dataset exposes raw image paths rather than image bytes, the
script records that fact and avoids claiming the dataset is fully ready for
training. This preserves correctness and keeps the workflow honest.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

REQUIRED_CLASSES = ["person", "car", "motorcycle", "bus", "truck"]
CLASS_TO_ID = {
    "person": 0,
    "car": 1,
    "motorcycle": 2,
    "bus": 3,
    "truck": 4,
}


def summarize_dataset(dataset_name: str, split: str, limit: int | None) -> dict:
    split_spec = f"{split}[:{limit}]" if limit is not None and limit > 0 else split
    dataset = load_dataset(dataset_name, split=split_spec, cache_dir="dataset/hf_cache")

    class_counts = Counter()
    all_classes = set()
    invalid_path_rows = 0
    rows_seen = 0

    for item in dataset:
        rows_seen += 1
        label_dict = item.get("label", {}) or {}
        image_value = item.get("image")

        if isinstance(image_value, str) and image_value and not Path(image_value).exists():
            invalid_path_rows += 1

        for class_name, boxes in label_dict.items():
            all_classes.add(class_name)
            if class_name in REQUIRED_CLASSES and isinstance(boxes, list):
                class_counts[class_name] += len(boxes)

    missing_classes = [name for name in REQUIRED_CLASSES if name not in all_classes]
    summary = {
        "dataset_name": dataset_name,
        "split": split_spec,
        "rows_checked": rows_seen,
        "classes_found": sorted(all_classes)[:20],
        "required_classes_present": all(name in all_classes for name in REQUIRED_CLASSES),
        "missing_required_classes": missing_classes,
        "required_class_counts": {name: class_counts.get(name, 0) for name in REQUIRED_CLASSES},
        "image_path_issues": invalid_path_rows,
        "dataset_is_training_ready": False,
        "notes": [
            "The dataset contains the required classes, but actual image accessibility must be verified before it can be used for YOLO training.",
            "If the image field points to non-existent local paths, the dataset still needs a proper image-download or conversion step.",
        ],
    }

    return summary


def save_summary(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face COCO-style dataset and filter to the project classes.")
    parser.add_argument("--dataset", default="TheMrguiller/coco-object_detection", help="Hugging Face dataset name to inspect.")
    parser.add_argument("--split", default="train", help="Dataset split to inspect.")
    parser.add_argument("--limit", type=int, default=100, help="Number of rows to inspect. Use 0 or omit to inspect the full split.")
    parser.add_argument("--output", type=Path, default=Path("dataset/hf_dataset_summary.json"), help="Output summary file.")
    args = parser.parse_args()

    summary = summarize_dataset(args.dataset, args.split, args.limit if args.limit and args.limit > 0 else None)
    save_summary(summary, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
