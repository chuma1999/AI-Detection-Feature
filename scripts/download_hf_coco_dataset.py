#!/usr/bin/env python3
"""Download a Hugging Face COCO-style object-detection dataset.

This script verifies that the dataset contains the required five classes:
- person
- car
- motorcycle
- bus
- truck

It does not train or evaluate a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

REQUIRED_CLASSES = ["person", "car", "motorcycle", "bus", "truck"]


def analyze_dataset(dataset_name: str, split_name: str, limit: int | None = None) -> dict:
    split = f"{split_name}[:{limit}]" if limit is not None else split_name
    dataset = load_dataset(dataset_name, split=split, cache_dir="dataset/hf_cache")

    class_names = set()
    rows = 0
    for item in dataset:
        rows += 1
        labels = item.get("label", {})
        if isinstance(labels, dict):
            class_names.update(labels.keys())

    required_present = all(name in class_names for name in REQUIRED_CLASSES)

    return {
        "dataset_name": dataset_name,
        "rows": rows,
        "class_names": sorted(class_names)[:20],
        "required_present": required_present,
        "required_missing": [name for name in REQUIRED_CLASSES if name not in class_names],
        "split": split,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and inspect a Hugging Face COCO-style object detection dataset.")
    parser.add_argument("--dataset", default="TheMrguiller/coco-object_detection", help="Hugging Face dataset repo to use.")
    parser.add_argument("--split", default="train", help="Dataset split to download, for example train or validation.")
    parser.add_argument("--limit", type=int, default=100, help="Number of rows to download for quick validation; set to 0 or omit for the full split.")
    parser.add_argument("--output", type=Path, default=Path("dataset/hf_dataset_summary.json"), help="Path for the JSON summary file.")
    args = parser.parse_args()

    limit = args.limit if args.limit and args.limit > 0 else None
    summary = analyze_dataset(args.dataset, args.split, limit=limit)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
