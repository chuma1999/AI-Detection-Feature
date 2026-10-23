#!/usr/bin/env python3
"""Prepare a COCO subset for YOLO training.

This script keeps the current app untouched and prepares a YOLO-compatible
training dataset from the approved COCO dataset. It filters the dataset down to
just the five required classes and creates a clean train/val/test structure.

It does not train or evaluate a model.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

REQUIRED_CLASSES = ["person", "car", "motorcycle", "bus", "truck"]
CLASS_TO_ID = {
    "person": 0,
    "car": 1,
    "motorcycle": 2,
    "bus": 3,
    "truck": 4,
}


def load_json(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def build_category_map(categories: list[dict]) -> dict[int, str]:
    return {category["id"]: category["name"] for category in categories}


def is_required_class(category_name: str) -> bool:
    return category_name in CLASS_TO_ID


def coco_to_yolo_box(x: float, y: float, width: float, height: float, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("Invalid bounding box size.")

    x_center = (x + width / 2.0) / image_width
    y_center = (y + height / 2.0) / image_height
    box_width = width / image_width
    box_height = height / image_height

    return x_center, y_center, box_width, box_height


def write_yolo_label(label_path: Path, content: str) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(content, encoding="utf-8")


def get_image_files(root: Path) -> list[Path]:
    return sorted(root.glob("**/*"))


def build_dataset_split(image_ids: list[str], train_ratio: float, val_ratio: float, seed: int) -> tuple[list[str], list[str], list[str]]:
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0):
        raise ValueError("Train and validation ratios must be between 0 and 1.")

    if round(train_ratio + val_ratio, 4) >= 1.0:
        raise ValueError("Train + validation ratios must be less than 1.0.")

    rng = random.Random(seed)
    shuffled = image_ids[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_ids = shuffled[:train_end]
    val_ids = shuffled[train_end:val_end]
    test_ids = shuffled[val_end:]

    return train_ids, val_ids, test_ids


def prepare_dataset(dataset_root: Path, output_dir: Path, seed: int = 42) -> dict[str, int]:
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()

    annotations_root = dataset_root / "annotations"
    if not annotations_root.exists():
        raise FileNotFoundError(f"COCO annotations folder not found: {annotations_root}")

    train_json = annotations_root / "instances_train2017.json"
    if not train_json.exists():
        raise FileNotFoundError(f"Training annotation file not found: {train_json}")

    categories_data = load_json(train_json)
    category_map = build_category_map(categories_data.get("categories", []))
    valid_ids = {category_id for category_id, category_name in category_map.items() if is_required_class(category_name)}

    if not valid_ids:
        raise ValueError("No allowed classes found in the COCO category list.")

    images_dir = dataset_root / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"COCO images folder not found: {images_dir}")

    train_images_dir = images_dir / "train2017"
    if not train_images_dir.exists():
        raise FileNotFoundError(f"COCO train image folder not found: {train_images_dir}")

    images_by_id: dict[int, dict] = {}
    for image in categories_data.get("images", []):
        images_by_id[image["id"]] = image

    annotations_by_image: dict[int, list[dict]] = {image_id: [] for image_id in images_by_id}
    for annotation in categories_data.get("annotations", []):
        image_id = annotation.get("image_id")
        if image_id in annotations_by_image:
            annotations_by_image[image_id].append(annotation)

    valid_image_ids: list[str] = []
    valid_image_lookup: dict[str, Path] = {}

    for image_id, image_info in images_by_id.items():
        file_name = image_info.get("file_name")
        if not file_name:
            continue

        image_path = train_images_dir / file_name
        if not image_path.exists():
            continue

        valid_annotations = []
        for annotation in annotations_by_image.get(image_id, []):
            category_id = annotation.get("category_id")
            category_name = category_map.get(category_id)
            if category_name in CLASS_TO_ID:
                valid_annotations.append(annotation)

        if valid_annotations:
            valid_image_ids.append(str(image_id))
            valid_image_lookup[str(image_id)] = image_path

    if not valid_image_ids:
        raise ValueError("No valid training images were found after filtering classes.")

    train_ids, val_ids, test_ids = build_dataset_split(valid_image_ids, train_ratio=0.80, val_ratio=0.10, seed=seed)

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    summary_counts = Counter()
    label_counts = Counter()

    for split_name, split_ids in splits.items():
        for image_id in split_ids:
            image_path = valid_image_lookup[image_id]
            destination_image = output_dir / "images" / split_name / image_path.name
            destination_image.write_bytes(image_path.read_bytes())

            label_path = output_dir / "labels" / split_name / (image_path.stem + ".txt")
            label_lines: list[str] = []

            for annotation in annotations_by_image.get(int(image_id), []):
                category_id = annotation.get("category_id")
                category_name = category_map.get(category_id)
                if category_name not in CLASS_TO_ID:
                    continue

                x, y, width, height = annotation.get("bbox", [0.0, 0.0, 0.0, 0.0])
                if width <= 0 or height <= 0:
                    continue

                image_info = images_by_id.get(int(image_id), {})
                image_width = int(image_info.get("width", 0))
                image_height = int(image_info.get("height", 0))
                if image_width <= 0 or image_height <= 0:
                    continue

                try:
                    x_center, y_center, box_width, box_height = coco_to_yolo_box(x, y, width, height, image_width, image_height)
                except ValueError:
                    continue

                label_class = CLASS_TO_ID[category_name]
                label_lines.append(f"{label_class} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
                label_counts[category_name] += 1

            if label_lines:
                write_yolo_label(label_path, "\n".join(label_lines) + "\n")
            summary_counts[split_name] += 1

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "train: ./images/train\n"
        "val: ./images/val\n"
        "test: ./images/test\n\n"
        "nc: 5\n"
        "names: ['person', 'car', 'motorcycle', 'bus', 'truck']\n",
        encoding="utf-8",
    )

    summary = {
        "train_images": summary_counts["train"],
        "val_images": summary_counts["val"],
        "test_images": summary_counts["test"],
        "class_counts": dict(label_counts),
    }

    print("Dataset preparation summary")
    print(f"Training images: {summary['train_images']}")
    print(f"Validation images: {summary['val_images']}")
    print(f"Test images: {summary['test_images']}")
    print("Class counts:")
    for class_name in REQUIRED_CLASSES:
        print(f"  {class_name}: {label_counts.get(class_name, 0)}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a filtered COCO dataset for YOLO training.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to the COCO dataset root folder.")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"), help="Output folder for the prepared YOLO dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for the train/val/test split.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(args.dataset_root, args.output_dir, seed=args.seed)
