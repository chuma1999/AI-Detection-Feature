#!/usr/bin/env python3
"""Download a small real-image COCO subset for this project and convert it to YOLO format.

This script keeps the training workflow honest:
- it downloads real COCO images from the official source, not just metadata paths
- it filters to the required classes: person, car, motorcycle, bus, truck
- it creates a local YOLO dataset structure and validates the result

It is intended as a dataset-preparation step only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import ssl
import zipfile
from collections import Counter
from pathlib import Path
from urllib.request import HTTPSHandler, build_opener

REQUIRED_CLASSES = ["person", "car", "motorcycle", "bus", "truck"]
CLASS_TO_ID = {
    "person": 0,
    "car": 1,
    "motorcycle": 2,
    "bus": 3,
    "truck": 4,
}
DEFAULT_ANNOTATION_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and zipfile.is_zipfile(destination):
        print(f"Using existing valid archive: {destination}")
        return
    if destination.exists():
        destination.unlink()
    print(f"Downloading {url} -> {destination}")
    context = ssl._create_unverified_context()
    opener = build_opener(HTTPSHandler(context=context))
    with opener.open(url) as response, destination.open("wb") as output_file:
        output_file.write(response.read())


def extract_annotation_zip(zip_path: Path, extract_dir: Path) -> Path:
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"The archive is invalid or incomplete: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    annotation_json = extract_dir / "annotations" / "instances_train2017.json"
    if not annotation_json.exists():
        raise FileNotFoundError(f"Expected annotation JSON not found after extraction: {annotation_json}")
    return annotation_json


def coco_to_yolo_box(x: float, y: float, width: float, height: float, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("Invalid bounding box size.")
    x_center = (x + width / 2.0) / image_width
    y_center = (y + height / 2.0) / image_height
    box_width = width / image_width
    box_height = height / image_height
    return x_center, y_center, box_width, box_height


def write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if lines:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        path.write_text("", encoding="utf-8")


def build_dataset(output_dir: Path, limit: int, seed: int = 42) -> dict:
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    annotation_zip = raw_dir / "annotations_trainval2017.zip"
    download_file(DEFAULT_ANNOTATION_URL, annotation_zip)

    annotation_root = raw_dir / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    annotation_json_path = extract_annotation_zip(annotation_zip, raw_dir)

    with annotation_json_path.open("r", encoding="utf-8") as handle:
        annotations_data = json.load(handle)

    category_map = {category["id"]: category["name"] for category in annotations_data.get("categories", [])}
    image_map: dict[int, dict] = {image["id"]: image for image in annotations_data.get("images", [])}
    annotations_by_image: dict[int, list[dict]] = {image_id: [] for image_id in image_map}
    for annotation in annotations_data.get("annotations", []):
        image_id = annotation.get("image_id")
        if image_id in annotations_by_image:
            annotations_by_image[image_id].append(annotation)

    selected_image_ids: list[int] = []
    for image_id, image_info in image_map.items():
        valid_annotations = []
        for annotation in annotations_by_image.get(image_id, []):
            category_name = category_map.get(annotation.get("category_id"))
            if category_name in CLASS_TO_ID:
                valid_annotations.append(annotation)
        if valid_annotations:
            selected_image_ids.append(image_id)
        if len(selected_image_ids) >= limit:
            break

    if not selected_image_ids:
        raise RuntimeError("No images with the required classes were found in the COCO annotations.")

    random.Random(seed).shuffle(selected_image_ids)
    total = len(selected_image_ids)
    train_end = math.floor(total * 0.8)
    val_end = math.floor(total * 0.9)
    splits = {
        "train": selected_image_ids[:train_end],
        "val": selected_image_ids[train_end:val_end],
        "test": selected_image_ids[val_end:],
    }

    final_dataset_dir = output_dir
    for split_name in ("train", "val", "test"):
        (final_dataset_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (final_dataset_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    class_counts = Counter()
    summary = {"train": 0, "val": 0, "test": 0}

    for split_name, image_ids in splits.items():
        for image_id in image_ids:
            image_info = image_map[image_id]
            file_name = image_info["file_name"]
            image_url = f"https://images.cocodataset.org/train2017/{file_name}"
            local_image_path = final_dataset_dir / "images" / split_name / file_name
            if not local_image_path.exists():
                download_file(image_url, local_image_path)

            label_lines: list[str] = []
            for annotation in annotations_by_image.get(image_id, []):
                category_name = category_map.get(annotation.get("category_id"))
                if category_name not in CLASS_TO_ID:
                    continue

                x, y, width, height = annotation.get("bbox", [0.0, 0.0, 0.0, 0.0])
                image_width = int(image_info.get("width", 0))
                image_height = int(image_info.get("height", 0))
                if image_width <= 0 or image_height <= 0 or width <= 0 or height <= 0:
                    continue

                try:
                    x_center, y_center, box_width, box_height = coco_to_yolo_box(x, y, width, height, image_width, image_height)
                except ValueError:
                    continue

                class_id = CLASS_TO_ID[category_name]
                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
                class_counts[category_name] += 1

            write_label(final_dataset_dir / "labels" / split_name / (Path(file_name).stem + ".txt"), label_lines)
            summary[split_name] += 1

    data_yaml = final_dataset_dir / "data.yaml"
    data_yaml.write_text(
        "train: ./images/train\n"
        "val: ./images/val\n"
        "test: ./images/test\n\n"
        "nc: 5\n"
        "names: ['person', 'car', 'motorcycle', 'bus', 'truck']\n",
        encoding="utf-8",
    )

    return {
        "summary": summary,
        "class_counts": dict(class_counts),
        "total_images": sum(summary.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a real-image COCO subset and build the local YOLO dataset structure.")
    parser.add_argument("--output", type=Path, default=Path("dataset"), help="Dataset output folder")
    parser.add_argument("--limit", type=int, default=120, help="Maximum number of real COCO images to download and prepare")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dataset(args.output, limit=args.limit, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
