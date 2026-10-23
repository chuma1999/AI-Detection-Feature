#!/usr/bin/env python3
"""Display a few sample images from the prepared YOLO dataset.

This tool is intentionally simple: it reads images and labels, draws bounding
boxes, and saves annotated samples to an output folder. It is useful for checking
that annotation boxes and class names are sensible before training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

CLASS_NAMES = ["person", "car", "motorcycle", "bus", "truck"]


def load_label_file(label_path: Path) -> list[list[float]]:
    if not label_path.exists():
        return []

    entries: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])
        except ValueError:
            continue
        entries.append([class_id, x_center, y_center, box_width, box_height])
    return entries


def xywh_to_xyxy(x_center: float, y_center: float, box_width: float, box_height: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = (x_center - box_width / 2.0) * width
    y1 = (y_center - box_height / 2.0) * height
    x2 = (x_center + box_width / 2.0) * width
    y2 = (y_center + box_height / 2.0) * height
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def draw_annotations(image: cv2.Mat, entries: list[list[float]]) -> None:
    height, width = image.shape[:2]
    for class_id, x_center, y_center, box_width, box_height in entries:
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            continue
        x1, y1, x2, y2 = xywh_to_xyxy(x_center, y_center, box_width, box_height, width, height)
        label = f"{CLASS_NAMES[class_id]}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (max(0, x1), max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def sample_split(dataset_dir: Path, split_name: str, max_images: int = 5) -> list[Path]:
    image_dir = dataset_dir / "images" / split_name
    if not image_dir.exists():
        return []

    image_files = sorted(image_dir.glob("*"))
    files = [p for p in image_files if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    return files[:max_images]


def visualize_split(dataset_dir: Path, split_name: str, output_dir: Path, max_images: int = 5) -> int:
    sample_files = sample_split(dataset_dir, split_name, max_images=max_images)
    if not sample_files:
        print(f"No images found in split: {split_name}")
        return 0

    split_output = output_dir / split_name
    split_output.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_path in sample_files:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Unable to read image: {image_path}")
            continue

        label_path = dataset_dir / "labels" / split_name / (image_path.stem + ".txt")
        entries = load_label_file(label_path)
        draw_annotations(image, entries)

        output_path = split_output / image_path.name
        cv2.imwrite(str(output_path), image)
        print(f"Saved annotated preview: {output_path}")
        count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Render sample annotated images from a YOLO dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"), help="Path to the dataset root folder.")
    parser.add_argument("--max-images", type=int, default=5, help="Maximum sample images per split to visualize.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/dataset_visualization"), help="Folder to save annotated previews.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split_name in ("train", "val", "test"):
        found = visualize_split(dataset_dir, split_name, output_dir, max_images=args.max_images)
        total += found
        print(f"Split {split_name}: {found} preview image(s) created.")

    print(f"Total annotated previews created: {total}")
    print(f"Preview folder: {output_dir}")


if __name__ == "__main__":
    main()
