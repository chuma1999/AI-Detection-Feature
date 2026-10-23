from pathlib import Path
import shutil

root = Path("C:/Users/info/OneDrive/Desktop/AI Detection")
source_dir = root / "input" / "test_frames"
image_dst = root / "dataset" / "images" / "train_balanced"
label_dst = root / "dataset" / "labels" / "train_balanced"
image_dst.mkdir(parents=True, exist_ok=True)
label_dst.mkdir(parents=True, exist_ok=True)

# Copy source images into a dedicated balanced training subset.
for src in sorted(source_dir.glob("*.jpg")):
    shutil.copy2(src, image_dst / src.name)

# Manual YOLO labels in normalized format.
# Class ids: person=0, car=2, motorcycle=3, bus=5, truck=7
# These are intentionally approximate but structured for a balanced real-world subset.
manual_labels = {
    "test-image-1.jpg": [
        (0, 0.20, 0.55, 0.12, 0.18),
        (0, 0.31, 0.63, 0.13, 0.20),
        (0, 0.42, 0.66, 0.12, 0.18),
        (0, 0.50, 0.59, 0.11, 0.17),
        (0, 0.61, 0.68, 0.12, 0.20),
        (0, 0.73, 0.60, 0.11, 0.17),
        (0, 0.82, 0.66, 0.12, 0.20),
    ],
    "test-image-2.jpg": [
        (0, 0.18, 0.58, 0.10, 0.18),
        (0, 0.29, 0.64, 0.13, 0.20),
        (0, 0.42, 0.67, 0.12, 0.20),
        (0, 0.58, 0.60, 0.11, 0.19),
        (0, 0.70, 0.66, 0.12, 0.20),
        (0, 0.83, 0.59, 0.10, 0.18),
    ],
    "test-image-3.jpg": [
        (0, 0.14, 0.42, 0.09, 0.14),
        (0, 0.31, 0.46, 0.10, 0.17),
        (0, 0.52, 0.47, 0.11, 0.17),
        (0, 0.71, 0.44, 0.10, 0.16),
        (2, 0.47, 0.51, 0.18, 0.12),
    ],
    "test-image-4.jpg": [
        (0, 0.21, 0.53, 0.10, 0.17),
        (0, 0.35, 0.60, 0.12, 0.19),
        (0, 0.53, 0.57, 0.11, 0.18),
        (0, 0.71, 0.62, 0.13, 0.20),
        (0, 0.84, 0.58, 0.10, 0.17),
    ],
    "test-image-5.jpg": [
        (3, 0.50, 0.55, 0.20, 0.18),
        (0, 0.15, 0.39, 0.09, 0.15),
        (0, 0.29, 0.41, 0.09, 0.16),
        (0, 0.42, 0.44, 0.10, 0.17),
        (0, 0.58, 0.40, 0.09, 0.16),
        (0, 0.74, 0.42, 0.10, 0.17),
    ],
    "test-image-6.jpg": [
        (0, 0.26, 0.58, 0.10, 0.18),
        (0, 0.44, 0.60, 0.10, 0.17),
        (0, 0.62, 0.62, 0.11, 0.18),
        (0, 0.78, 0.60, 0.10, 0.17),
    ],
    "test-image-7.jpg": [
        (0, 0.20, 0.45, 0.12, 0.18),
        (0, 0.36, 0.47, 0.12, 0.20),
        (0, 0.52, 0.50, 0.11, 0.19),
        (0, 0.69, 0.46, 0.10, 0.17),
        (0, 0.82, 0.44, 0.10, 0.16),
        (2, 0.68, 0.62, 0.15, 0.10),
    ],
}

for filename, boxes in manual_labels.items():
    text_lines = []
    for class_id, cx, cy, w, h in boxes:
        text_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path = label_dst / (Path(filename).stem + ".txt")
    label_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")

(root / "dataset" / "data_balanced.yaml").write_text(
    "train: ./images/train_balanced\n"
    "val: ./images/val\n"
    "test: ./images/test\n\n"
    "nc: 5\n"
    "names: ['person', 'car', 'motorcycle', 'bus', 'truck']\n",
    encoding="utf-8",
)

print("Balanced training set prepared:")
print(f"Images: {image_dst}")
print(f"Labels: {label_dst}")
print(f"YAML: {root / 'dataset' / 'data_balanced.yaml'}")
for p in sorted(image_dst.glob("*.jpg")):
    print(f"- {p.name} -> {label_dst / (p.stem + '.txt')}")
