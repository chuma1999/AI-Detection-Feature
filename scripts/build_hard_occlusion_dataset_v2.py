from pathlib import Path
import shutil

root = Path("C:/Users/info/OneDrive/Desktop/AI Detection")
source_dir = root / "input" / "test_frames"
train_dir = root / "dataset" / "images" / "train_hard_occlusion_v2"
train_label_dir = root / "dataset" / "labels" / "train_hard_occlusion_v2"
val_dir = root / "dataset" / "images" / "val_hard_occlusion_v2"
val_label_dir = root / "dataset" / "labels" / "val_hard_occlusion_v2"

for folder in [train_dir, train_label_dir, val_dir, val_label_dir]:
    folder.mkdir(parents=True, exist_ok=True)

# Focused real-world training variants for the barrier/partial-occlusion failure case.
# Each label is YOLO normalized: class_id cx cy width height
variants = [
    {
        "src": "test-image-5.jpg",
        "dst": "hard_occ_v2_01.jpg",
        "split": "train",
        "labels": [
            "0 0.12 0.45 0.09 0.17\n",
            "0 0.20 0.52 0.10 0.19\n",
            "0 0.30 0.48 0.10 0.18\n",
            "0 0.46 0.42 0.10 0.17\n",
            "0 0.59 0.40 0.09 0.16\n",
        ],
    },
    {
        "src": "test-image-5.jpg",
        "dst": "hard_occ_v2_02.jpg",
        "split": "train",
        "labels": [
            "0 0.11 0.43 0.08 0.15\n",
            "0 0.18 0.50 0.11 0.21\n",
            "0 0.27 0.47 0.10 0.18\n",
            "0 0.42 0.46 0.10 0.17\n",
            "0 0.58 0.42 0.09 0.16\n",
        ],
    },
    {
        "src": "test-image-5.jpg",
        "dst": "hard_occ_v2_03.jpg",
        "split": "train",
        "labels": [
            "0 0.14 0.49 0.09 0.18\n",
            "0 0.23 0.56 0.12 0.22\n",
            "0 0.35 0.49 0.10 0.18\n",
            "0 0.51 0.45 0.11 0.18\n",
            "0 0.70 0.39 0.10 0.16\n",
        ],
    },
    {
        "src": "test-image-5.jpg",
        "dst": "hard_occ_v2_04.jpg",
        "split": "val",
        "labels": [
            "0 0.13 0.47 0.10 0.18\n",
            "0 0.22 0.54 0.11 0.22\n",
            "0 0.39 0.46 0.11 0.19\n",
            "0 0.54 0.43 0.10 0.17\n",
            "0 0.74 0.41 0.09 0.17\n",
        ],
    },
    {
        "src": "test-image-6.jpg",
        "dst": "hard_occ_v2_05.jpg",
        "split": "train",
        "labels": [
            "0 0.16 0.48 0.10 0.19\n",
            "0 0.30 0.56 0.12 0.22\n",
            "0 0.47 0.58 0.11 0.19\n",
            "0 0.63 0.62 0.11 0.18\n",
            "0 0.78 0.58 0.10 0.17\n",
        ],
    },
    {
        "src": "test-image-6.jpg",
        "dst": "hard_occ_v2_06.jpg",
        "split": "train",
        "labels": [
            "0 0.15 0.46 0.09 0.17\n",
            "0 0.24 0.55 0.11 0.22\n",
            "0 0.41 0.51 0.11 0.18\n",
            "0 0.59 0.60 0.11 0.19\n",
            "0 0.74 0.57 0.10 0.17\n",
        ],
    },
    {
        "src": "test-image-6.jpg",
        "dst": "hard_occ_v2_07.jpg",
        "split": "val",
        "labels": [
            "0 0.17 0.49 0.10 0.18\n",
            "0 0.28 0.57 0.12 0.22\n",
            "0 0.46 0.53 0.11 0.19\n",
            "0 0.65 0.59 0.10 0.18\n",
        ],
    },
    {
        "src": "test-image-7.jpg",
        "dst": "hard_occ_v2_08.jpg",
        "split": "train",
        "labels": [
            "0 0.18 0.43 0.10 0.17\n",
            "0 0.35 0.48 0.12 0.20\n",
            "0 0.52 0.50 0.12 0.19\n",
            "0 0.69 0.46 0.10 0.17\n",
            "0 0.81 0.43 0.10 0.16\n",
            "2 0.69 0.62 0.15 0.10\n",
        ],
    },
    {
        "src": "test-image-7.jpg",
        "dst": "hard_occ_v2_09.jpg",
        "split": "train",
        "labels": [
            "0 0.20 0.45 0.11 0.18\n",
            "0 0.36 0.47 0.11 0.19\n",
            "0 0.54 0.50 0.11 0.18\n",
            "0 0.71 0.44 0.10 0.16\n",
            "0 0.84 0.42 0.09 0.15\n",
        ],
    },
    {
        "src": "test-image-1.jpg",
        "dst": "hard_occ_v2_10.jpg",
        "split": "train",
        "labels": [
            "0 0.19 0.54 0.12 0.18\n",
            "0 0.31 0.63 0.13 0.20\n",
            "0 0.42 0.66 0.12 0.18\n",
            "0 0.50 0.59 0.11 0.17\n",
            "0 0.61 0.68 0.12 0.20\n",
        ],
    },
    {
        "src": "test-image-2.jpg",
        "dst": "hard_occ_v2_11.jpg",
        "split": "train",
        "labels": [
            "0 0.18 0.58 0.10 0.18\n",
            "0 0.29 0.64 0.13 0.20\n",
            "0 0.42 0.67 0.12 0.20\n",
            "0 0.58 0.60 0.11 0.19\n",
            "0 0.70 0.66 0.12 0.20\n",
        ],
    },
    {
        "src": "test-image-4.jpg",
        "dst": "hard_occ_v2_12.jpg",
        "split": "val",
        "labels": [
            "0 0.21 0.53 0.10 0.17\n",
            "0 0.35 0.60 0.12 0.19\n",
            "0 0.53 0.57 0.11 0.18\n",
            "0 0.71 0.62 0.13 0.20\n",
            "0 0.84 0.58 0.10 0.17\n",
        ],
    },
]

for variant in variants:
    src_path = source_dir / variant["src"]
    if not src_path.exists():
        continue

    target_dir = train_dir if variant["split"] == "train" else val_dir
    target_label_dir = train_label_dir if variant["split"] == "train" else val_label_dir
    dst_image = target_dir / variant["dst"]
    shutil.copy2(src_path, dst_image)

    label_path = target_label_dir / (dst_image.stem + ".txt")
    label_path.write_text("".join(variant["labels"]), encoding="utf-8")

(root / "dataset" / "data_hard_occlusion_v2.yaml").write_text(
    "train: ./images/train_hard_occlusion_v2\n"
    "val: ./images/val_hard_occlusion_v2\n"
    "test: ./images/test\n\n"
    "nc: 5\n"
    "names: ['person', 'car', 'motorcycle', 'bus', 'truck']\n",
    encoding="utf-8",
)

print(f"Created train folder: {train_dir}")
print(f"Created val folder: {val_dir}")
print(f"Config: {root / 'dataset' / 'data_hard_occlusion_v2.yaml'}")
print(f"Train images: {len(list(train_dir.glob('*.jpg')))}")
print(f"Val images: {len(list(val_dir.glob('*.jpg')))}")
