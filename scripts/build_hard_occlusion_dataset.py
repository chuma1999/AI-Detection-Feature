from pathlib import Path
import shutil

root = Path("C:/Users/info/OneDrive/Desktop/AI Detection")
source_dir = root / "input" / "test_frames"
train_dir = root / "dataset" / "images" / "train_hard_occlusion"
label_dir = root / "dataset" / "labels" / "train_hard_occlusion"
train_dir.mkdir(parents=True, exist_ok=True)
label_dir.mkdir(parents=True, exist_ok=True)

# Use several copies of the metal-bar scene and vary the occluded-person box positions.
# This creates a small, focused dataset for the specific failure mode: people partially hidden by barriers.
variants = [
    {
        "src": "test-image-5.jpg",
        "dst": "occluded_person_barrier_01.jpg",
        "labels": [
            "0 0.12 0.46 0.10 0.19\n",
            "0 0.19 0.52 0.12 0.25\n",
            "0 0.31 0.48 0.11 0.20\n",
            "0 0.49 0.44 0.11 0.18\n",
            "0 0.60 0.39 0.10 0.17\n",
        ],
    },
    {
        "src": "test-image-5.jpg",
        "dst": "occluded_person_barrier_02.jpg",
        "labels": [
            "0 0.15 0.53 0.12 0.23\n",
            "0 0.22 0.41 0.10 0.18\n",
            "0 0.38 0.46 0.11 0.20\n",
            "0 0.51 0.42 0.12 0.20\n",
            "0 0.69 0.38 0.10 0.17\n",
        ],
    },
    {
        "src": "test-image-1.jpg",
        "dst": "occluded_person_barrier_03.jpg",
        "labels": [
            "0 0.09 0.50 0.10 0.18\n",
            "0 0.18 0.52 0.12 0.24\n",
            "0 0.26 0.48 0.11 0.18\n",
            "0 0.39 0.44 0.10 0.17\n",
        ],
    },
    {
        "src": "test-image-2.jpg",
        "dst": "occluded_person_barrier_04.jpg",
        "labels": [
            "0 0.10 0.48 0.09 0.17\n",
            "0 0.18 0.56 0.12 0.24\n",
            "0 0.27 0.44 0.10 0.18\n",
            "0 0.41 0.47 0.11 0.18\n",
        ],
    },
    {
        "src": "test-image-6.jpg",
        "dst": "occluded_person_barrier_05.jpg",
        "labels": [
            "0 0.13 0.50 0.11 0.22\n",
            "0 0.22 0.58 0.12 0.23\n",
            "0 0.46 0.51 0.12 0.20\n",
            "0 0.61 0.42 0.10 0.17\n",
        ],
    },
    {
        "src": "test-image-6.jpg",
        "dst": "occluded_person_barrier_06.jpg",
        "labels": [
            "0 0.16 0.47 0.10 0.20\n",
            "0 0.24 0.54 0.13 0.25\n",
            "0 0.43 0.47 0.11 0.19\n",
            "0 0.72 0.46 0.10 0.17\n",
        ],
    },
]

for variant in variants:
    src_path = source_dir / variant["src"]
    dst_path = train_dir / variant["dst"]
    if not src_path.exists():
        continue
    shutil.copy2(src_path, dst_path)
    label_path = label_dir / (dst_path.stem + ".txt")
    label_path.write_text("".join(variant["labels"]), encoding="utf-8")

(root / "dataset" / "data_hard_occlusion.yaml").write_text(
    "train: ./images/train_hard_occlusion\n"
    "val: ./images/val\n"
    "test: ./images/test\n\n"
    "nc: 5\n"
    "names: ['person', 'car', 'motorcycle', 'bus', 'truck']\n",
    encoding="utf-8",
)

print(f"Created hard-occlusion dataset at: {train_dir}")
print(f"Label folder: {label_dir}")
print(f"Config: {root / 'dataset' / 'data_hard_occlusion.yaml'}")
print(f"Total examples: {len(list(train_dir.glob('*.jpg')))}")
