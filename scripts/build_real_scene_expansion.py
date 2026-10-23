from __future__ import annotations

import shutil
from pathlib import Path

import cv2

from ultralytics import YOLO

root = Path("C:/Users/info/OneDrive/Desktop/AI Detection")
input_dir = root / "input"
source_test_frames = input_dir / "test_frames"
video_sources = [
    input_dir / "car.mp4",
    input_dir / "objects.mp4",
    input_dir / "people_vehicles.mp4",
    input_dir / "person.mp4",
    input_dir / "real_demo.mp4",
    input_dir / "test_video.mp4",
]

train_dir = root / "dataset" / "images" / "train_real_expanded"
val_dir = root / "dataset" / "images" / "val_real_expanded"
train_label_dir = root / "dataset" / "labels" / "train_real_expanded"
val_label_dir = root / "dataset" / "labels" / "val_real_expanded"

for folder in [train_dir, val_dir, train_label_dir, val_label_dir]:
    folder.mkdir(parents=True, exist_ok=True)

model_path = root / "runs" / "detect" / "runs" / "train" / "yolo11n_fine_tuned_v1" / "weights" / "best.pt"
if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

model = YOLO(str(model_path))


def copy_existing_labeled_images() -> list[Path]:
    copied: list[Path] = []
    for img_path in sorted(source_test_frames.glob("*.jpg")):
        dst = train_dir / img_path.name
        if not dst.exists():
            shutil.copy2(img_path, dst)
        label_src = root / "dataset" / "labels" / "train_balanced" / (img_path.stem + ".txt")
        if label_src.exists():
            shutil.copy2(label_src, train_label_dir / (img_path.stem + ".txt"))
        copied.append(img_path)
    return copied


def extract_video_frames() -> list[Path]:
    frame_paths: list[Path] = []
    for video_path in video_sources:
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        frame_index = 0
        chosen = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % 25 == 0 and frame is not None:
                save_path = root / "dataset" / "raw" / "real_scene_expansion" / f"{video_path.stem}_{frame_index}.jpg"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), frame)
                frame_paths.append(save_path)
                chosen += 1
            frame_index += 1
        cap.release()
        print(f"Extracted {chosen} frames from {video_path.name}")
    return frame_paths


def write_yolo_label(label_path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    lines = [f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls_id, cx, cy, w, h in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def generate_pseudo_labels() -> None:
    extracted = extract_video_frames()
    if not extracted:
        return

    # Use a conservative set of frames for training and a small validation split.
    for index, frame_path in enumerate(extracted):
        target_dir = val_dir if index % 5 == 0 else train_dir
        target_label_dir = val_label_dir if index % 5 == 0 else train_label_dir

        dst_path = target_dir / frame_path.name
        if not dst_path.exists():
            shutil.copy2(frame_path, dst_path)

        results = model.predict(
            str(frame_path),
            imgsz=1280,
            conf=0.15,
            iou=0.6,
            max_det=40,
            verbose=False,
            device="cpu",
        )

        boxes: list[tuple[int, float, float, float, float]] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in {0, 1, 2, 3, 4}:
                    continue
                conf = float(box.conf[0])
                if conf < 0.15:
                    continue
                x_center, y_center, width, height = box.xywhn[0].tolist()
                boxes.append((cls_id, x_center, y_center, width, height))

        if boxes:
            write_yolo_label(target_label_dir / (dst_path.stem + ".txt"), boxes)
        else:
            # Include the image even if it ends up empty so we can review it later.
            (target_label_dir / (dst_path.stem + ".txt")).write_text("", encoding="utf-8")


copy_existing_labeled_images()
generate_pseudo_labels()

(root / "dataset" / "data_real_scene_expanded.yaml").write_text(
    "train: ./images/train_real_expanded\n"
    "val: ./images/val_real_expanded\n"
    "test: ./images/test\n\n"
    "nc: 5\n"
    "names: ['person', 'car', 'motorcycle', 'bus', 'truck']\n",
    encoding="utf-8",
)

train_count = len(list(train_dir.glob("*.jpg")))
val_count = len(list(val_dir.glob("*.jpg")))
print(f"Expanded real-scene dataset created")
print(f"Train images: {train_count}")
print(f"Val images: {val_count}")
print(f"YAML: {root / 'dataset' / 'data_real_scene_expanded.yaml'}")
