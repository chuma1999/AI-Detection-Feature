# Training Guide

This project trains a compact YOLO11n object-detection model for five classes:

- person
- car
- motorcycle
- bus
- truck

The training is intentionally small and conservative because the project dataset is limited and the environment is CPU-based.

## 1. Starting model

The project starts from the official Ultralytics pretrained YOLO11n checkpoint:

- Original model: `models/pretrained/yolo11n_original.pt`
- This file is kept as the clean baseline and is not overwritten.

The project does not train from scratch. It begins from the pretrained model and performs fine-tuning for this specific class set.

## 2. Training configuration

The main training settings are stored in `configs/training_config.yaml` and match the actual training run recorded in the project.

### Final project training settings

- Model: `yolo11n`
- Base checkpoint: `models/pretrained/yolo11n_original.pt`
- Dataset: `dataset/data.yaml`
- Epochs: 25
- Image size: 640
- Batch size: 2
- Learning rate: 0.001
- Optimizer: AdamW
- Weight decay: 0.0005
- Momentum: 0.937
- Warmup epochs: 3
- Device: CPU
- Seed: 42

This configuration was designed to reduce instability on a small dataset while still allowing the model to adapt to the target classes.

## 3. Training observations

The selected fine-tuned run is:

- `runs/detect/runs/train/yolo11n_fine_tuned_v1`
- Saved checkpoint: `models/fine_tuned/best.pt`

Important observations from the training run:

- The model improved strongly in overall precision and mAP compared with the original checkpoint.
- Precision improved from 0.372 to 0.965 on the project evaluation split.
- Recall improved from 0.515 to 0.809.
- The model became more selective and produced fewer random false positives.
- The small dataset meant the model remained sensitive to class imbalance and limited sample diversity.

## 4. Problems encountered during training

Several issues affected the project:

### Small dataset size

The prepared project dataset is very small. This makes training unstable and makes the validation metrics less reliable than a large production dataset.

### Class imbalance

The dataset is not balanced across all classes. Some classes such as bus and truck are underrepresented or absent in the held-out test set.

### Real-world mismatch

The project uses a subset of COCO plus a few real-scene examples, but this is still not a full CCTV dataset. Some common conditions remain hard for the model, including:

- very distant objects
- partially occluded people
- crowded scenes
- low-contrast scenes
- small vehicles or people in the background

### CPU training limitations

The model was trained on CPU rather than GPU. This means training is slower and can be less stable than a GPU-backed run.

## 5. Why the project used a conservative setup

The training configuration was intentionally low-risk:

- low batch size to avoid unstable gradients
- moderate learning rate
- light augmentation only
- early stopping support via patience
- CPU-friendly settings

This was a practical choice for a student project and a limited dataset.

## 6. Beginner guide

A simple way to think about training is this:

1. The model starts with general knowledge about objects.
2. The project gives it examples of people, cars, motorcycles, buses, and trucks.
3. The model adjusts its internal weights so it can better recognize those objects in the project context.
4. Training is not about memorizing one image. It is about learning patterns that generalize to similar scenes.

In other words, the model is not being taught to recognize a single scene. It is learning the visual features that make a person, a car, or a motorbike look like those objects.

## 7. Training summary

This project used a careful fine-tuning process to adapt a general object detector to a small, class-specific problem. The final model is useful for the project goals, but it should be treated as a practical prototype model with known limitations rather than a final production-grade security detector.

## 8. Related files

- `configs/training_config.yaml`
- `runs/detect/runs/train/yolo11n_fine_tuned_v1/args.yaml`
- `models/fine_tuned/model_config.yaml`
- `models/fine_tuned/evaluation_results.json`
