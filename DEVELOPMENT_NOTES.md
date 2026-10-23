# Development Notes

## 1. Project goal

This project was built as a practical object-detection prototype for a small set of classes:

- person
- car
- motorcycle
- bus
- truck

The goal is to show a working, understandable pipeline for data preparation, model training, evaluation, and inference using YOLO11n.

## 2. Architecture

The application follows a simple structure:

```text
video/image source -> frame processing -> YOLO model -> detections -> display/logging
```

This makes the system easier to understand and adapt. The detector is separated from the source input so it can work with different kinds of video or image feeds.

## 3. Dataset work

The dataset process is one of the most important parts of the project.

The project uses a COCO-style dataset filtered to the classes that matter for this project. The raw data is converted to YOLO format and then split into train, validation, and test sets.

This matters because the model can only learn as well as the training examples it receives.

## 4. Training approach

The model starts from the official pretrained YOLO11n checkpoint and is then fine-tuned with a small but focused dataset.

The project training configuration uses a conservative setup:

- `epochs: 25`
- `imgsz: 640`
- `batch: 2`
- `lr0: 0.001`
- `device: cpu`

This was chosen because the dataset is small and the environment is limited.

## 5. Evaluation mindset

The project treats evaluation as a real decision-making step, not just a generic report.

The model is compared against the original YOLO11n baseline on the same test split. The key metrics are:

- precision
- recall
- mAP50
- false positives
- false negatives

This is important because a model can look good in casual use but still be weak in objective benchmark terms.

## 6. Main issues encountered

The project encountered several realistic problems:

- small dataset size
- class imbalance
- weak evidence for some classes
- difficult real-world scenes
- distant or occluded objects
- CPU-only training constraints

These issues are common in compact prototype projects and should not be treated as a failure. Instead, they are part of the honest engineering process.

## 7. Why the final model was selected

The selected model is the fine-tuned checkpoint stored at:

- `models/fine_tuned/best.pt`

This model was chosen because it had the best overall evaluation on the project split, especially in precision and mAP50.

## 8. Beginner guide

A beginner-friendly way to understand the project is:

- The model needs examples to learn.
- The dataset is teaching material.
- Training adjusts the model so it can better recognize objects in similar situations.
- Evaluation checks whether the model is actually improving and whether it is making useful predictions.

This is why dataset quality, model choice, and evaluation all matter so much.

## 9. Current project status

The project is in the documentation and validation stage. The selected model, dataset configuration, training setup, and evaluation summary are now documented and saved in the repository.

The key reference files are:

- `README.md`
- `documentation/DATASET.md`
- `documentation/TRAINING.md`
- `documentation/EVALUATION.md`
- `documentation/MODEL_USAGE.md`

## 10. Final note

This project is a practical prototype. It is useful for learning how object detection works in a real project, but it should be treated as a small, usable model rather than a production-ready security platform.
