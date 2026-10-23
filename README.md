# AI Detection Project

This repository contains a small but complete object-detection project built around YOLO11n, OpenCV, and a filtered COCO-style training dataset.

The project focuses on five classes:

- person
- car
- motorcycle
- bus
- truck

It is designed to demonstrate a practical end-to-end workflow:

- dataset preparation
- training a compact YOLO model
- evaluating the baseline and fine-tuned results
- running inference on project videos and images
- saving and reusing the best model

## Project overview

The project has three main layers:

1. Video or image input
2. YOLO detection model
3. Annotation and logging output

This separation keeps the detection logic reusable and easier to understand.

## Key model files

- Original preset model: `models/pretrained/yolo11n_original.pt`
- Selected project model: `models/fine_tuned/best.pt`
- Training config: `configs/training_config.yaml`
- Dataset config: `dataset/data.yaml`

## Key documentation

- [documentation/DATASET.md](documentation/DATASET.md)
- [documentation/TRAINING.md](documentation/TRAINING.md)
- [documentation/EVALUATION.md](documentation/EVALUATION.md)
- [documentation/MODEL_USAGE.md](documentation/MODEL_USAGE.md)

## Quick start

### 1. Create and activate the environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Run inference with the trained model

```powershell
python main.py --source input/person.mp4 --model models/fine_tuned/best.pt --confidence 0.12
```

### 4. Or load the model directly in Python

```python
from ultralytics import YOLO

model = YOLO("models/fine_tuned/best.pt")
results = model("input/person.mp4", conf=0.12, imgsz=1280)
```

## Model status

The selected working checkpoint is the fine-tuned YOLO11n model saved at:

- `models/fine_tuned/best.pt`

This is the model chosen for the project after comparing it with the original YOLO11n baseline.

## Evaluation summary

The project compares the original model and the fine-tuned model on the same held-out split:

- Original precision: 0.372
- Original recall: 0.515
- Original mAP50: 0.376
- Fine-tuned precision: 0.965
- Fine-tuned recall: 0.809
- Fine-tuned mAP50: 0.886

The fine-tuned model improved overall detection quality and was selected as the project model.

## Beginner guide

If you are new to object detection, the simple idea is:

- the model looks at an image
- it tries to find objects it was trained to recognize
- it predicts a class and draws a box around the object
- it reports a confidence score telling how sure it is

A good model is not just one that predicts lots of objects. It is one that predicts the right objects, in roughly the right place, with a reasonable confidence level.

## Repository structure

```text
AI Detection/
├── configs/
├── dataset/
├── documentation/
├── input/
├── models/
├── output/
├── runs/
├── scripts/
├── src/
├── tests/
├── .gitignore
├── DEVELOPMENT_NOTES.md
├── README.md
├── main.py
├── requirements.txt
└── yolo11n.pt
```

## Notes

This project is a working prototype and an educational implementation. It is useful for understanding dataset preparation, model tuning, evaluation, and inference flow, but it should still be treated as a small-scale project model rather than a large production deployment system.
