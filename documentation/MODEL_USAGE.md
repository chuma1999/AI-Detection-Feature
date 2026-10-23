# Model Usage Guide

This document explains where the models live, how to load them, and how to run inference in a simple way.

## 1. Original model location

The preserved baseline model is stored here:

- `models/pretrained/yolo11n_original.pt`

This file is the original YOLO11n checkpoint from Ultralytics. It is kept untouched so it can serve as the baseline reference for comparison and future experiments.

## 2. Trained model location

The project-selected trained model is stored here:

- `models/fine_tuned/best.pt`

This is the checkpoint used for the working project application and validation runs.

The project metadata file is:

- `models/fine_tuned/model_config.yaml`

The evaluation summary is:

- `models/fine_tuned/evaluation_results.json`

## 3. How to load the model

Use the Ultralytics Python API:

```python
from ultralytics import YOLO

model = YOLO("models/fine_tuned/best.pt")
```

The model object is then used to run prediction on images or videos.

## 4. How to run inference on a single image

```python
from ultralytics import YOLO
import cv2

model = YOLO("models/fine_tuned/best.pt")
image = cv2.imread("input/test_frames/test-image-1.jpg")

results = model(image, conf=0.12, imgsz=1280, verbose=False)
annotated = results[0].plot()
cv2.imshow("detections", annotated)
cv2.waitKey(0)
```

## 5. How to run inference on a video

```python
from ultralytics import YOLO

model = YOLO("models/fine_tuned/best.pt")

results = model(
    "input/person.mp4",
    conf=0.12,
    imgsz=1280,
    save=True,
    project="output",
    name="project_inference",
    exist_ok=True,
)

print("Finished inference")
```

This saves annotated output frames or a prediction run inside the `output` folder.

## 6. How to run detection with the project app

The project main entry point already wires the model into the detection flow:

```powershell
python main.py --source input/person.mp4 --model models/fine_tuned/best.pt --confidence 0.12
```

This runs the project pipeline that loads the model, reads frames, draws boxes, and logs detections.

## 7. Required dependencies

To use the model correctly, install the project dependencies first:

```powershell
python -m pip install -r requirements.txt
```

The main packages are:

- `ultralytics`
- `opencv-python`
- `numpy`
- `pyyaml`
- `pytest`

## 8. Supported classes

The project model is trained for these classes:

- person
- car
- motorcycle
- bus
- truck

This is configured in the project dataset and model metadata.

## 9. Confidence threshold

A common confidence threshold for this project is:

- `conf=0.12`

This value is suitable for the small, real-world validation scenes used during project testing. A higher threshold can reduce false positives but may also miss distant objects.

## 10. Beginner guide

Think of the model as a trained assistant that looks at an image and predicts:

- what objects are present
- where they are in the image
- how certain the model is about each prediction

The model does not understand scenes the way people do. It works by analyzing patterns in pixels and matching them to what it learned during training.

That is why a model can work well for common objects while still struggling with clutter, occlusion, or distant targets.

## 11. Best practice

- Use the original checkpoint only as a baseline.
- Use `models/fine_tuned/best.pt` for the project workflow.
- Keep inference thresholds moderate when testing on real-world scenes.
- Validate on actual project images and videos rather than relying only on training metrics.

## 12. Summary

The project is designed so the model can be reused without rebuilding the application logic. The model file is independent from the project app, which makes it easy to swap checkpoints, rerun experiments, and compare performance.
