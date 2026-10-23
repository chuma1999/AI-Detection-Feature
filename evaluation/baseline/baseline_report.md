# Baseline evaluation report

## Model evaluated

- Model: yolo11n_original.pt
- Model source: Ultralytics official pretrained YOLO11n weights
- Model path: models/pretrained/yolo11n_original.pt
- Library: ultralytics 8.4.183
- Dataset: held-out test split only
- Classes evaluated: person, car, motorcycle, bus, truck

## Test-set context

The evaluation used only the held-out test data and not the training data. The test split in this project contains 2 images and 5 labeled instances.

## Baseline metrics reported by Ultralytics

### Overall

- Precision: 0.372
- Recall: 0.515
- mAP50: 0.376

### Per-class

- person: Precision 0.743, Recall 0.781, mAP50 0.527
- car: Precision 0.000, Recall 0.249, mAP50 0.224
- motorcycle: no detections in this small test set
- bus: no detections in this small test set
- truck: no detections in this small test set

## Beginner-friendly explanation

### Precision

Precision answers: “Of all the boxes the model predicted, how many were actually correct?”

If precision is low, the model is making lots of false positives. It is guessing too often.

### Recall

Recall answers: “Of all the actual objects that really exist, how many did the model find?”

If recall is low, the model is missing objects. It is making many false negatives.

### mAP

mAP means mean Average Precision. It combines precision and recall across different confidence thresholds. It is a single summary score that tells us how strong the detector is overall.

A higher mAP means the model is both finding objects and avoiding bad predictions.

### False positive

A false positive is when the model predicts an object that is not actually there. Example: the model marks a box around a background pattern or a random area that is not a person or car.

### False negative

A false negative is when a real object exists but the model misses it. Example: a person is in the image but the model does not draw a box around them.

## Interpretation

This original pretrained YOLO11n model is useful, but it is not yet tuned for this exact project. The overall baseline is moderate, and the person class is the strongest. The limited test set is small, so some classes are not represented strongly enough for a robust final conclusion.

This baseline is the reference point for the upcoming fine-tuning stage. The fine-tuned model should be compared against these numbers.

## Important note

This stage does not change the model or train it. We are only measuring the original checkpoint as a baseline before fine-tuning begins.
