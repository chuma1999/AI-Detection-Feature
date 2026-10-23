# Fine-tuned model evaluation report

## Model evaluated

- Model: yolo11n_fine_tuned_v1
- Model path: runs/detect/runs/train/yolo11n_fine_tuned_v1/weights/best.pt
- Base model: models/pretrained/yolo11n_original.pt
- Test dataset: kept exactly the same held-out test split used for the baseline evaluation
- Classes evaluated: person, car, motorcycle, bus, truck

## Evaluation setup

This evaluation used the exact same test data as the baseline run. The dataset was not changed. We only checked how the fine-tuned model performs on that held-out test split.

## Overall metrics reported by Ultralytics

- Precision: 0.965
- Recall: 0.809
- mAP50: 0.886
- mAP50-95: 0.701

## Per-class performance

### person

- Precision: 1.000
- Recall: 0.618
- mAP50: 0.777
- mAP50-95: 0.506
- Ground-truth instances: 4

### motorcycle

- Precision: 0.929
- Recall: 1.000
- mAP50: 0.995
- mAP50-95: 0.895
- Ground-truth instances: 1

### car

- No ground-truth car instances were present in this held-out test split.
- No class-level car metric is available from this evaluation run.

### bus

- No ground-truth bus instances were present in this held-out test split.
- No class-level bus metric is available from this evaluation run.

### truck

- No ground-truth truck instances were present in this held-out test split.
- No class-level truck metric is available from this evaluation run.

## False positives and false negatives

The standard Ultralytics validation output does not produce a single clean total of false positives and false negatives in the same way a classroom example would. In this project, the held-out test set is extremely small:

- 2 images
- 5 total labeled objects

Because the test set is so small, the false-positive and false-negative picture can swing a lot from one image to another. That means these counts are useful for inspection, but they are not stable enough to use as a final judgment of model quality.

For this reason, the project records the official precision, recall, and mAP values as the main summary and treats the test split as a very limited sample, not a final large-scale benchmark.

## Beginner-friendly explanation

### Precision

Precision tells us: “Of all the boxes the model predicted, how many were actually correct?”

A high precision means the model is not guessing too much. It is mostly making correct detections.

### Recall

Recall tells us: “Of all the real objects that really existed, how many did the model find?”

A high recall means the model is finding most of the objects in the image.

### mAP

mAP means mean Average Precision. It is a single summary score that combines both precision and recall across different confidence levels.

A higher mAP usually means the model is doing a better job overall.

### Why the result is still limited

This held-out test set is tiny. It contains only a few objects, and some classes do not appear at all. So even though the numbers look good, they should be treated as a small sample, not as a final proof that the model is excellent.

## Important note

This stage is only for recording the results. We are not claiming that the fine-tuned model is better yet. We are simply measuring it on the exact same test split used for the baseline and saving the results clearly.

## Saved output

- Validation artifacts: [evaluation/fine_tuned](evaluation/fine_tuned)
- Model evaluation images are stored in this folder.
