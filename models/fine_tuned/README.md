# Best project model

## Selected model

The selected project checkpoint is the fine-tuned YOLO11n model saved at `models/fine_tuned/best.pt`.

## Why this model was selected

This model was chosen after comparing it against the original pretrained checkpoint using the same held-out test split. The fine-tuned model improved the overall project metrics substantially:

- Precision: 0.965 vs 0.372
- Recall: 0.809 vs 0.515
- mAP50: 0.886 vs 0.376
- False positives: 17 vs 30

This is a clear overall improvement on the project dataset, even though person AP was slightly lower in one narrow class metric. The full-project conclusion is that the fine-tuned model is the better operational choice for this project.

## Model identity

- Model name: yolo11n_fine_tuned_v1
- Model version: v1
- Base model: models/pretrained/yolo11n_original.pt
- Dataset: dataset/data.yaml
- Date: 2026-10-22
- Library versions: Python 3.13.16, ultralytics 8.4.183, OpenCV 5.0.0, PyYAML 6.0.3

## Known limitations

- The test set is very small.
- Some classes remain underrepresented or absent.
- The model still struggles with distant, dark, crowded, and partially occluded objects.
- The class balance is still limited for bus and truck.

## Important note

This bundle is self-contained and can be loaded independently. It does not integrate with Sentinel.
