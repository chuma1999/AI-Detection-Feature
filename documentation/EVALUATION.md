# Evaluation Report

This project evaluates the original pretrained YOLO11n model against the selected fine-tuned project model on the same held-out evaluation split.

## 1. Baseline model

- Original model: `models/pretrained/yolo11n_original.pt`
- Model type: YOLO11n pretrained from Ultralytics
- Baseline results:
  - Precision: 0.372
  - Recall: 0.515
  - mAP50: 0.376
  - False positives: 30
  - False negatives: 5

## 2. Fine-tuned model

- Selected project model: `models/fine_tuned/best.pt`
- Fine-tuned results:
  - Precision: 0.965
  - Recall: 0.809
  - mAP50: 0.886
  - False positives: 17
  - False negatives: 5

## 3. Comparison

The fine-tuned model improved the project metrics on the same evaluation split:

- Precision: 0.372 -> 0.965
- Recall: 0.515 -> 0.809
- mAP50: 0.376 -> 0.886
- False positives: 30 -> 17
- False negatives: 5 -> 5

This is a clear improvement in overall predictive quality, especially in precision and false-positive reduction.

## 4. Per-class results

The project reports the following class-level evidence:

- person: AP = 0.506
- motorcycle: AP = 0.995
- car: not reliably reported on this split
- bus: not reliably reported on this split
- truck: not reliably reported on this split

This is a limitation of the tiny evaluation set. The project has insufficient examples for bus, truck, and several car examples to make a strong class-by-class claim.

## 5. Precision, recall, and mAP

### Precision

Precision measures how often the model is correct when it predicts an object. The fine-tuned model is much more precise than the original model.

### Recall

Recall measures how many real objects the model successfully finds. The fine-tuned model finds more of the real objects in this benchmark set than the original model.

### mAP

mAP50 is the mean average precision at an IoU threshold of 0.5. It is a common summary metric for object detection.

The fine-tuned model achieved:

- mAP50 = 0.886

This is a major lift over the baseline result of 0.376.

## 6. False positives and false negatives

### False positives

The fine-tuned model reduced false positives from 30 to 17. This matters because false alarms can be more damaging in a real detection system than a missed detection in some settings.

### False negatives

False negatives stayed at 5 on this small evaluation split. That means the model still misses some real objects, especially in difficult scenes.

## 7. Weaknesses and limitations

The evaluation is useful, but it is not a final production benchmark.

Main weaknesses include:

- Very small held-out test split
- Unclear evidence for some classes
- Limited bus/truck examples
- Some classes are absent from the evaluation set
- Real-world scenes remain harder than training examples

In other words, the project has strong overall evidence on this sample, but not enough data to say that all classes are equally robust.

## 8. Beginner guide

Object-detection evaluation is a way to answer: “Does the model do a good job?”

- Precision asks: “When the model says it found something, is it usually right?”
- Recall asks: “Did the model find most of the real things that were there?”
- mAP asks: “How strong was the model across the whole detection task?”

A model can improve overall metrics while still struggling on a few difficult classes. That is exactly what this project observed.

## 9. Final assessment

The selected fine-tuned model is better than the original pretrained checkpoint on the project evaluation split. It is the better choice for the project workflow and is the model currently saved as the working project checkpoint.

However, the evaluation setup remains limited. The project should still be treated as a strong prototype system, not a completed large-scale production detector.

## 10. Related files

- `models/fine_tuned/evaluation_results.json`
- `evaluation/fine_tuned/fine_tuned_report.md`
- `models/fine_tuned/model_config.yaml`
- `dataset/data.yaml`
