# Stage 11 — Error analysis

## Goal

This stage reviews where the model still struggles and explains the most likely reasons without retraining or changing the model.

## What the model is still getting wrong

### 1) False positives

The saved prediction JSON contains several boxes that are obviously not real objects. In the fine-tuned predictions, many boxes are low-confidence and some are very large or stretched across background regions.

Examples from the raw output include:

- very large boxes that cover a wide region of the image
- boxes with extreme coordinates such as x=0 or y=0
- low-confidence detections on background texture or empty space
- predictions that appear to match the same object more than once

These are classic signs of background confusion and unstable bounding-box proposals.

### 2) False negatives

The project test split is extremely small, but the model still misses some real objects. The important pattern is that the missed objects are usually not large, obvious objects. The missed or weak cases are likely:

- small objects far away from the camera
- partially hidden objects
- objects in bad lighting or strong shadows
- objects close to the image edge
- crowded scenes where objects overlap

The validation image clearly shows that some people and objects are still much harder to recognize when they are small or partially blocked.

### 3) Classification confusion

The model is not only failing to detect objects; it is sometimes mixing classes. The project examples include the kind of confusion we wanted to watch for:

- car → truck
- truck → bus
- motorcycle → car

This usually happens when:

- the object shapes are visually similar
- the camera angle changes the object appearance
- the training data does not contain enough similar examples
- the model sees too few examples of some classes

## Likely causes

### A) Dataset size is too small

This is the biggest issue.

The project’s final held-out test set contains only:

- 2 test images
- 5 real labeled objects total

And the training set is also very small when compared to a normal object-detection dataset. That means the model has not learned enough variation in:

- object size
- camera angle
- lighting
- occlusion
- scene background
- object appearance

With so few examples, even a good architecture can overfit or behave inconsistently.

### B) Class imbalance is severe

The ground-truth counts in the test split show a clear imbalance:

- person: 4
- car: 1
- motorcycle: 0
- bus: 0
- truck: 0

That means the model is being judged mainly on person and car examples, while the other classes are effectively not represented in the test split. This makes conclusions for motorcycle, bus, and truck weak.

It also means the training set is likely uneven, which leads to a model that learns some classes much better than others.

### C) Poor or limited annotations can make learning noisy

We saw that some predictions are repeated or low-quality, and some boxes are clearly not matching the object shape well. This can happen when:

- the annotation boxes are not tight enough
- objects are partially hidden and annotations are inconsistent
- classes are mixed or labeled differently on different images
- a class is under-labeled over the dataset

If the annotation quality is inconsistent, the model learns fuzzy boundaries and class rules.

### D) Lighting and contrast affect detection

The example validation image includes outdoor scenes with strong sunlight, shadows, and natural environment variation. These conditions matter:

- dark or shadowed objects are harder to detect
- light reflections can confuse the model
- objects at the edge of the image or in uneven light may disappear

This is a common source of missed detections.

### E) Object size is a major weakness

Small or distant objects are the hardest to detect. In object detection, small objects usually have weaker feature detail. When the object occupies only a tiny part of the frame, the model often misses it or predicts a weak box.

This is especially likely in scenes where the camera is far from the object or the object is partially occluded.

### F) Camera angle and perspective matter

The training and test scenes are not always a clean frontal view. Objects can appear at odd angles, side profiles, or partial views. This makes detection harder, especially for classes like motorcycle, bus, and truck where the shape changes depending on perspective.

### G) Occlusion and crowded scenes reduce accuracy

When objects overlap or partly hide each other, the detector often misses one of them. This is a common pattern in real-world image detection, and it is even worse when the dataset is small.

### H) Similar classes are difficult

The model is especially likely to confuse similar-looking classes such as:

- motorcycle vs person
- car vs truck
- bus vs truck

This happens because the silhouettes and shapes can be similar in some views, especially when the object is small, blurry, or partially visible.

## Representative conclusions

### The strongest pattern

The model does not fail randomly. It fails in structured ways:

- background confusion
- small and distant objects missed
- class confusion between similar object types
- weak performance when the dataset has few examples for a class

### Most likely root causes

The most likely causes are:

1. very small dataset size
2. class imbalance
3. weak representation of some classes
4. inconsistent object scale and lighting
5. insufficient examples for difficult angles and occlusions

## Final interpretation

The model is not “bad everywhere,” but it is still weak in exactly the areas that are usually hardest for small object-detection datasets:

- distant or tiny objects
- visually similar classes
- underrepresented classes
- poor lighting
- occluded objects
- small, limited test data

At this point, the issue is not a single bug in the code. It is mainly a data-quality and dataset-coverage problem.

## Important note

This analysis is meant to explain the weaknesses and likely causes only. No retraining is being done here, and no model changes are being made.
