# YOLO dataset structure

This folder is reserved for the dataset used in the model training project.

The project is designed to keep the original live detection prototype separate from the model-training workflow.

## Required structure

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml
└── README.md
```

## Class order

The final model must contain exactly these five classes:

0. person
1. car
2. motorcycle
3. bus
4. truck

## Split strategy

We use a deterministic 80/10/10 split for the filtered COCO subset:

- train: 80%
- val: 10%
- test: 10%

This keeps the test set independent from the training process and reduces the risk of data leakage.

## Validation rules

Before training:

- confirm all images have labels
- confirm all labels are valid
- confirm class IDs are in the correct range
- confirm all classes are in the required list
- confirm there are no duplicate or corrupt files

## Script usage

To prepare the dataset from a COCO root folder:

```powershell
python scripts/prepare_coco_dataset.py --dataset-root C:/path/to/coco
```

To validate the generated dataset:

```powershell
python scripts/validate_dataset.py --dataset-dir dataset
```

This stage does not train or evaluate the model.

## Note

The raw COCO archives are downloaded on demand if a full regeneration is required
