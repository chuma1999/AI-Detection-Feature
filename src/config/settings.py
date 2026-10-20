"""Configuration values for the AI detection prototype.

This module keeps the object classes and model configuration in one place so the
system is easy to change later. The goal is to make it simple to add more
supported classes or adjust the confidence threshold without rewriting the
processing logic.
"""

from __future__ import annotations

# Supported object classes for this prototype.
# These are the classes that the detector will accept and log.
SUPPORTED_CLASSES: set[str] = {"person", "car", "motorcycle", "bus", "truck"}

# Lightweight YOLO model used by this prototype.
# The model name is intentionally simple so it can be changed in one place.
DEFAULT_MODEL_NAME: str = "yolo11n.pt"

# Lower confidence values are filtered out before we keep a detection.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25

# YOLO class identifiers for the COCO model used by Ultralytics.
# Mapping the class name to its ID makes the filtering logic easy to understand.
MODEL_CLASS_IDS: dict[str, int] = {
    "person": 0,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}
