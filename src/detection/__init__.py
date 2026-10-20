"""Object detection engine and data models."""

from src.detection.detector import YOLODetector
from src.detection.models import BoundingBox, Detection

__all__ = ["YOLODetector", "Detection", "BoundingBox"]
