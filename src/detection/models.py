"""Reusable data structures for detection results.

These model objects keep the detector output clean and easy to reuse by later
components such as a JSON logger, a REST API layer, or a Security & Intrusion
module. The important idea is that the AI model produces raw results, while this
module defines the structured result that the rest of the system consumes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """A bounding box defined by the rectangle corners.

    The coordinates use the OpenCV convention:
      x1, y1 = upper-left corner
      x2, y2 = lower-right corner
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def as_dict(self) -> dict[str, float]:
        """Convert the box into a JSON-friendly dictionary."""
        return {
            "x1": float(self.x1),
            "y1": float(self.y1),
            "x2": float(self.x2),
            "y2": float(self.y2),
        }


@dataclass(frozen=True)
class Detection:
    """A single accepted object detection.

    The detector produces this structure after processing a frame. This keeps the
    detection logic clean and reusable by the display layer, JSON logger, or any
    future service that needs detection results.
    """

    timestamp: str
    object_class: str
    confidence: float
    bounding_box: BoundingBox

    def as_dict(self) -> dict[str, object]:
        """Convert the detection into a JSON-friendly structure."""
        return {
            "timestamp": self.timestamp,
            "object_class": self.object_class,
            "confidence": float(self.confidence),
            "bounding_box": self.bounding_box.as_dict(),
        }
