"""JSON logger for storing detection records.

This logger is intentionally independent from the detector and the video source.
It receives already-created detection objects and writes them to a structured JSON
file. This keeps the logging responsibility separate and makes the result easy to
read later in a web service, dashboard, or future Security & Intrusion module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.detection.models import Detection


class JSONDetectionLogger:
    """Save accepted detection records to JSON.

    The logger writes structured information into output/detections.json. Each
    record contains:
      - timestamp
      - object_class
      - confidence
      - bounding_box coordinates
    """

    def __init__(self, output_path: str = "output/detections.json") -> None:
        """Create the logger with a path to the output JSON file.

        Args:
            output_path: Destination path for the detection log.
        """
        self.output_path = Path(output_path)

    def save(self, source: str, detections: list[Detection]) -> dict[str, Any]:
        """Write detections to JSON and return the payload.

        Args:
            source: The video source path or identifier.
            detections: List of detection objects to log.

        Returns:
            The JSON dictionary payload that was written.
        """
        payload: dict[str, Any] = {
            "source": source,
            "detections": [detection.as_dict() for detection in detections],
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with self.output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)
            file_handle.write("\n")

        return payload
