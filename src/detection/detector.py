"""YOLO-based object detection component.

This module is intentionally independent from the video source. The detector does
not care whether frames come from a local file, a camera feed, or a future RTSP
stream. It simply accepts a frame, runs the model, filters supported classes, and
returns structured detection objects.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from src.config.settings import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_NAME,
    MODEL_CLASS_IDS,
    SUPPORTED_CLASSES,
)
from src.detection.models import BoundingBox, Detection
from src.video.base import VideoSource


class YOLODetector:
    """Run a pretrained YOLO model on individual frames.

    The detector receives a frame and a timestamp, then returns a list of
    structured detection objects that can be rendered on screen or saved as JSON.
    This separation is important: the detection logic stays reusable even when the
    input source changes from a file to a live CCTV stream.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        supported_classes: set[str] | None = None,
    ) -> None:
        """Create the detector and load the YOLO model.

        Args:
            model_name: Name of the pretrained YOLO model to load.
            confidence_threshold: Minimum confidence score for accepted objects.
            supported_classes: Optional override for the list of accepted classes.
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.supported_classes = supported_classes or set(SUPPORTED_CLASSES)
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        """Load the lightweight pretrained YOLO model.

        This uses the official Ultralytics YOLO package. If the model file is not
        already present, the package will download it on first use.
        """
        try:
            return YOLO(self.model_name)
        except Exception as exc:  # pragma: no cover - depends on external model download
            raise RuntimeError(
                f"Unable to load YOLO model '{self.model_name}'. "
                "Check the model name and ensure the environment can download or access it."
            ) from exc

    def detect_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float = 0.0,
    ) -> list[Detection]:
        """Process one frame and return structured detections.

        Args:
            frame: The image frame returned by OpenCV.
            timestamp_ms: The source timeline position in milliseconds.

        Returns:
            A list of accepted detections for this frame.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Cannot detect objects in an empty frame.")

        formatted_timestamp = VideoSource.format_timestamp(timestamp_ms)
        detections: list[Detection] = []

        # Ultralytics returns one result object per frame. We inspect each result,
        # then each box within the result to find objects that match our accepted
        # classes and confidence threshold.
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        for result in results:
            if getattr(result, "boxes", None) is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = result.names.get(class_id, str(class_id))

                # Ignore unsupported objects, such as bicycles or traffic signs,
                # if they are not part of this prototype's supported classes.
                if class_name not in self.supported_classes:
                    continue

                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detection = Detection(
                    timestamp=formatted_timestamp,
                    object_class=class_name,
                    confidence=round(confidence, 4),
                    bounding_box=BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    ),
                )
                detections.append(detection)

        return detections

    @staticmethod
    def supported_class_ids(class_names: set[str] | None = None) -> set[int]:
        """Return the YOLO numeric IDs that correspond to the configured classes."""
        names = class_names or SUPPORTED_CLASSES
        return {
            MODEL_CLASS_IDS[name]
            for name in names
            if name in MODEL_CLASS_IDS
        }
