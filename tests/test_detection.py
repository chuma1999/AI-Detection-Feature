from __future__ import annotations

import numpy as np

from src.detection.detector import YOLODetector
from src.detection.models import BoundingBox, Detection


class FakeBox:
    """Minimal box stub that mimics the Ultralytics `box` object."""

    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.conf = [np.array(confidence, dtype=float)]
        self.cls = [np.array(class_id, dtype=float)]


class FakeResult:
    """Minimal result stub that mimics a YOLO prediction result."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.names = {
            0: "person",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
            1: "bicycle",
        }


class FakeYOLO:
    """Fake YOLO model used to test filtering and shape conversion."""

    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, frame, conf=None, verbose=False):
        return [
            FakeResult(
                [
                    FakeBox([10, 20, 120, 220], 0.91, 0),
                    FakeBox([190, 40, 380, 200], 0.88, 2),
                    FakeBox([100, 100, 200, 180], 0.72, 1),
                ]
            )
        ]


def test_detection_data_model_has_required_fields() -> None:
    """The detection object should have the required fields for logging and display."""
    detection = Detection(
        timestamp="00:00:02.450",
        object_class="person",
        confidence=0.91,
        bounding_box=BoundingBox(x1=10, y1=20, x2=120, y2=220),
    )

    assert detection.timestamp == "00:00:02.450"
    assert detection.object_class == "person"
    assert detection.confidence == 0.91
    assert detection.bounding_box.x2 == 120


def test_detector_filters_supported_classes(monkeypatch) -> None:
    """The detector should accept only supported object classes and ignore others."""

    monkeypatch.setattr("src.detection.detector.YOLO", FakeYOLO)

    detector = YOLODetector(model_name="yolo11n.pt", confidence_threshold=0.50)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    detections = detector.detect_frame(frame, timestamp_ms=2450)

    assert len(detections) == 2
    assert {d.object_class for d in detections} == {"person", "car"}
    assert all(d.timestamp == "00:00:02.450" for d in detections)
    assert all(d.confidence >= 0.50 for d in detections)
    assert all(isinstance(d.bounding_box, BoundingBox) for d in detections)


def test_detector_returns_empty_list_when_no_objects_match() -> None:
    """A frame with no accepted objects should return an empty list."""

    class EmptyFakeYOLO:
        def __init__(self, model_name):
            self.model_name = model_name

        def __call__(self, frame, conf=None, verbose=False):
            return [FakeResult([FakeBox([10, 20, 30, 40], 0.20, 1)])]

    import src.detection.detector as detector_module

    original_model = detector_module.YOLO
    detector_module.YOLO = EmptyFakeYOLO
    try:
        detector = YOLODetector(model_name="yolo11n.pt", confidence_threshold=0.50)
        detections = detector.detect_frame(np.zeros((100, 100, 3), dtype=np.uint8), timestamp_ms=3000)
        assert detections == []
    finally:
        detector_module.YOLO = original_model
