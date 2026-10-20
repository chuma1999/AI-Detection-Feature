import json
from pathlib import Path

from src.detection.models import BoundingBox, Detection
from src.logging.json_logger import JSONDetectionLogger


def test_json_logger_creates_valid_json_file(tmp_path: Path) -> None:
    """The logger should create a valid JSON file with detection records."""
    output_path = tmp_path / "detections.json"
    logger = JSONDetectionLogger(str(output_path))

    detections = [
        Detection(
            timestamp="00:00:02.450",
            object_class="person",
            confidence=0.91,
            bounding_box=BoundingBox(x1=120, y1=80, x2=340, y2=470),
        ),
        Detection(
            timestamp="00:00:04.120",
            object_class="car",
            confidence=0.87,
            bounding_box=BoundingBox(x1=500, y1=210, x2=850, y2=420),
        ),
    ]

    payload = logger.save("input/test_video.mp4", detections)

    assert output_path.exists()
    assert payload["source"] == "input/test_video.mp4"
    assert len(payload["detections"]) == 2

    with output_path.open("r", encoding="utf-8") as file_handle:
        loaded = json.load(file_handle)

    assert loaded == payload
    assert loaded["detections"][0]["object_class"] == "person"
    assert loaded["detections"][1]["bounding_box"]["x2"] == 850


def test_json_logger_handles_multiple_detections() -> None:
    """The logger should support storing more than one detection record."""
    logger = JSONDetectionLogger("output/test_detections.json")

    detections = [
        Detection(
            timestamp="00:00:01.000",
            object_class="person",
            confidence=0.81,
            bounding_box=BoundingBox(x1=10, y1=20, x2=90, y2=120),
        ),
        Detection(
            timestamp="00:00:01.500",
            object_class="truck",
            confidence=0.78,
            bounding_box=BoundingBox(x1=150, y1=60, x2=220, y2=160),
        ),
    ]

    payload = logger.save("input/demo.mp4", detections)

    assert payload["detections"][0]["timestamp"] == "00:00:01.000"
    assert payload["detections"][1]["object_class"] == "truck"
    assert payload["detections"][1]["confidence"] == 0.78
