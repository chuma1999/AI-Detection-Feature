import numpy as np

from src.detection.models import BoundingBox, Detection
from src.display.renderer import DetectionRenderer


def test_renderer_draws_box_and_text() -> None:
    """The renderer should annotate the frame with box and text information."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detector_result = Detection(
        timestamp="00:00:02.450",
        object_class="person",
        confidence=0.91,
        bounding_box=BoundingBox(x1=20, y1=30, x2=120, y2=180),
    )

    renderer = DetectionRenderer()
    annotated = renderer.render(frame, [detector_result])

    assert annotated is not None
    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype
    assert np.any(annotated != 0)


def test_renderer_handles_single_detection() -> None:
    """The helper for a single detection should also produce an annotated frame."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    detection = Detection(
        timestamp="00:00:01.000",
        object_class="car",
        confidence=0.82,
        bounding_box=BoundingBox(x1=10, y1=10, x2=80, y2=90),
    )

    renderer = DetectionRenderer()
    annotated = renderer.render_single(frame, detection)

    assert annotated.shape == frame.shape
    assert np.any(annotated != 0)
