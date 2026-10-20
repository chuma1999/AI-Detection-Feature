"""Main application orchestration for the AI detection prototype.

This module ties together the individual components in a clean pipeline:

    Video source
        ↓
    Frame
        ↓
    YOLO detector
        ↓
    Detection objects
        ↓
    OpenCV renderer
        ↓
    JSON logger

Each component is responsible only for its own job. The video source reads
frames, the detector interprets them, the renderer displays the results, and the
logger stores the detections in JSON.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config.settings import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_MODEL_NAME
from src.detection.detector import YOLODetector
from src.display.renderer import DetectionRenderer
from src.logging.json_logger import JSONDetectionLogger
from src.video.file_source import VideoFileSource


class DetectionApplication:
    """Run the detection pipeline for a local video source.

    This is the orchestration layer of the prototype. It has no direct AI logic;
    it simply coordinates the source, detector, renderer, and logger.
    """

    def __init__(
        self,
        source_path: str,
        model_name: str = DEFAULT_MODEL_NAME,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        output_path: str = "output/detections.json",
    ) -> None:
        """Initialize the application with its dependencies.

        Args:
            source_path: Path to the local video file.
            model_name: YOLO model name to load.
            confidence_threshold: Minimum confidence score for accepted detections.
            output_path: Destination path for the JSON log file.
        """
        self.source_path = source_path
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.output_path = output_path

        self.video_source: Optional[VideoFileSource] = None
        self.detector: Optional[YOLODetector] = None
        self.renderer: Optional[DetectionRenderer] = None
        self.logger: Optional[JSONDetectionLogger] = None

    def _validate_source(self) -> None:
        """Check that the input video path exists before opening it."""
        if not self.source_path or not Path(self.source_path).exists():
            raise FileNotFoundError(f"Video source not found: {self.source_path}")

    def setup(self) -> None:
        """Create and configure each component used by the pipeline."""
        self._validate_source()

        self.video_source = VideoFileSource(self.source_path)
        self.detector = YOLODetector(
            model_name=self.model_name,
            confidence_threshold=self.confidence_threshold,
        )
        self.renderer = DetectionRenderer()
        self.logger = JSONDetectionLogger(self.output_path)

    def process(self) -> None:
        """Process the entire video and display the results in a window."""
        if self.video_source is None or self.detector is None:
            raise RuntimeError("Application has not been set up. Call setup() first.")

        if self.renderer is None or self.logger is None:
            raise RuntimeError("Renderer or logger is not configured.")

        try:
            self.video_source.open()
        except RuntimeError as exc:
            raise RuntimeError(f"Unable to open video source '{self.source_path}': {exc}") from exc

        detections: list[dict[str, object]] = []

        while True:
            frame, timestamp_ms = self.video_source.read_frame()

            if frame is None:
                break

            try:
                frame_detections = self.detector.detect_frame(frame, timestamp_ms=timestamp_ms)
            except Exception as exc:  # pragma: no cover - depends on runtime model issues
                raise RuntimeError(f"Detection failed on frame at {timestamp_ms} ms: {exc}") from exc

            for detection in frame_detections:
                detections.append(detection.as_dict())

            rendered_frame = self.renderer.render(frame, frame_detections)
            cv2.imshow("AI Detection Prototype", rendered_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("User pressed 'q'. Stopping video playback.")
                break

        self.logger.save(self.source_path, [
            self._restore_detection(item) for item in detections
        ])

        cv2.destroyAllWindows()
        self.video_source.close()

    @staticmethod
    def _restore_detection(item: dict[str, object]) -> object:
        """Convert a saved dictionary back into a detection object.

        This helper is used only for the logger call in the current pipeline. The
        logger expects `Detection` objects, so we reconstruct them from the JSON-
        friendly format before writing the final record.
        """
        from src.detection.models import BoundingBox, Detection

        box = item["bounding_box"]
        return Detection(
            timestamp=str(item["timestamp"]),
            object_class=str(item["object_class"]),
            confidence=float(item["confidence"]),
            bounding_box=BoundingBox(
                x1=float(box["x1"]),
                y1=float(box["y1"]),
                x2=float(box["x2"]),
                y2=float(box["y2"]),
            ),
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the application."""
    parser = argparse.ArgumentParser(description="AI detection prototype for local video files.")
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the input video file, for example: input/test_video.mp4",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="YOLO model name to load. Default: yolo11n.pt",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum confidence threshold for accepted detections.",
    )
    parser.add_argument(
        "--output",
        default="output/detections.json",
        help="Destination path for the JSON detection log.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the application from the command line."""
    args = parse_args()

    app = DetectionApplication(
        source_path=args.source,
        model_name=args.model,
        confidence_threshold=args.confidence,
        output_path=args.output,
    )

    try:
        app.setup()
        app.process()
        print(f"Detection log saved to: {app.output_path}")
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:  # pragma: no cover - safety net for runtime conditions
        print(f"Unexpected error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
