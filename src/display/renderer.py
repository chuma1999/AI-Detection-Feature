"""Rendering utilities for OpenCV video output.

This module keeps the display logic separate from the AI detection engine. The
renderer receives a frame and a list of detection objects, then draws the boxes,
labels, and confidence scores onto the image. This keeps the system easy to
extend later for different output targets, such as a desktop window, a saved
video file, or a future security-monitoring interface.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.detection.models import Detection


class DetectionRenderer:
    """Draw detection results onto a frame.

    The renderer is intentionally a simple presentation layer. It does not do any
    model inference or object filtering; it only takes already-validated
    detections and draws them onto the current frame so a user can see the result.
    """

    def __init__(
        self,
        label_font_scale: float = 0.6,
        label_thickness: int = 2,
        box_thickness: int = 2,
        text_color: tuple[int, int, int] = (0, 255, 0),
        box_color: tuple[int, int, int] = (0, 255, 0),
    ) -> None:
        """Configure the visual style for the rendered annotations.

        Args:
            label_font_scale: Size of the text used for labels.
            label_thickness: Thickness of the label text.
            box_thickness: Thickness of the bounding box lines.
            text_color: OpenCV BGR color for text.
            box_color: OpenCV BGR color for boxes.
        """
        self.label_font_scale = label_font_scale
        self.label_thickness = label_thickness
        self.box_thickness = box_thickness
        self.text_color = text_color
        self.box_color = box_color

    def render(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw all accepted detections and return the updated image.

        Args:
            frame: The current OpenCV video frame.
            detections: List of structured detection objects.

        Returns:
            The frame with bounding boxes and labels drawn onto it.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Cannot render onto an empty frame.")

        rendered_frame = frame.copy()

        for detection in detections:
            box = detection.bounding_box
            label = f"{detection.object_class} {detection.confidence:.2f}"

            # Convert the box coordinates to integer pixel values for OpenCV.
            x1 = int(round(box.x1))
            y1 = int(round(box.y1))
            x2 = int(round(box.x2))
            y2 = int(round(box.y2))

            # Draw the box around the detected object.
            cv2.rectangle(
                rendered_frame,
                (x1, y1),
                (x2, y2),
                self.box_color,
                self.box_thickness,
            )

            # Draw the label and confidence near the top-left of the box.
            cv2.putText(
                rendered_frame,
                label,
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.label_font_scale,
                self.text_color,
                self.label_thickness,
            )

        return rendered_frame

    def render_single(self, frame: np.ndarray, detection: Detection) -> np.ndarray:
        """Draw one detection and return the updated frame.

        This helper is useful for testing or for later UI workflows where a single
        detection may need to be visualized on its own.
        """
        return self.render(frame, [detection])
