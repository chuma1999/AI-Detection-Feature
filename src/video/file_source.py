"""Concrete local-video implementation.

The file-based source is the first supported input for the prototype. It uses
OpenCV to open a local MP4 or other supported video file and read frames one by
one. The design is intentionally separated from the detector so a future CCTV or
RTSP source can replace this class without changing the detection logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.video.base import VideoSource


class VideoFileSource(VideoSource):
    """Read a local video file frame by frame.

    The class wraps OpenCV's VideoCapture and exposes a simple interface that
    returns a frame together with its playback timestamp.

    This is the initial implementation for local files. In the future, another
    class such as RTSPVideoSource could implement the same interface while using
    a different input mechanism.
    """

    def __init__(self, path: str):
        """Create a local video source.

        Args:
            path: Path to the video file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the input path is empty.
        """
        if not path or not str(path).strip():
            raise ValueError("Video file path cannot be empty.")

        video_path = Path(path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        super().__init__(str(video_path))
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open the local video using OpenCV.

        Returns:
            True if the video opened successfully.

        Raises:
            RuntimeError: If the file cannot be opened as a video.
        """
        if self.is_open and self._capture is not None:
            return True

        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video file: {self.source}")

        self._capture = capture
        self.is_open = True
        return True

    def read_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Read the next frame and the video timestamp.

        Returns:
            A tuple containing:
              - frame: the current OpenCV image, or None when the video ends
              - timestamp_ms: video playback time in milliseconds

        The timestamp is taken from the video's own timeline, not from the system
        clock. This is important because later the same logic must work with CCTV
        or RTSP streams as well.
        """
        if self._capture is None:
            self.open()

        if self._capture is None:
            raise RuntimeError("Video capture is not initialized.")

        success, frame = self._capture.read()
        if not success or frame is None:
            return None, float(self._capture.get(cv2.CAP_PROP_POS_MSEC))

        timestamp_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
        return frame, timestamp_ms

    def close(self) -> None:
        """Release the OpenCV video capture and clean up resources."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self.is_open = False

    def __enter__(self) -> "VideoFileSource":
        """Support context-manager usage for short-lived video processing."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Ensure resources are always released when leaving a context."""
        self.close()
