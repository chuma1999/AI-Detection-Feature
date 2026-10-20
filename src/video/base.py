"""Base interfaces for video sources.

This module defines the abstraction used by the AI detection pipeline.
The important idea is that the detector works with frames, not with a specific
video source. That keeps the design flexible for later CCTV or RTSP streams.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class VideoSource(ABC):
    """Abstract base class for all video sources.

    A video source is responsible for providing frames and timestamps to the
    rest of the system. The detector should depend on this abstraction instead of
    a concrete file path or live camera protocol.

    This keeps the design easy to extend later to:

    - a local video file
    - an RTSP stream
    - a CCTV stream
    - another future frame provider

    The detector receives frames and timestamps, but does not need to know where
    the frames came from.
    """

    def __init__(self, source: str):
        """Store the source identifier for the concrete video backend.

        Args:
            source: A file path, stream URL, or other source identifier.
        """
        self.source = source
        self.is_open = False

    @abstractmethod
    def open(self) -> bool:
        """Open the video source and return True if it is ready.

        Returns:
            True when the source has been successfully opened.
        """

    @abstractmethod
    def read_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Read the next frame and the corresponding timestamp.

        Returns:
            A tuple of (frame, timestamp_seconds). The frame is None when the
            end of the video has been reached.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the underlying video resources."""

    @staticmethod
    def format_timestamp(timestamp_ms: float) -> str:
        """Convert milliseconds into a readable HH:MM:SS.mmm format.

        This keeps detection timestamps easy to read and consistent with the
        requirement that timestamps reflect the actual position within the video.
        """
        total_seconds = max(timestamp_ms / 1000.0, 0.0)
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
