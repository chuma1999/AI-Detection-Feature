from pathlib import Path

import cv2
import numpy as np
import pytest

from src.video.base import VideoSource
from src.video.file_source import VideoFileSource


def _create_test_video(path: Path, frames: int = 5, width: int = 320, height: int = 240) -> Path:
    """Create a tiny MP4 file for testing the video source."""
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (width, height),
    )
    assert writer.isOpened(), "Unable to create a temporary test video."

    for index in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (
            (index * 30) % 255,
            (index * 60) % 255,
            (index * 90) % 255,
        )
        writer.write(frame)

    writer.release()
    return path


def test_video_source_is_abstract() -> None:
    """The base class should not be directly instantiated."""
    with pytest.raises(TypeError):
        VideoSource()


def test_video_file_source_opens_and_reads_frames(tmp_path: Path) -> None:
    """A valid video file should be opened and frames should be readable."""
    video_path = _create_test_video(tmp_path / "sample.mp4")

    source = VideoFileSource(str(video_path))
    assert source.open() is True

    frame, timestamp = source.read_frame()
    assert frame is not None
    assert frame.shape[0] > 0
    assert frame.shape[1] > 0
    assert timestamp >= 0.0

    source.close()


def test_video_file_source_raises_for_missing_file() -> None:
    """An invalid video path should raise a useful error."""
    with pytest.raises(FileNotFoundError):
        VideoFileSource("input/does_not_exist.mp4")


def test_video_file_source_raises_for_invalid_content(tmp_path: Path) -> None:
    """A file that is not a valid video should fail clearly when opened."""
    invalid_file = tmp_path / "not_a_video.txt"
    invalid_file.write_text("this is not a video", encoding="utf-8")

    source = VideoFileSource(str(invalid_file))
    with pytest.raises(RuntimeError):
        source.open()
