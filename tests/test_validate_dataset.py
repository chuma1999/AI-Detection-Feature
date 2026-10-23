from pathlib import Path

from scripts.validate_dataset import validate_dataset


def write_valid_label(path: Path) -> None:
    path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")


def test_validate_dataset_accepts_project_label_layout(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)

        image_path = image_dir / "example.jpg"
        image_path.write_bytes(b"fake jpeg")
        write_valid_label(label_dir / "example.txt")

    summary = validate_dataset(dataset_dir)

    assert summary["train"]["images"] == 1
    assert summary["train"]["missing_labels"] == 0
    assert summary["train"]["invalid_boxes"] == 0
    assert summary["val"]["missing_labels"] == 0
    assert summary["test"]["missing_labels"] == 0
