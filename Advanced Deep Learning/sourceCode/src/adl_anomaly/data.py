from __future__ import annotations

"""Dataset discovery helpers for the Spacepresso folder layout.

The challenge data is organized by class and split:

    class_XX/train/good
    class_XX/train/anomaly_YY
    class_XX/ground_truth_train/anomaly_YY
    class_XX/test

The training code works with small `ImageRecord` objects instead of passing raw
paths around. This keeps the class name, split, view id, and optional mask path
together whenever a sample moves through a dataloader.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
# Matches multiview naming conventions (e.g., "sample123_view2")
VIEW_RE = re.compile(r"^(?P<sample>.+)_view(?P<view>\d+)$")


@dataclass(frozen=True)
class ImageRecord:
    """Single image plus the metadata needed for training or submission."""

    path: Path
    class_name: str
    split: str
    anomaly_name: str | None = None
    mask_path: Path | None = None

    @property
    def image_id(self) -> str:
        return self.path.stem

    @property
    def sample_id(self) -> str:
        # Extracts the core object identity, ignoring the specific camera view angle
        match = VIEW_RE.match(self.path.stem)
        return match.group("sample") if match else self.path.stem

    @property
    def view(self) -> int | None:
        match = VIEW_RE.match(self.path.stem)
        return int(match.group("view")) if match else None


def list_classes(data_root: Path) -> list[str]:
    """Return the available class folders in stable order."""
    return sorted(p.name for p in data_root.glob("class_*") if p.is_dir())


def _images_under(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def good_train_records(data_root: Path, class_name: str) -> list[ImageRecord]:
    """Images known to be normal; these are used for synthetic defect generation."""
    root = data_root / class_name / "train" / "good"
    return [
        ImageRecord(path=p, class_name=class_name, split="train_good")
        for p in _images_under(root)
    ]


def anomaly_train_records(data_root: Path, class_name: str) -> list[ImageRecord]:
    """Annotated training anomalies paired with their binary masks when present."""
    records: list[ImageRecord] = []
    train_root = data_root / class_name / "train"
    for anomaly_dir in sorted(train_root.glob("anomaly_*")):
        if not anomaly_dir.is_dir():
            continue
        # Maps each anomaly image to its corresponding ground truth mask by matching the file name
        mask_root = data_root / class_name / "ground_truth_train" / anomaly_dir.name
        for image_path in _images_under(anomaly_dir):
            mask_path = mask_root / image_path.name
            records.append(
                ImageRecord(
                    path=image_path,
                    class_name=class_name,
                    split="train_anomaly",
                    anomaly_name=anomaly_dir.name,
                    mask_path=mask_path if mask_path.exists() else None,
                )
            )
    return records


def test_records(data_root: Path, class_name: str) -> list[ImageRecord]:
    """Unlabeled test images that must appear in the final submission."""
    root = data_root / class_name / "test"
    return [
        ImageRecord(path=p, class_name=class_name, split="test")
        for p in _images_under(root)
    ]


class ImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
            # Returns (height, width) to match PyTorch tensor conventions
            original_size = image.size[::-1]
            tensor = self.transform(image)
        return tensor, record, original_size


def load_mask(path: Path) -> "np.ndarray":
    import numpy as np

    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.float32)
    # Binarizes the mask into a strict 0/1 pixel map
    return (mask > 0).astype(np.uint8)
