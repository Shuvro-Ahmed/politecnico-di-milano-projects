from __future__ import annotations

"""Supervised segmentation models for the final Spacepresso submission.

The pipeline trains one binary segmentation model per object class. Real anomaly
masks are used when available, while normal images are augmented with synthetic
defects so every class has enough positive masks for training. At prediction
time the per-class models produce continuous score maps that are encoded into
the q8rle submission format.
"""

import csv
import json
import math
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .data import anomaly_train_records, good_train_records, list_classes, load_mask, test_records
from .postprocess import foreground_mask_from_image, postprocess_score
from .q8rle import float_matrix_to_q8rle


def aggregate_views(
    records_and_scores: list[tuple],
    method: str = "suppress_low",
    global_mean_threshold: float = 0.02,
    min_views: int = 3,
) -> list[tuple]:
    """Apply a small multi-view consistency rule to one physical sample.

    Each sample is photographed from multiple viewpoints. If every view has a
    very low anomaly score, the whole sample is treated as normal; otherwise the
    original per-view masks are kept. This was only used during prediction, not
    during training.
    """
    if method == "none":
        return records_and_scores

    view_means = np.array([float(score.mean()) for _, score in records_and_scores])
    sample_confidence = float(view_means.max())

    if method == "suppress_low" and sample_confidence < global_mean_threshold:
        return [(rec, np.zeros_like(score)) for rec, score in records_and_scores]

    if method == "reweight":
        weight = np.clip(sample_confidence / max(global_mean_threshold, 1e-8), 0.0, 1.0)
        return [
            (rec, np.clip(score * weight, 0.0, 1.0).astype(np.float32))
            for rec, score in records_and_scores
        ]

    if method == "clip_low":
        below_threshold = view_means < global_mean_threshold
        n_below = int(below_threshold.sum())
        if n_below > len(records_and_scores) - min_views:
            return [(rec, np.zeros_like(score)) for rec, score in records_and_scores]
        return records_and_scores

    return records_and_scores


def _require_cv2():
    """Import OpenCV only when a segmentation command actually needs it."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Install opencv-python-headless before using segmentation commands") from exc
    return cv2


def _require_albumentations():
    """Import Albumentations lazily so CLI help works before dependencies load."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as exc:
        raise ImportError("Install albumentations before using segmentation commands") from exc
    return A, ToTensorV2


def _require_smp():
    """Import segmentation-models-pytorch lazily for the A/B model families."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError("Install segmentation-models-pytorch before using segmentation commands") from exc
    return smp


def _require_transformers():
    """Import Hugging Face Transformers lazily for the SegFormer model family."""
    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError as exc:
        raise ImportError("Install transformers before using SegFormer model S") from exc
    return SegformerForSemanticSegmentation


def seed_everything(seed: int) -> None:
    """Make data splits, synthetic defects, and torch initialization repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the CLI device flag into a concrete torch device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _read_rgb(path: Path, image_size: int) -> np.ndarray:
    """Read an RGB image and resize it to the model training resolution."""
    cv2 = _require_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def _read_mask(path: Path, image_size: int) -> np.ndarray:
    """Read a binary anomaly mask and resize it without interpolation artifacts."""
    cv2 = _require_cv2()
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)


def _training_aug():
    """Augment training samples while preserving the paired segmentation mask."""
    A, ToTensorV2 = _require_albumentations()
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.ColorJitter(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def _inference_aug():
    """Normalize validation/test images using ImageNet statistics."""
    A, ToTensorV2 = _require_albumentations()
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def _odd(n: int) -> int:
    return n if n % 2 else n + 1


def _perlin_like(shape: tuple[int, int], scale: float, octaves: int = 4) -> np.ndarray:
    """Generate smooth noise used to make synthetic defects less geometric."""
    h, w = shape
    noise = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    freq = 1.0 / max(scale, 1.0)
    for _ in range(octaves):
        phase_x = random.uniform(0.0, 2.0 * math.pi)
        phase_y = random.uniform(0.0, 2.0 * math.pi)
        angle = random.uniform(0.0, math.pi)
        xs = np.linspace(phase_x, phase_x + freq * w * 2.0 * math.pi, w, dtype=np.float32)
        ys = np.linspace(phase_y, phase_y + freq * h * 2.0 * math.pi, h, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        noise += amp * np.sin(xv * math.cos(angle) + yv * math.sin(angle))
        amp *= 0.5
        freq *= 2.0
    lo, hi = float(noise.min()), float(noise.max())
    return (noise - lo) / (hi - lo + 1e-8)


def _object_mask(image: np.ndarray) -> np.ndarray:
    """Estimate the object area so synthetic defects stay on the product.

    The challenge images usually have a simple background. This mask is not a
    label; it only prevents synthetic scratches or missing chunks from being
    painted on empty background pixels.
    """
    cv2 = _require_cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cr = max(4, min(h, w) // 8)
    center = binary[h // 2 - cr : h // 2 + cr, w // 2 - cr : w // 2 + cr]
    if center.size and center.mean() < 127:
        binary = cv2.bitwise_not(binary)

    cs = max(4, min(h, w) // 12)
    corners = np.concatenate(
        [
            binary[:cs, :cs].ravel(),
            binary[:cs, -cs:].ravel(),
            binary[-cs:, :cs].ravel(),
            binary[-cs:, -cs:].ravel(),
        ]
    )
    if corners.mean() > 80:
        binary = cv2.bitwise_not(binary)

    fill_ratio = float(np.count_nonzero(binary)) / float(binary.size)
    if fill_ratio < 0.04 or fill_ratio > 0.88:
        corner_gray = np.concatenate(
            [
                gray[:cs, :cs].ravel(),
                gray[:cs, -cs:].ravel(),
                gray[-cs:, :cs].ravel(),
                gray[-cs:, -cs:].ravel(),
            ]
        )
        bg = float(corner_gray.mean())
        if bg < 80:
            _, binary = cv2.threshold(blurred, int(bg + 35), 255, cv2.THRESH_BINARY)
        else:
            _, binary = cv2.threshold(blurred, int(bg - 35), 255, cv2.THRESH_BINARY_INV)
        center = binary[h // 2 - cr : h // 2 + cr, w // 2 - cr : w // 2 + cr]
        if center.size and center.mean() < 127:
            binary = cv2.bitwise_not(binary)

    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    open_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary
    clean = np.zeros_like(binary)
    cv2.drawContours(clean, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return clean


def _sample_mask_point(mask: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    i = random.randrange(len(ys))
    return int(xs[i]), int(ys[i])


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    cv2 = _require_cv2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.erode(mask, kernel, iterations=1)


def _gen_line_defect(image: np.ndarray, obj_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Draw a thin scratch-like defect and return its supervision mask."""
    cv2 = _require_cv2()
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    point = _sample_mask_point(_erode(obj_mask, 3))
    if point is None:
        return image, mask
    x, y = point
    local = image[y, x].astype(np.float32)
    color = tuple(int(v) for v in np.clip(local * random.choice([random.uniform(0.04, 0.30), random.uniform(1.7, 3.0)]), 0, 255))
    pts = [(x, y)]
    angle = random.uniform(0.0, 2.0 * math.pi)
    for _ in range(random.randint(3, 10)):
        angle += random.uniform(-0.55, 0.55)
        step = random.randint(5, 22)
        pts.append(
            (
                int(np.clip(pts[-1][0] + math.cos(angle) * step, 0, w - 1)),
                int(np.clip(pts[-1][1] + math.sin(angle) * step, 0, h - 1)),
            )
        )
    for i, (p0, p1) in enumerate(zip(pts, pts[1:])):
        thickness = max(1, random.randint(1, 3) - i // 4)
        cv2.line(image, p0, p1, color, thickness)
        cv2.line(mask, p0, p1, 1.0, thickness + 1)
    return image, mask * (obj_mask / 255.0)


def _gen_blob_defect(image: np.ndarray, obj_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create a stain/blob defect with organic edges."""
    cv2 = _require_cv2()
    h, w = image.shape[:2]
    point = _sample_mask_point(_erode(obj_mask, 6))
    if point is None:
        return image, np.zeros((h, w), dtype=np.float32)
    cx, cy = point
    r = random.randint(5, 38)
    region = np.zeros((h, w), dtype=np.float32)
    if random.random() < 0.55:
        cv2.circle(region, (cx, cy), r, 1.0, -1)
    else:
        cv2.ellipse(region, (cx, cy), (r, random.randint(4, max(5, r))), random.randint(0, 180), 0, 360, 1.0, -1)
    organic = region * (_perlin_like((h, w), max(6.0, r * 1.5), 4) > random.uniform(0.22, 0.48)).astype(np.float32)
    organic *= obj_mask / 255.0
    if organic.sum() < 15:
        return image, np.zeros((h, w), dtype=np.float32)
    mode = random.choice(["dark", "bright", "rust"])
    img_f = image.astype(np.float32)
    area = organic > 0
    if mode == "dark":
        img_f[area] *= random.uniform(0.05, 0.45)
    elif mode == "bright":
        img_f[area] = np.clip(img_f[area] * random.uniform(1.25, 1.8) + 15, 0, 255)
    else:
        img_f[:, :, 0][area] = np.clip(img_f[:, :, 0][area] * random.uniform(1.0, 1.45) + 25, 0, 255)
        img_f[:, :, 1][area] *= random.uniform(0.35, 0.75)
        img_f[:, :, 2][area] *= random.uniform(0.10, 0.45)
    return img_f.astype(np.uint8), organic


def _gen_missing_chunk(image: np.ndarray, obj_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a missing or chipped object part near the foreground boundary."""
    cv2 = _require_cv2()
    h, w = image.shape[:2]
    contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, np.zeros((h, w), dtype=np.float32)
    boundary = max(contours, key=cv2.contourArea).reshape(-1, 2)
    cx, cy = boundary[random.randrange(len(boundary))]
    cx = int(np.clip(cx + random.randint(-12, 12), 0, w - 1))
    cy = int(np.clip(cy + random.randint(-12, 12), 0, h - 1))
    axes = (random.randint(max(8, w // 10), max(12, w // 3)), random.randint(max(8, h // 10), max(12, h // 3)))
    erase = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(erase, (cx, cy), axes, random.randint(0, 180), 0, 360, 1.0, -1)
    organic = erase * (_perlin_like((h, w), min(axes) * 0.9, 3) > 0.32).astype(np.float32)
    anomaly = organic * (obj_mask / 255.0)
    if anomaly.sum() < 40:
        anomaly = erase * (obj_mask / 255.0)
    corners = [image[:18, :18], image[:18, -18:], image[-18:, :18], image[-18:, -18:]]
    bg = min(corners, key=lambda x: float(x.mean())).reshape(-1, 3).mean(axis=0)
    soft = cv2.GaussianBlur(anomaly, (_odd(min(axes[0], 25) * 2 - 1), _odd(min(axes[0], 25) * 2 - 1)), 0)[..., None]
    out = np.clip(image * (1.0 - soft) + bg * soft, 0, 255).astype(np.uint8)
    return out, anomaly.astype(np.float32)


def _gen_structural_copypaste(image: np.ndarray, obj_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Paste an object patch onto another object location to mimic structural defects."""
    cv2 = _require_cv2()
    h, w = image.shape[:2]
    inner = _erode(obj_mask, 6)
    src = _sample_mask_point(inner)
    dst = _sample_mask_point(inner)
    if src is None or dst is None:
        return image, np.zeros((h, w), dtype=np.float32)
    sx, sy = src
    dx, dy = dst
    size = random.randint(12, 42)
    if sx + size >= w or sy + size >= h or dx + size >= w or dy + size >= h:
        return image, np.zeros((h, w), dtype=np.float32)
    if obj_mask[sy : sy + size, sx : sx + size].mean() < 100:
        return image, np.zeros((h, w), dtype=np.float32)
    patch = image[sy : sy + size, sx : sx + size].copy()
    patch = np.rot90(patch, random.randint(0, 3)).copy()
    if random.random() < 0.5:
        patch = cv2.flip(patch, random.choice([0, 1, -1]))
    dst_obj = (obj_mask[dy : dy + size, dx : dx + size] > 127).astype(np.float32)
    image[dy : dy + size, dx : dx + size] = (
        patch * dst_obj[..., None] + image[dy : dy + size, dx : dx + size] * (1.0 - dst_obj[..., None])
    ).astype(np.uint8)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[dy : dy + size, dx : dx + size] = dst_obj
    return image, mask


def _texture_copypaste(
    image: np.ndarray,
    obj_mask: np.ndarray,
    anomaly_pairs: list[tuple[Path, Path]],
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Paste a real anomaly texture onto a normal image when masks are available."""
    cv2 = _require_cv2()
    h, w = image.shape[:2]
    if not anomaly_pairs:
        return image, np.zeros((h, w), dtype=np.float32)
    anomaly_image_path, anomaly_mask_path = random.choice(anomaly_pairs)
    anomaly_image = _read_rgb(anomaly_image_path, image_size)
    anomaly_mask = _read_mask(anomaly_mask_path, image_size)
    k = random.randint(0, 3)
    anomaly_image = np.rot90(anomaly_image, k).copy()
    anomaly_mask = np.rot90(anomaly_mask, k).copy()
    if random.random() < 0.5:
        flip = random.choice([0, 1, -1])
        anomaly_image = cv2.flip(anomaly_image, flip)
        anomaly_mask = cv2.flip(anomaly_mask, flip)
    shift_x, shift_y = random.randint(-image_size // 4, image_size // 4), random.randint(-image_size // 4, image_size // 4)
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    anomaly_image = cv2.warpAffine(anomaly_image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    anomaly_mask = cv2.warpAffine(anomaly_mask, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    anomaly_image = np.clip(anomaly_image.astype(np.float32) * random.uniform(0.45, 1.55), 0, 255).astype(np.uint8)
    hard = (anomaly_mask > 0.5).astype(np.float32) * (obj_mask / 255.0)
    if hard.sum() < 15:
        return image, np.zeros((h, w), dtype=np.float32)
    soft = cv2.GaussianBlur(hard, (11, 11), 0)[..., None]
    out = np.clip(image * (1.0 - soft) + anomaly_image * soft, 0, 255).astype(np.uint8)
    return out, hard


class AnomalyCopyPasteDataset(Dataset):
    """Online dataset that mixes real anomalies, normal samples, and synthetic masks.

    A new anomaly variant can be generated every time an item is read. This makes
    the small labeled anomaly set more useful without storing augmented images on
    disk.
    """
    def __init__(
        self,
        good_paths: list[Path],
        anomaly_pairs: list[tuple[Path, Path]],
        image_size: int,
        transforms,
        real_prob: float = 0.20,
        good_prob: float = 0.15,
        synthetic_prob: float = 0.25,
        missing_prob: float = 0.20,
        copypaste_prob: float = 0.20,
        epoch_multiplier: int = 1,
        cache_object_masks: bool = True,
    ):
        self.good_paths = good_paths
        self.anomaly_pairs = anomaly_pairs
        self.image_size = image_size
        self.transforms = transforms
        self.probs = {
            "real": float(real_prob),
            "good": float(good_prob),
            "synthetic": float(synthetic_prob),
            "missing": float(missing_prob),
            "copypaste": float(copypaste_prob),
        }
        self.epoch_multiplier = max(1, int(epoch_multiplier))
        self.cache_object_masks = cache_object_masks
        self._mask_cache: dict[Path, np.ndarray] = {}
        self._generators = [_gen_line_defect, _gen_blob_defect, _gen_structural_copypaste]

    def __len__(self) -> int:
        return len(self.good_paths) * self.epoch_multiplier

    def _branch(self) -> str:
        """Sample which type of training example to produce."""
        items = list(self.probs.items())
        total = sum(max(0.0, weight) for _, weight in items)
        if total <= 0:
            return "good"
        pick = random.random() * total
        running = 0.0
        for name, weight in items:
            running += max(0.0, weight)
            if pick <= running:
                return name
        return items[-1][0]

    def _object_mask_for(self, path: Path, image: np.ndarray) -> np.ndarray:
        """Cache object masks because they are reused many times per epoch."""
        if self.cache_object_masks and path in self._mask_cache:
            return self._mask_cache[path]
        mask = _object_mask(image)
        if self.cache_object_masks:
            self._mask_cache[path] = mask
        return mask

    def __getitem__(self, index: int):
        base_index = index % len(self.good_paths)
        branch = self._branch()
        if branch == "real" and self.anomaly_pairs:
            # Real anomaly masks keep the model anchored to the true label style.
            image_path, mask_path = random.choice(self.anomaly_pairs)
            image = _read_rgb(image_path, self.image_size)
            mask = _read_mask(mask_path, self.image_size)
        else:
            good_path = self.good_paths[base_index]
            image = _read_rgb(good_path, self.image_size)
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            object_mask = self._object_mask_for(good_path, image)

            if branch == "good":
                # Normal images with an empty mask teach the model not to fire everywhere.
                pass
            elif branch == "missing" and object_mask.sum() > 300:
                image, mask = _gen_missing_chunk(image.copy(), object_mask)
            elif branch == "copypaste" and self.anomaly_pairs:
                image, mask = _texture_copypaste(image.copy(), object_mask, self.anomaly_pairs, self.image_size)
            elif object_mask.sum() > 300:
                combined = np.zeros_like(mask)
                for _ in range(random.randint(1, 3)):
                    image, anomaly_mask = random.choice(self._generators)(image.copy(), object_mask)
                    combined = np.clip(combined + anomaly_mask, 0.0, 1.0)
                mask = combined

        augmented = self.transforms(image=image, mask=mask)
        image_tensor = augmented["image"]
        mask_tensor = augmented["mask"].float().unsqueeze(0)
        return image_tensor, mask_tensor


class SegmentationEvalDataset(Dataset):
    """Dataset wrapper used for validation and test-time prediction."""
    def __init__(self, records, image_size: int):
        self.records = records
        self.image_size = image_size
        self.transforms = _inference_aug()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
            original_size = image.size[::-1]
            resized = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            arr = np.asarray(resized, dtype=np.uint8)
        tensor = self.transforms(image=arr)["image"]
        return tensor, record, original_size


def _collate_eval(batch):
    images, records, sizes = zip(*batch)
    return torch.stack(images), list(records), list(sizes)


def _anomaly_pairs(data_root: Path, class_name: str) -> list[tuple[Path, Path]]:
    """Collect real anomaly image/mask pairs for one class."""
    pairs = []
    for record in anomaly_train_records(data_root, class_name):
        if record.mask_path is not None:
            pairs.append((record.path, record.mask_path))
    return pairs


def _split_good(paths: list[Path], seed: int, val_fraction: float = 0.1) -> tuple[list[Path], list[Path]]:
    """Hold out a small normal-image split for validation loss tracking."""
    paths = list(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    val_count = max(1, int(round(len(paths) * val_fraction))) if len(paths) > 1 else 0
    return paths[val_count:], paths[:val_count]


def _model_path(work_dir: Path, class_name: str, model_type: str) -> Path:
    """Consistent checkpoint path for one class/model family."""
    return work_dir / "segmentation" / f"{class_name}_{model_type}.pt"


def _find_initial_checkpoint(
    initial_weights_dir: Path | None,
    work_dir: Path,
    class_name: str,
    model_type: str,
) -> Path | None:
    """Find a previously saved checkpoint to use as a training starting point.

    The backup directory used by the notebook stores files as:

        checkpoints/outputs_seg_ab/segmentation/class_01_A.pt

    This helper also accepts simpler layouts such as:

        checkpoints/segmentation/class_01_A.pt
        checkpoints/class_01_A.pt

    A final recursive search is included so a Drive folder can be reused even if
    the files were moved manually.
    """
    if initial_weights_dir is None:
        return None
    filename = f"{class_name}_{model_type}.pt"
    candidates = [
        initial_weights_dir / work_dir.name / "segmentation" / filename,
        initial_weights_dir / "segmentation" / filename,
        initial_weights_dir / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(initial_weights_dir.rglob(filename)) if initial_weights_dir.exists() else []
    return matches[0] if matches else None


def _load_initial_weights_if_available(
    model: nn.Module,
    initial_weights_dir: Path | None,
    work_dir: Path,
    class_name: str,
    model_type: str,
    device: torch.device,
) -> Path | None:
    """Warm-start a model from a matching checkpoint if one is available."""
    checkpoint_path = _find_initial_checkpoint(initial_weights_dir, work_dir, class_name, model_type)
    if checkpoint_path is None:
        return None
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state_dict)
    return checkpoint_path


class SegFormerForAnomaly(nn.Module):
    """SegFormer-B2 adapted from semantic segmentation to one anomaly channel."""
    def __init__(self, pretrained_model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512"):
        super().__init__()
        segformer_cls = _require_transformers()
        self.segformer = segformer_cls.from_pretrained(
            pretrained_model_name,
            ignore_mismatched_sizes=True,
        )
        config = self.segformer.config
        self.segformer.decode_head.classifier = nn.Conv2d(config.decoder_hidden_size, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.segformer(x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_segmentation_model(
    model_type: str,
    device: torch.device,
    encoder_weights: str | None = "imagenet",
    segformer_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
):
    """Create one of the three model families used in the final ensemble.

    A and B are CNN segmentation models from segmentation-models-pytorch. S is a
    transformer model. The diversity between these families is useful for the
    final blend.
    """
    if model_type == "A":
        smp = _require_smp()
        model = smp.Unet(
            encoder_name="efficientnet-b4",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
    elif model_type == "B":
        smp = _require_smp()
        model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b5",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
    elif model_type == "S":
        model = SegFormerForAnomaly(pretrained_model_name=segformer_name)
    else:
        raise ValueError(f"Unknown segmentation model type: {model_type}")
    return model.to(device)


class HybridSegmentationLoss(torch.nn.Module):
    """Lovasz + focal loss used for the A/B CNN models."""
    def __init__(self):
        super().__init__()
        smp = _require_smp()
        self.lovasz = smp.losses.LovaszLoss(mode="binary", from_logits=True)
        self.focal = smp.losses.FocalLoss(mode="binary", alpha=0.25, gamma=2.0)

    def forward(self, y_pred, y_true):
        return 0.7 * self.lovasz(y_pred, y_true) + 0.3 * self.focal(y_pred, y_true)


class BCEDiceLoss(torch.nn.Module):
    """BCE + Dice loss used for the SegFormer model."""
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        smp = _require_smp()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary")
        self.bce_weight = float(bce_weight)

    def forward(self, y_pred, y_true):
        return self.bce_weight * self.bce(y_pred, y_true) + (1.0 - self.bce_weight) * self.dice(y_pred, y_true)


def build_segmentation_loss(name: str) -> torch.nn.Module:
    """Resolve the CLI loss name into a torch module."""
    if name == "hybrid":
        return HybridSegmentationLoss()
    if name == "bce_dice":
        return BCEDiceLoss()
    raise ValueError(f"Unknown segmentation loss: {name}")


def train_segmentation(args) -> None:
    """Train the requested segmentation model family for each selected class."""
    seed_everything(args.seed)
    data_root = Path(args.data)
    work_dir = Path(args.work_dir)
    backup_dir = Path(args.backup_dir) if getattr(args, "backup_dir", None) else None
    initial_weights_dir = (
        Path(args.initial_weights_dir)
        if getattr(args, "initial_weights_dir", None)
        else None
    )
    classes = args.classes or list_classes(data_root)
    device = resolve_device(args.device)
    criterion = build_segmentation_loss(args.seg_loss)

    summaries: dict[str, dict] = {}
    for class_name in classes:
        good_paths = [r.path for r in good_train_records(data_root, class_name)]
        anomaly_pairs = _anomaly_pairs(data_root, class_name)
        if not good_paths:
            print(f"{class_name}: skipping, no good images")
            continue
        train_good, val_good = _split_good(good_paths, args.seed)

        for model_type in args.seg_models:
            print(f"\nTraining segmentation {class_name} model {model_type}")
            # A fresh dataset is created for each model family because the
            # synthetic masks are sampled online during training.
            train_dataset = AnomalyCopyPasteDataset(
                train_good,
                anomaly_pairs,
                image_size=args.image_size,
                transforms=_training_aug(),
                real_prob=args.real_prob,
                good_prob=args.good_prob,
                synthetic_prob=args.synthetic_prob,
                missing_prob=args.missing_prob,
                copypaste_prob=args.copypaste_prob,
                epoch_multiplier=args.epoch_multiplier,
            )
            val_dataset = AnomalyCopyPasteDataset(
                val_good or train_good,
                anomaly_pairs,
                image_size=args.image_size,
                transforms=_inference_aug(),
                real_prob=args.real_prob,
                good_prob=args.good_prob,
                synthetic_prob=args.synthetic_prob,
                missing_prob=args.missing_prob,
                copypaste_prob=args.copypaste_prob,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

            model = build_segmentation_model(model_type, device, segformer_name=args.segformer_name)
            initial_checkpoint = _load_initial_weights_if_available(
                model,
                initial_weights_dir,
                work_dir,
                class_name,
                model_type,
                device,
            )
            if initial_checkpoint is not None:
                print(f"{class_name} {model_type}: initialized from {initial_checkpoint}")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            # Cosine decay keeps the later epochs conservative while still
            # allowing the first epochs to adapt from ImageNet/ADE pretraining.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.min_lr,
            )
            scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
            best_val = float("inf")
            best_epoch = 0
            epochs_without_improvement = 0
            save_path = _model_path(work_dir, class_name, model_type)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            for epoch in range(1, args.epochs + 1):
                model.train()
                train_loss = 0.0
                for images, masks in tqdm(train_loader, desc=f"{class_name} {model_type} epoch {epoch}/{args.epochs}"):
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                        loss = criterion(model(images), masks)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss += float(loss.item()) * images.size(0)

                scheduler.step()
                train_loss /= max(1, len(train_loader.dataset))

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images = images.to(device, non_blocking=True)
                        masks = masks.to(device, non_blocking=True)
                        loss = criterion(model(images), masks)
                        val_loss += float(loss.item()) * images.size(0)
                val_loss /= max(1, len(val_loader.dataset))
                print(
                    f"{class_name} {model_type}: epoch {epoch}/{args.epochs} "
                    f"loss={train_loss:.5f} val_loss={val_loss:.5f}"
                )

                # Only the best validation checkpoint is kept. This is the file
                # later loaded by predict-seg, so early stopping does not change
                # the checkpoint selection rule.
                improved = val_loss < (best_val - args.early_stopping_min_delta)
                if improved:
                    best_val = val_loss
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "model_type": model_type,
                            "image_size": args.image_size,
                            "segformer_name": args.segformer_name,
                            "best_val_loss": best_val,
                        },
                        save_path,
                    )
                    print(f"{class_name} {model_type}: saved {save_path}")
                    if backup_dir is not None:
                        backup_path = backup_dir / work_dir.name / "segmentation" / save_path.name
                        backup_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(save_path, backup_path)
                        print(f"{class_name} {model_type}: backed up {backup_path}")
                else:
                    epochs_without_improvement += 1

                if (
                    args.early_stopping_patience > 0
                    and epochs_without_improvement >= args.early_stopping_patience
                ):
                    print(
                        f"{class_name} {model_type}: early stopping at epoch {epoch}; "
                        f"best epoch was {best_epoch} with val_loss={best_val:.5f}"
                    )
                    break

            summaries[f"{class_name}_{model_type}"] = {
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "path": str(save_path),
                "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint is not None else None,
                "train_good": len(train_good),
                "val_good": len(val_good),
                "anomaly_masks": len(anomaly_pairs),
            }
            del model, optimizer, scheduler, scaler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "segmentation_training_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )


def _find_prediction_checkpoint(
    work_dir: Path,
    weights_dir: Path | None,
    class_name: str,
    model_type: str,
) -> Path:
    """Find the checkpoint used for prediction.

    Prediction normally reads from `work_dir/segmentation`. When Colab is
    restarted, the local `/content` work directory may be gone, while the backup
    on Drive is still available. `weights_dir` lets inference fall back to that
    persistent folder.
    """
    filename = f"{class_name}_{model_type}.pt"
    candidates = [_model_path(work_dir, class_name, model_type)]
    if weights_dir is not None:
        candidates.extend(
            [
                weights_dir / work_dir.name / "segmentation" / filename,
                weights_dir / "segmentation" / filename,
                weights_dir / filename,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if weights_dir is not None and weights_dir.exists():
        matches = sorted(weights_dir.rglob(filename))
        if matches:
            return matches[0]
    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Checkpoint not found for {class_name} {model_type}. Searched:\n{searched}")


def _load_segmentation_models(
    work_dir: Path,
    class_name: str,
    model_types: list[str],
    device: torch.device,
    weights_dir: Path | None = None,
):
    """Load the best saved checkpoint for every requested model family."""
    models = []
    for model_type in model_types:
        path = _find_prediction_checkpoint(work_dir, weights_dir, class_name, model_type)
        print(f"{class_name} {model_type}: loading {path}")
        payload = torch.load(path, map_location=device)
        saved_type = payload.get("model_type", model_type)
        model = build_segmentation_model(
            saved_type,
            device,
            encoder_weights=None,
            segformer_name=payload.get("segformer_name", "nvidia/segformer-b2-finetuned-ade-512-512"),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
    return models


def _predict_tta(model, images: torch.Tensor, mode: str, scales: list[float]) -> np.ndarray:
    """Predict one model with flip and scale test-time augmentation.

    The model always receives normalized tensors. Predictions are unflipped and
    resized back to the base image size before averaging.
    """
    specs: list[tuple[str, tuple[int, ...]]] = [("none", ())]
    if mode in {"hflip", "hvflip"}:
        specs.append(("hflip", (3,)))
    if mode == "hvflip":
        specs.append(("vflip", (2,)))
        specs.append(("hvflip", (2, 3)))

    maps = []
    _, _, in_h, in_w = images.shape
    with torch.no_grad():
        for scale in scales:
            if scale <= 0:
                raise ValueError(f"TTA scales must be positive, got {scale}")
            if abs(scale - 1.0) < 1e-6:
                scaled = images
            else:
                h = int(math.ceil(in_h * scale / 32.0) * 32)
                w = int(math.ceil(in_w * scale / 32.0) * 32)
                scaled = F.interpolate(images, size=(h, w), mode="bilinear", align_corners=False)
            for name, dims in specs:
                batch = torch.flip(scaled, dims=dims) if dims else scaled
                pred = torch.sigmoid(model(batch))
                if pred.shape[-2:] != (in_h, in_w):
                    pred = F.interpolate(pred, size=(in_h, in_w), mode="bilinear", align_corners=False)
                pred_np = pred.detach().cpu().numpy()[:, 0]
                if name == "hflip":
                    pred_np = pred_np[:, :, ::-1]
                elif name == "vflip":
                    pred_np = pred_np[:, ::-1, :]
                elif name == "hvflip":
                    pred_np = pred_np[:, ::-1, ::-1]
                maps.append(pred_np)
    return np.mean(maps, axis=0).astype(np.float32)


def _resize_score(score: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a score map back to the original image height and width."""
    cv2 = _require_cv2()
    h, w = size
    resized = cv2.resize(score, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def _fuse_model_scores(model_scores: list[np.ndarray], mode: str) -> np.ndarray:
    """Combine predictions from multiple model families.

    `uncertainty` gives more weight to the model that is farther from 0.5 on a
    given batch, which downweights ambiguous predictions.
    """
    stack = np.stack(model_scores, axis=0).astype(np.float32)
    if mode == "mean":
        return stack.mean(axis=0)
    if mode == "softmax":
        return np.square(stack).mean(axis=0)
    if mode == "geometric":
        return np.power(np.clip(stack.prod(axis=0), 0.0, 1.0), 1.0 / stack.shape[0])
    if mode == "uncertainty":
        uncertainty = np.mean(1.0 - np.abs(stack - 0.5) * 2.0, axis=(2, 3))
        inv = 1.0 / (uncertainty + 1e-6)
        weights = inv / inv.sum(axis=0, keepdims=True)
        return (stack * weights[:, :, None, None]).sum(axis=0)
    raise ValueError(f"Unknown segmentation fusion mode: {mode}")


def _score_segmentation_batch(
    models,
    images: torch.Tensor,
    sizes,
    device: torch.device,
    tta: str,
    tta_scales: list[float],
    fusion: str,
) -> list[np.ndarray]:
    """Run all loaded models on one batch and return one fused map per image."""
    images = images.to(device, non_blocking=True)
    model_scores = [_predict_tta(model, images, tta, tta_scales) for model in models]
    fused = _fuse_model_scores(model_scores, fusion).astype(np.float32)
    return [_resize_score(score, size) for score, size in zip(fused, sizes)]


def _postprocess_seg_score(
    score: np.ndarray,
    record=None,
    blur_sigma: float = 1.5,
    low_cut: float = 0.01,
    component_threshold: float = 0.1,
    min_component_area: int = 8,
    foreground_mask: bool = False,
    foreground_background_weight: float = 0.0,
) -> np.ndarray:
    """Apply final smoothing, threshold cleanup, and optional foreground masking."""
    processed = postprocess_score(
        score,
        low=0.0,
        high=1.0,
        gamma=1.0,
        blur_sigma=blur_sigma,
        low_cut=low_cut,
        component_threshold=component_threshold,
        min_component_area=min_component_area,
    )
    if foreground_mask and record is not None:
        mask = foreground_mask_from_image(record.path, output_shape=processed.shape)
        processed = processed * mask + processed * (1.0 - mask) * float(foreground_background_weight)
    return np.clip(processed, 0.0, 1.0).astype(np.float32)


def evaluate_segmentation(args) -> None:
    """Measure pixel average precision on annotated training anomalies."""
    from sklearn.metrics import average_precision_score

    seed_everything(args.seed)
    data_root = Path(args.data)
    work_dir = Path(args.work_dir)
    weights_dir = Path(args.weights_dir) if getattr(args, "weights_dir", None) else None
    classes = args.classes or list_classes(data_root)
    device = resolve_device(args.device)

    all_scores = []
    all_masks = []
    per_class = {}
    for class_name in classes:
        models = _load_segmentation_models(work_dir, class_name, args.seg_models, device, weights_dir=weights_dir)
        records = [r for r in anomaly_train_records(data_root, class_name) if r.mask_path is not None]
        dataset = SegmentationEvalDataset(records, args.image_size)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate_eval,
        )
        class_scores = []
        class_masks = []

        if args.multiview:
            class_results: list[tuple] = []
            for images, batch_records, sizes in tqdm(loader, desc=f"Evaluate seg {class_name}"):
                score_maps = _score_segmentation_batch(
                    models, images, sizes, device, args.tta, args.tta_scales, args.fusion
                )
                for score, record in zip(score_maps, batch_records):
                    score = _postprocess_seg_score(
                        score,
                        record=record,
                        blur_sigma=args.blur_sigma,
                        low_cut=args.low_cut,
                        component_threshold=args.component_threshold,
                        min_component_area=args.min_component_area,
                        foreground_mask=args.foreground_mask,
                        foreground_background_weight=args.foreground_background_weight,
                    )
                    class_results.append((record, score))

            samples: dict[str, list] = {}
            for record, score in class_results:
                samples.setdefault(record.sample_id, []).append((record, score))
            for items in samples.values():
                items = aggregate_views(
                    items,
                    method=args.multiview_method,
                    global_mean_threshold=args.multiview_threshold,
                )
                for record, score in items:
                    mask = load_mask(record.mask_path)
                    class_scores.append(score.reshape(-1))
                    class_masks.append(mask.reshape(-1))
        else:
            for images, batch_records, sizes in tqdm(loader, desc=f"Evaluate seg {class_name}"):
                score_maps = _score_segmentation_batch(
                    models, images, sizes, device, args.tta, args.tta_scales, args.fusion
                )
                for score, record in zip(score_maps, batch_records):
                    score = _postprocess_seg_score(
                        score,
                        record=record,
                        blur_sigma=args.blur_sigma,
                        low_cut=args.low_cut,
                        component_threshold=args.component_threshold,
                        min_component_area=args.min_component_area,
                        foreground_mask=args.foreground_mask,
                        foreground_background_weight=args.foreground_background_weight,
                    )
                    class_scores.append(score.reshape(-1))
                    class_masks.append(load_mask(record.mask_path).reshape(-1))

        y_score = np.concatenate(class_scores)
        y_true = np.concatenate(class_masks)
        ap = float(average_precision_score(y_true, y_score))
        per_class[class_name] = ap
        all_scores.append(y_score)
        all_masks.append(y_true)
        print(f"{class_name}: segmentation pixel AP={ap:.5f}")
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    global_ap = float(average_precision_score(np.concatenate(all_masks), np.concatenate(all_scores)))
    result = {"pixel_ap": global_ap, "per_class": per_class, "params": {k: v for k, v in vars(args).items() if k != "func"}}
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "segmentation_validation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"global segmentation pixel AP={global_ap:.5f}")


def predict_segmentation(args) -> None:
    """Write q8rle masks for all selected test images."""
    seed_everything(args.seed)
    data_root = Path(args.data)
    work_dir = Path(args.work_dir)
    weights_dir = Path(args.weights_dir) if getattr(args, "weights_dir", None) else None
    out_csv = Path(args.output)
    classes = args.classes or list_classes(data_root)
    device = resolve_device(args.device)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        for class_name in classes:
            # Checkpoints are loaded per class to keep GPU memory bounded.
            models = _load_segmentation_models(work_dir, class_name, args.seg_models, device, weights_dir=weights_dir)
            dataset = SegmentationEvalDataset(test_records(data_root, class_name), args.image_size)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=_collate_eval,
            )

            if args.multiview:
                # Multi-view aggregation is optional. It can suppress samples
                # where every view has a very weak anomaly response.
                class_results: list[tuple] = []
                for images, records, sizes in tqdm(loader, desc=f"Predict seg {class_name}"):
                    score_maps = _score_segmentation_batch(
                        models, images, sizes, device, args.tta, args.tta_scales, args.fusion
                    )
                    for score, record in zip(score_maps, records):
                        score = _postprocess_seg_score(
                            score,
                            record=record,
                            blur_sigma=args.blur_sigma,
                            low_cut=args.low_cut,
                            component_threshold=args.component_threshold,
                            min_component_area=args.min_component_area,
                            foreground_mask=args.foreground_mask,
                            foreground_background_weight=args.foreground_background_weight,
                        )
                        class_results.append((record, score))

                samples: dict[str, list] = {}
                for record, score in class_results:
                    samples.setdefault(record.sample_id, []).append((record, score))
                for items in samples.values():
                    items = aggregate_views(
                        items,
                        method=args.multiview_method,
                        global_mean_threshold=args.multiview_threshold,
                    )
                    for record, score in items:
                        writer.writerow([record.image_id, float_matrix_to_q8rle(score)])
            else:
                # The final submitted notebook uses this simpler per-image path.
                for images, records, sizes in tqdm(loader, desc=f"Predict seg {class_name}"):
                    score_maps = _score_segmentation_batch(
                        models, images, sizes, device, args.tta, args.tta_scales, args.fusion
                    )
                    for score, record in zip(score_maps, records):
                        score = _postprocess_seg_score(
                            score,
                            record=record,
                            blur_sigma=args.blur_sigma,
                            low_cut=args.low_cut,
                            component_threshold=args.component_threshold,
                            min_component_area=args.min_component_area,
                            foreground_mask=args.foreground_mask,
                            foreground_background_weight=args.foreground_background_weight,
                        )
                        writer.writerow([record.image_id, float_matrix_to_q8rle(score)])

            del models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.zip:
        zip_path = out_csv.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_csv, arcname=out_csv.name)
        print(f"Wrote {zip_path}")
    else:
        print(f"Wrote {out_csv}")
