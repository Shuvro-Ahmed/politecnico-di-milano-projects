from __future__ import annotations

"""Post-processing utilities for continuous anomaly masks.

The neural networks output dense probability-like maps. These helpers make the
maps more submission-friendly by smoothing small artifacts, removing tiny
components, and optionally suppressing background pixels outside the object.
"""

import cv2
import numpy as np
from PIL import Image


def normalize_scores(score: np.ndarray, low: float, high: float, gamma: float) -> np.ndarray:
    """Map raw scores to [0, 1] with optional gamma sharpening."""
    denom = max(float(high - low), 1e-6)
    x = np.clip((score.astype(np.float32) - low) / denom, 0.0, 1.0)
    if gamma != 1.0:
        x = np.power(x, gamma, dtype=np.float32)
    return x


def remove_small_components(score: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    """Drop connected components that are too small to be plausible defects."""
    if min_area <= 0 or threshold <= 0:
        return score

    binary = (score >= threshold).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary, dtype=bool)
    # Loop starts at 1 because label 0 is always reserved for the background
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep |= labels == label

    cleaned = score.copy()
    # Zero out pixels that crossed the threshold but belong to discarded small components
    cleaned[~keep & (binary > 0)] = 0.0
    return cleaned


def postprocess_score(
    score: np.ndarray,
    low: float,
    high: float,
    gamma: float = 1.25,
    blur_sigma: float = 1.2,
    low_cut: float = 0.015,
    component_threshold: float = 0.12,
    min_component_area: int = 8,
) -> np.ndarray:
    """Apply the same smoothing and cleanup parameters used for final prediction."""
    x = normalize_scores(score, low=low, high=high, gamma=gamma)
    if blur_sigma > 0:
        # Enforce an odd kernel size (e.g., 3, 5, 7...) as required by OpenCV's GaussianBlur
        ksize = max(3, int(round(blur_sigma * 6)) | 1)
        x = cv2.GaussianBlur(x, (ksize, ksize), blur_sigma)
    if low_cut > 0:
        x[x < low_cut] = 0.0
    x = remove_small_components(x, threshold=component_threshold, min_area=min_component_area)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def foreground_mask_from_image(
    image_path,
    output_shape: tuple[int, int],
    value_threshold: int = 20,
    saturation_threshold: int = 0,
    min_area_ratio: float = 0.005,
    close_size: int = 9,
    dilate_size: int = 7,
) -> np.ndarray:
    """Estimate the object foreground from the RGB image.

    The dataset backgrounds are mostly dark, so a simple value/saturation mask is
    enough to reduce false positives outside the object without relying on test
    labels.
    """
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    mask = value > value_threshold
    if saturation_threshold > 0:
        mask |= saturation > saturation_threshold
    mask = mask.astype(np.uint8)

    if close_size > 0:
    # Morphological close/open filters out tiny holes inside the object and stray dust in the background
        k = max(3, int(close_size) | 1)
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = int(mask.shape[0] * mask.shape[1] * min_area_ratio)
    kept = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            kept[labels == label] = 1

    # Fallback: if everything gets wiped out by the area constraint, preserve the largest detected component
    if kept.sum() == 0 and n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        kept[labels == largest] = 1

    if dilate_size > 0:
        # Slightly expand the mask boundary to ensure edge defects aren't accidentally truncated
        k = max(3, int(dilate_size) | 1)
        kernel = np.ones((k, k), dtype=np.uint8)
        kept = cv2.dilate(kept, kernel, iterations=1)

    if kept.shape != output_shape:
        # OpenCV resize expects (width, height), so we invert the standard NumPy (rows, cols) shape
        kept = cv2.resize(kept, output_shape[::-1], interpolation=cv2.INTER_NEAREST)
    return kept.astype(np.float32)
