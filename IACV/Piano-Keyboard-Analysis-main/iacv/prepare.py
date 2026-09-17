from __future__ import annotations

import json
from pathlib import Path

import cv2
import skimage.io

from src.frames_extractor import FramesExtractor
from src.frame_without_hands_extractor import FrameWithoutHandsExtractor
from src.keys_extraction import KeysExtractorThroughLines

from .paths import resolve_paths


def _load_clean_frames(clean_dir: Path) -> list:
    paths = sorted(clean_dir.glob("cleaned_frame_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    frames = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is not None:
            frames.append(im)
    return frames


def run_prepare(
    video: str,
    fps: int = 10,
    maxframes: int = 600,
    deviation: float = 1.0,
    save_sampled: bool = False,
    manual_roi: bool = False,
    mse_threshold_no_hands: float = 500.0,
) -> None:
    p = resolve_paths(video)

    p.frames_clean_dir.mkdir(parents=True, exist_ok=True)
    p.artifacts_keys_dir.mkdir(parents=True, exist_ok=True)

    if not p.raw_video.exists():
        raise FileNotFoundError(f"Missing input video: {p.raw_video}")

    # 1) frames
    extractor = FramesExtractor(
        frame_per_second=fps,
        max_number_frames=maxframes,
        deviation_threshold=deviation,
        show_plots=False,
        save_frames=save_sampled,
    )
    extractor(str(p.raw_video))

    # 2) frame without hands
    frames = _load_clean_frames(p.frames_clean_dir)
    if not frames:
        raise FileNotFoundError(f"No cleaned frames found in {p.frames_clean_dir}")

    fwhe = FrameWithoutHandsExtractor(manual=manual_roi, show_plots=False, mse_threshold=mse_threshold_no_hands)
    clear_frame = fwhe(frames)

    cv2.imwrite(str(p.frame_without_hands_png), clear_frame)

    # also save the first cleaned frame as 'background' for reference
    first_bg = cv2.imread(str(sorted(p.frames_clean_dir.glob('cleaned_frame_*.png'))[0]))
    if first_bg is not None:
        cv2.imwrite(str(p.background_png), first_bg)

    # 3) keys
    image = skimage.io.imread(str(p.frame_without_hands_png))
    extractor_keys = KeysExtractorThroughLines()
    white_keys_coords, black_keys_coords, angle = extractor_keys(image)

    def key_to_dict(k):
        y_ul, x_ul, y_dr, x_dr = k.coords()
        return {"y_ul": int(y_ul), "x_ul": int(x_ul), "y_dr": int(y_dr), "x_dr": int(x_dr), "name": str(k)}

    white_out = {name: key_to_dict(k) for name, k in white_keys_coords.items()}
    black_out = {name: key_to_dict(k) for name, k in black_keys_coords.items()}

    p.white_keys_json.write_text(json.dumps(white_out, indent=2), encoding="utf-8")
    p.black_keys_json.write_text(json.dumps(black_out, indent=2), encoding="utf-8")
    p.rotation_angle_txt.write_text(str(angle), encoding="utf-8")

    # overlay (same drawing as legacy)
    if image.shape[2] == 4:
        base = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        base = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    overlay = base.copy()
    th = 2
    for key in white_keys_coords.values():
        y1, x1, y2, x2 = key.coords()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), th)
    for key in black_keys_coords.values():
        y1, x1, y2, x2 = key.coords()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), th)

    alpha = 0.35
    out = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)
    cv2.imwrite(str(p.keys_overlay_png), out)
