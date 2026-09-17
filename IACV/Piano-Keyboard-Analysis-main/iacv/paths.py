from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    video: str

    raw_video: Path
    frames_clean_dir: Path

    artifacts_dir: Path
    artifacts_keys_dir: Path
    artifacts_hands_dir: Path

    results_dir: Path

    background_png: Path
    frame_without_hands_png: Path

    white_keys_json: Path
    black_keys_json: Path
    rotation_angle_txt: Path
    keys_overlay_png: Path

    fingertips_raw_csv: Path
    fingertips_tracked_csv: Path
    landmarks_csv: Path
    pressed_per_finger_csv: Path
    pressed_per_finger_clean_csv: Path

    mapping_csv: Path
    pressed_keys_csv: Path
    vis_frames_dir: Path
    vis_mp4: Path


def resolve_paths(video_name: str) -> Paths:
    raw_video = ROOT / "data" / "raw_videos" / f"{video_name}.mp4"
    frames_clean_dir = ROOT / "data" / "frames" / video_name / "cleaned"

    artifacts_dir = ROOT / "artifacts" / video_name
    artifacts_keys_dir = artifacts_dir / "keys"
    artifacts_hands_dir = artifacts_dir / "hands"

    results_dir = ROOT / "results" / video_name

    background_png = artifacts_keys_dir / "background.png"
    frame_without_hands_png = artifacts_keys_dir / "frame_without_hands.png"

    white_keys_json = artifacts_keys_dir / "white_keys.json"
    black_keys_json = artifacts_keys_dir / "black_keys.json"
    rotation_angle_txt = artifacts_keys_dir / "rotation_angle.txt"
    keys_overlay_png = artifacts_keys_dir / "keys_overlay.png"

    fingertips_raw_csv = artifacts_hands_dir / "fingertips_raw.csv"
    fingertips_tracked_csv = artifacts_hands_dir / "fingertips_tracked.csv"
    landmarks_csv = artifacts_hands_dir / "landmarks_for_press.csv"
    pressed_per_finger_csv = artifacts_hands_dir / "pressed_per_finger.csv"
    pressed_per_finger_clean_csv = artifacts_hands_dir / "pressed_per_finger_clean.csv"

    mapping_csv = results_dir / "fingertips_to_keys.csv"
    pressed_keys_csv = results_dir / "pressed_keys_per_frame.csv"
    vis_frames_dir = results_dir / "frames"
    vis_mp4 = results_dir / "visualization.mp4"

    return Paths(
        video=video_name,
        raw_video=raw_video,
        frames_clean_dir=frames_clean_dir,
        artifacts_dir=artifacts_dir,
        artifacts_keys_dir=artifacts_keys_dir,
        artifacts_hands_dir=artifacts_hands_dir,
        results_dir=results_dir,
        background_png=background_png,
        frame_without_hands_png=frame_without_hands_png,
        white_keys_json=white_keys_json,
        black_keys_json=black_keys_json,
        rotation_angle_txt=rotation_angle_txt,
        keys_overlay_png=keys_overlay_png,
        fingertips_raw_csv=fingertips_raw_csv,
        fingertips_tracked_csv=fingertips_tracked_csv,
        landmarks_csv=landmarks_csv,
        pressed_per_finger_csv=pressed_per_finger_csv,
        pressed_per_finger_clean_csv=pressed_per_finger_clean_csv,
        mapping_csv=mapping_csv,
        pressed_keys_csv=pressed_keys_csv,
        vis_frames_dir=vis_frames_dir,
        vis_mp4=vis_mp4,
    )
