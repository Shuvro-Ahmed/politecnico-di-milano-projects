from __future__ import annotations

from .paths import resolve_paths


def run_render(video: str) -> None:
    p = resolve_paths(video)

    p.results_dir.mkdir(parents=True, exist_ok=True)

    from src.render import c1_map_fingertips_to_keys as c1
    from src.render import c2_pressed_keys_per_frame as c2
    from src.render import c3_visualize_pressed_keys as c3

    # C1 mapping
    c1.KEYS_WHITE = p.white_keys_json
    c1.KEYS_BLACK = p.black_keys_json
    c1.FINGERTIPS = p.fingertips_tracked_csv
    c1.OUT = p.mapping_csv
    c1.main()

    # C2 pressed keys per frame
    c2.MAPPING = p.mapping_csv
    c2.PRESSED = p.pressed_per_finger_clean_csv
    c2.OUT = p.pressed_keys_csv
    c2.main()

    # C3 visualization
    c3.VIDEO_NAME = video
    c3.FRAMES_DIR = p.frames_clean_dir
    c3.BACKGROUND = p.frame_without_hands_png
    c3.KEYS_WHITE = p.white_keys_json
    c3.KEYS_BLACK = p.black_keys_json
    c3.FINGERTIPS = p.fingertips_tracked_csv
    c3.PRESSED_KEYS = p.pressed_keys_csv

    c3.OUT_DIR = p.results_dir
    c3.OUT_FRAMES = p.vis_frames_dir
    c3.OUT_VIDEO = p.vis_mp4
    c3.OUT_BG_OVERLAY = p.results_dir / "keyboard_overlay.png"

    c3.main()
