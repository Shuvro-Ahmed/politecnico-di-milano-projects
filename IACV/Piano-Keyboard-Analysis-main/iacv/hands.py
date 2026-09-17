from __future__ import annotations

from .paths import resolve_paths


def run_hands(video: str) -> None:
    p = resolve_paths(video)

    p.artifacts_hands_dir.mkdir(parents=True, exist_ok=True)

    # Import legacy modules (logic unchanged) but override their path constants
    from src.hands import step3_all_frames_fingertips_raw as b1
    from src.hands import step4_track_and_denoise as b1_track
    from src.hands import step5a_extract_landmarks_for_press as b2
    from src.hands import step5B_press_decision as b2_press
    from src.hands import step6A_press_postprocess as b2_clean

    # B1 raw fingertips
    b1.FRAMES_DIR = str(p.frames_clean_dir)
    b1.OUT_DIR = str(p.artifacts_hands_dir)
    b1.VIS_DIR = str(p.artifacts_hands_dir / "vis_fingertips_raw")
    b1.main()

    # B1 tracked
    b1_track.IN_CSV = str(p.artifacts_hands_dir / "b1_fingertips_raw.csv")
    b1_track.OUT_CSV = str(p.fingertips_tracked_csv)
    b1_track.main()

    # B2 landmarks
    b2.FRAMES_DIR = str(p.frames_clean_dir)
    b2.OUT_CSV = str(p.landmarks_csv)
    b2.main()

    # B2 press decision
    b2_press.IN_CSV = str(p.landmarks_csv)
    b2_press.OUT_CSV = str(p.pressed_per_finger_csv)
    b2_press.main()

    # B2 postprocess
    b2_clean.IN_CSV = str(p.pressed_per_finger_csv)
    b2_clean.OUT_CSV = str(p.pressed_per_finger_clean_csv)
    b2_clean.main()
