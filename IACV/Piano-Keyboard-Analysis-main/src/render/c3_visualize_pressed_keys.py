import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VIDEO_NAME = "piano_video_1"  # change if needed
FRAMES_DIR = Path(f"data/frames/{VIDEO_NAME}/cleaned")
BACKGROUND = Path("outputs/frame_without_hands.png")

KEYS_WHITE = Path("outputs/keys/white_keys.json")
KEYS_BLACK = Path("outputs/keys/black_keys.json")
FINGERTIPS = Path("outputs/taskB/b1_fingertips_tracked.csv")
PRESSED_KEYS = Path("outputs/taskC/c2_pressed_keys_per_frame.csv")

OUT_DIR = Path("outputs/taskC")
OUT_FRAMES = OUT_DIR / "c3_vis_frames"
OUT_VIDEO = OUT_DIR / "c3_overlay.mp4"
OUT_BG_OVERLAY = OUT_DIR / "c3_keyboard_overlay.png"


def _load_boxes(fp: Path, color: str) -> dict[str, tuple[int, int, int, int]]:
    d = json.loads(fp.read_text(encoding="utf-8"))
    out = {}
    for key, b in d.items():
        x1, y1, x2, y2 = int(b["x_ul"]), int(b["y_ul"]), int(b["x_dr"]), int(b["y_dr"])
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        out[str(key)] = (x1, y1, x2, y2)
    return out


def _draw_keys(img, whites, blacks, pressed: set[str]):
    overlay = img.copy()

    for key, (x1, y1, x2, y2) in whites.items():
        if key in pressed:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for key, (x1, y1, x2, y2) in blacks.items():
        if key in pressed:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, dst=img)


def _parse_frame_id(name: str) -> int | None:
    m = re.search(r"cleaned_frame_(\d+)\.png$", name)
    return int(m.group(1)) if m else None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FRAMES.mkdir(parents=True, exist_ok=True)

    whites = _load_boxes(KEYS_WHITE, "white")
    blacks = _load_boxes(KEYS_BLACK, "black")

    # Always create a quick key-box sanity overlay on the existing background image.
    # TODO(refactor): align naming/usage with frame_without_hands2.png used elsewhere in the repo.
    if BACKGROUND.exists():
        bg = cv2.imread(str(BACKGROUND))
        if bg is not None:
            bg2 = bg.copy()
            _draw_keys(bg2, whites, blacks, pressed=set())
            cv2.imwrite(str(OUT_BG_OVERLAY), bg2)

    tips = pd.read_csv(FINGERTIPS).set_index("frame_id")
    pressed_df = pd.read_csv(PRESSED_KEYS).set_index("frame_id")

    def _sort_key(p: Path):
        fid = _parse_frame_id(p.name)
        return (fid is None, fid if fid is not None else 10**18)

    files = sorted([p for p in FRAMES_DIR.glob("cleaned_frame_*.png")], key=_sort_key)
    if not files:
        raise FileNotFoundError(f"No frames found in {FRAMES_DIR}")

    first = cv2.imread(str(files[0]))
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))

    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    hands = ["left", "right"]

    for fp in files:
        fid = _parse_frame_id(fp.name)
        if fid is None:
            continue

        img = cv2.imread(str(fp))
        if img is None:
            continue

        pressed = set()
        if fid in pressed_df.index:
            s = str(pressed_df.loc[fid].get("pressed_keys", ""))
            pressed = set([k for k in s.split(";") if k])

        _draw_keys(img, whites, blacks, pressed)

        # fingertips (optional)
        if fid in tips.index:
            r = tips.loc[fid]
            for hnd in hands:
                for f in fingers:
                    xk, yk = f"{hnd}_{f}_tip_x", f"{hnd}_{f}_tip_y"
                    if xk in tips.columns and yk in tips.columns:
                        x, y = r.get(xk, np.nan), r.get(yk, np.nan)
                        if not pd.isna(x) and not pd.isna(y):
                            cv2.circle(img, (int(float(x)), int(float(y))), 6, (255, 0, 0), -1)

        cv2.putText(img, f"frame_id={fid}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"pressed={';'.join(sorted(pressed))}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_fp = OUT_FRAMES / fp.name
        cv2.imwrite(str(out_fp), img)
        vw.write(img)

    vw.release()
    print(f"Wrote: {OUT_VIDEO}")


if __name__ == "__main__":
    main()
