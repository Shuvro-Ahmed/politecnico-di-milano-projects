import os
import re
import glob
import cv2
import pandas as pd
from tqdm import tqdm
import mediapipe as mp

FRAMES_DIR = "/Users/shuvroahmed/Documents/GitHub/IACV-Project/data/frames/piano_video_1/cleaned"
OUT_DIR = "outputs/taskB"
VIS_DIR = "outputs/taskB/vis_step3_raw"
MAX_FRAMES = None   # set e.g. 50 for quick test, or leave None for all
SAVE_VIS_EVERY = 25 # save one visualization every N frames

FINGERTIPS = {
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}

def frame_number(path: str) -> int:
    # extracts number from "cleaned_frame_123.png"
    m = re.search(r"cleaned_frame_(\d+)\.png", os.path.basename(path))
    return int(m.group(1)) if m else 10**9

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    paths = glob.glob(os.path.join(FRAMES_DIR, "cleaned_frame_*.png"))
    paths = sorted(paths, key=frame_number)
    if MAX_FRAMES is not None:
        paths = paths[:MAX_FRAMES]

    if len(paths) == 0:
        raise FileNotFoundError(f"No frames found in {FRAMES_DIR}")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
    )

    rows = []

    for i, fp in enumerate(tqdm(paths, desc="B1 raw fingertips")):
        img_bgr = cv2.imread(fp)
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)

        fr_name = os.path.basename(fp)
        fr_id = frame_number(fp)

        if not res.multi_hand_landmarks:
            rows.append({"frame_id": fr_id, "frame": fr_name, "detected": 0})
            continue

        # store detections
        for hand_idx, hand_lms in enumerate(res.multi_hand_landmarks):
            handed = None
            if res.multi_handedness and hand_idx < len(res.multi_handedness):
                handed = res.multi_handedness[hand_idx].classification[0].label  # Left/Right

            for finger, lid in FINGERTIPS.items():
                lm = hand_lms.landmark[lid]
                x = float(lm.x * w)
                y = float(lm.y * h)
                rows.append({
                    "frame_id": fr_id,
                    "frame": fr_name,
                    "hand_idx": hand_idx,
                    "handedness": handed,
                    "finger": finger,
                    "x": x,
                    "y": y,
                    "detected": 1,
                })

        # save a sparse visualization
        if (i % SAVE_VIS_EVERY) == 0:
            vis = img_bgr.copy()
            # draw only fingertips from THIS frame
            for r in rows:
                if r.get("frame_id") == fr_id and r.get("detected") == 1:
                    cv2.circle(vis, (int(r["x"]), int(r["y"])), 6, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(VIS_DIR, fr_name), vis)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "b1_fingertips_raw.csv")
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print("Vis samples in:", VIS_DIR)

if __name__ == "__main__":
    main()
