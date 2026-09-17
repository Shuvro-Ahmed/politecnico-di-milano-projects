import os
import re
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp

FRAMES_DIR = "/Users/shuvroahmed/Documents/GitHub/IACV-Project/data/frames/piano_video_1/cleaned"
OUT_CSV = "outputs/taskB/b2_landmarks_for_press.csv"
MAX_FRAMES = None  # None = all

# landmark ids for each finger (MediaPipe Hands)
# (mcp, pip, dip, tip)
FINGER_LMS = {
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
    "thumb":  (2, 3, 4, 4),   # thumb is special; keep simple
}

def frame_number(path: str) -> int:
    m = re.search(r"cleaned_frame_(\d+)\.png", os.path.basename(path))
    return int(m.group(1)) if m else 10**9

def lm_to_xy(lm, w, h):
    return float(lm.x * w), float(lm.y * h)

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "cleaned_frame_*.png")), key=frame_number)
    if MAX_FRAMES is not None:
        paths = paths[:MAX_FRAMES]

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
    )

    rows = []

    for fp in tqdm(paths, desc="B2 landmarks"):
        img = cv2.imread(fp)
        if img is None:
            continue
        h, w = img.shape[:2]
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        fr_id = frame_number(fp)

        if not res.multi_hand_landmarks:
            rows.append({"frame_id": fr_id, "detected": 0})
            continue

        for hand_idx, hand_lms in enumerate(res.multi_hand_landmarks):
            handed = None
            if res.multi_handedness and hand_idx < len(res.multi_handedness):
                handed = res.multi_handedness[hand_idx].classification[0].label  # Left/Right

            for finger, (mcp, pip, dip, tip) in FINGER_LMS.items():
                xm, ym = lm_to_xy(hand_lms.landmark[mcp], w, h)
                xp, yp = lm_to_xy(hand_lms.landmark[pip], w, h)
                xd, yd = lm_to_xy(hand_lms.landmark[dip], w, h)
                xt, yt = lm_to_xy(hand_lms.landmark[tip], w, h)

                rows.append({
                    "frame_id": fr_id,
                    "handedness": handed,
                    "finger": finger,
                    "mcp_x": xm, "mcp_y": ym,
                    "pip_x": xp, "pip_y": yp,
                    "dip_x": xd, "dip_y": yd,
                    "tip_x": xt, "tip_y": yt,
                    "detected": 1,
                })

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
