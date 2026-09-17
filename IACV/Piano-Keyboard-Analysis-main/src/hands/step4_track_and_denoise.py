import os
import numpy as np
import pandas as pd

IN_CSV = "outputs/taskB/b1_fingertips_raw.csv"
OUT_CSV = "outputs/taskB/b1_fingertips_tracked.csv"

FINGERS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
HANDS = ["Left", "Right"]  # mediapipe labels

SMOOTH_WINDOW = 7  # odd number works best (5,7,9). try 7 first.

def moving_median(series: pd.Series, win: int) -> pd.Series:
    # robust smoothing: removes spikes better than average
    return series.rolling(window=win, center=True, min_periods=1).median()

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    df = pd.read_csv(IN_CSV)

    # Keep only rows that actually have coordinates
    df = df[df["detected"] == 1].copy()

    # Build a clean table with one row per frame_id
    frame_ids = sorted(df["frame_id"].unique().tolist())
    out = pd.DataFrame({"frame_id": frame_ids})

    # helper: for each (hand, finger) pick the best point in that frame
    # (if there are duplicates, we take the first)
    for hand in HANDS:
        for finger in FINGERS:
            sub = df[(df["handedness"] == hand) & (df["finger"] == finger)][["frame_id", "x", "y"]]
            # If multiple entries per frame, keep first
            sub = sub.drop_duplicates(subset=["frame_id"], keep="first").set_index("frame_id")

            out[f"{hand.lower()}_{finger}_x"] = out["frame_id"].map(sub["x"])
            out[f"{hand.lower()}_{finger}_y"] = out["frame_id"].map(sub["y"])

    # ---- DENOISING (simple smoothing) ----
    for col in out.columns:
        if col.endswith("_x") or col.endswith("_y"):
            out[col] = moving_median(out[col], SMOOTH_WINDOW)

    out.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)
    print("Columns:", len(out.columns))
    print("Example columns:", out.columns[:10].tolist())

if __name__ == "__main__":
    main()
