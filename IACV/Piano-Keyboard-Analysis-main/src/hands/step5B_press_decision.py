import os
import numpy as np
import pandas as pd

IN_CSV = "outputs/taskB/b2_landmarks_for_press.csv"
OUT_CSV = "outputs/taskB/b2_pressed_per_finger.csv"

HANDS = ["Left", "Right"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

SMOOTH_WIN = 7  # temporal smoothing window (odd number)

def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def moving_median(arr: pd.Series, win: int) -> pd.Series:
    return arr.rolling(window=win, center=True, min_periods=1).median()

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    df = pd.read_csv(IN_CSV)
    df = df[df["detected"] == 1].copy()

    frame_ids = sorted(df["frame_id"].unique().tolist())
    out = pd.DataFrame({"frame_id": frame_ids})

    # We compute bend_score per (hand,finger) and then decide pressed
    for hand in HANDS:
        for finger in FINGERS:
            sub = df[(df["handedness"] == hand) & (df["finger"] == finger)].copy()

            # make one row per frame_id
            sub = sub.drop_duplicates(subset=["frame_id"], keep="first").set_index("frame_id")

            # compute distances
            pip_tip = dist(sub["pip_x"], sub["pip_y"], sub["tip_x"], sub["tip_y"])
            mcp_tip = dist(sub["mcp_x"], sub["mcp_y"], sub["tip_x"], sub["tip_y"])

            # avoid divide-by-zero
            bend = pip_tip / (mcp_tip + 1e-6)

            # align to full frame_id timeline (missing frames become NaN)
            bend_full = out["frame_id"].map(bend)

            # smooth bend score over time
            bend_smooth = moving_median(bend_full, SMOOTH_WIN)

            # auto-threshold: use per-sequence median - a bit (robust baseline)
            # interpretation: "pressed" when bend_score is significantly smaller than typical
            # percentile-based threshold (robust)
            valid = bend_smooth.dropna().values
            if len(valid) == 0:
                pressed = pd.Series(np.nan, index=bend_smooth.index)
            else:
                thresh = np.percentile(valid, 15)  # bottom 15% = most bent
                margin = 0.01  # small; tweak 0.005–0.02 if needed
                pressed = (bend_smooth <= (thresh - margin)).astype(float)
                pressed[bend_smooth.isna()] = np.nan

            out[f"{hand.lower()}_{finger}_bend"] = bend_smooth
            out[f"{hand.lower()}_{finger}_pressed"] = pressed

    out.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)
    print("Columns:", len(out.columns))
    print("Tip: pressed columns end with _pressed (1=pressed, 0=not pressed).")

if __name__ == "__main__":
    main()
