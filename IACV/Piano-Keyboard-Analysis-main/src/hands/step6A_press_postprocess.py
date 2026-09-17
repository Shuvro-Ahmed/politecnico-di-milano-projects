import os
import numpy as np
import pandas as pd

IN_CSV = "outputs/taskB/b2_pressed_per_finger.csv"
OUT_CSV = "outputs/taskB/b2_pressed_per_finger_clean.csv"

MIN_ON_FRAMES = 3   # must be pressed for >=3 frames to count
MIN_OFF_FRAMES = 2  # must be off for >=2 frames to switch off

def run_length_filter(x: np.ndarray, min_on: int, min_off: int) -> np.ndarray:
    """
    NaN-safe debouncing:
    - Apply run-length filtering only inside contiguous non-NaN segments
    - Preserve NaNs as NaNs
    """
    x = x.astype(float)
    y = x.copy()

    n = len(y)
    isnan = np.isnan(y)

    # find contiguous non-NaN segments
    i = 0
    while i < n:
        if isnan[i]:
            i += 1
            continue

        # segment [i:j) is non-NaN
        j = i
        while j < n and not isnan[j]:
            j += 1

        seg = y[i:j].copy().astype(int)  # seg is 0/1 ints only

        # --- remove short ON runs ---
        k = 0
        while k < len(seg):
            if seg[k] == 1:
                t = k
                while t < len(seg) and seg[t] == 1:
                    t += 1
                if (t - k) < min_on:
                    seg[k:t] = 0
                k = t
            else:
                k += 1

        # --- fill short OFF gaps (only if surrounded by 1s inside this segment) ---
        k = 0
        while k < len(seg):
            if seg[k] == 0:
                t = k
                while t < len(seg) and seg[t] == 0:
                    t += 1
                if (t - k) < min_off:
                    left_one = (k - 1 >= 0 and seg[k - 1] == 1)
                    right_one = (t < len(seg) and seg[t] == 1)
                    if left_one and right_one:
                        seg[k:t] = 1
                k = t
            else:
                k += 1

        y[i:j] = seg.astype(float)
        i = j

    # NaNs stay NaNs automatically
    return y

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    df = pd.read_csv(IN_CSV)

    pressed_cols = [c for c in df.columns if c.endswith("_pressed")]
    for c in pressed_cols:
        arr = df[c].values.astype(float)
        df[c] = run_length_filter(arr, MIN_ON_FRAMES, MIN_OFF_FRAMES)

    df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)
    print("Processed pressed columns:", len(pressed_cols))

if __name__ == "__main__":
    main()
