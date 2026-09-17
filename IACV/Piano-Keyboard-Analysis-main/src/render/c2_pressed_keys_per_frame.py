from pathlib import Path

import numpy as np
import pandas as pd

MAPPING = Path("outputs/taskC/c1_fingertips_to_keys.csv")
PRESSED = Path("outputs/taskB/b2_pressed_per_finger_clean.csv")
OUT = Path("outputs/taskC/c2_pressed_keys_per_frame.csv")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(MAPPING)
    p = pd.read_csv(PRESSED)

    m = m.set_index("frame_id")
    p = p.set_index("frame_id")

    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    hands = ["left", "right"]

    rows = []
    for frame_id in sorted(set(m.index).union(p.index)):
        pressed_keys = []
        row = {"frame_id": int(frame_id)}

        for h in hands:
            for f in fingers:
                key_col = f"{h}_{f}_key"
                pr_col = f"{h}_{f}_pressed"
                out_col = f"{h}_{f}_pressed_key"

                k = m[key_col].get(frame_id, np.nan) if key_col in m.columns else np.nan
                pr = p[pr_col].get(frame_id, np.nan) if pr_col in p.columns else np.nan

                if (not pd.isna(k)) and (not pd.isna(pr)) and float(pr) == 1.0:
                    k = str(k)
                    row[out_col] = k
                    pressed_keys.append(k)
                else:
                    row[out_col] = np.nan

        row["pressed_keys"] = ";".join(sorted(set(pressed_keys)))
        rows.append(row)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
