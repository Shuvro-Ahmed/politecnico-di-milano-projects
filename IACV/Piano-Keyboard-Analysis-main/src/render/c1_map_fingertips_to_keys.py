import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS_WHITE = Path("outputs/keys/white_keys.json")
KEYS_BLACK = Path("outputs/keys/black_keys.json")
FINGERTIPS = Path("outputs/taskB/b1_fingertips_tracked.csv")
OUT = Path("outputs/taskC/c1_fingertips_to_keys.csv")

# Pads make mapping robust if key boxes were extracted in a slightly different crop/rotation.
PAD_X = 5
PAD_Y = 250


def _load_boxes(fp: Path, color: str) -> list[dict]:
    d = json.loads(fp.read_text(encoding="utf-8"))
    out = []
    for key_id, b in d.items():
        x1, y1, x2, y2 = float(b["x_ul"]), float(b["y_ul"]), float(b["x_dr"]), float(b["y_dr"])
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        out.append(
            {
                "key": str(key_id),
                "color": color,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2.0,
                "area": (x2 - x1) * (y2 - y1),
            }
        )
    return out


def _contains(b: dict, x: float, y: float) -> bool:
    return (
        (b["x1"] - PAD_X) <= x <= (b["x2"] + PAD_X)
        and (b["y1"] - PAD_Y) <= y <= (b["y2"] + PAD_Y)
    )


def _pick_box(boxes: list[dict], x: float, y: float) -> dict | None:
    inside = [b for b in boxes if _contains(b, x, y)]
    if inside:
        inside.sort(key=lambda b: (b["area"], abs(b["cx"] - x)))
        return inside[0]
    return None


def _nearest_by_x(boxes: list[dict], x: float) -> dict:
    return min(boxes, key=lambda b: abs(b["cx"] - x))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    tips = pd.read_csv(FINGERTIPS)
    whites = _load_boxes(KEYS_WHITE, "white")
    blacks = _load_boxes(KEYS_BLACK, "black")

    # fingertip columns are wide: <hand>_<finger>_tip_x/y
    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    hands = ["left", "right"]

    rows = []
    for _, r in tips.iterrows():
        out = {"frame_id": int(r["frame_id"]) }
        for h in hands:
            for f in fingers:
                xk, yk = f"{h}_{f}_tip_x", f"{h}_{f}_tip_y"
                if xk not in tips.columns or yk not in tips.columns:
                    continue
                x, y = r[xk], r[yk]
                if pd.isna(x) or pd.isna(y):
                    out[f"{h}_{f}_key"] = np.nan
                    continue

                x, y = float(x), float(y)

                # black first, then white; if neither contains, fallback to nearest white by x
                b = _pick_box(blacks, x, y)
                if b is None:
                    b = _pick_box(whites, x, y)
                if b is None:
                    b = _nearest_by_x(whites, x)

                out[f"{h}_{f}_key"] = b["key"] if b else np.nan
        rows.append(out)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    # TODO(refactor): ensure key-box coordinate space matches TaskB fingertip coordinates.
    # Key extraction currently references frame_without_hands2.png but repo contains outputs/frame_without_hands.png.
    main()
