import os
import json
import cv2
import skimage.io

from keys_extraction import KeysExtractorThroughLines


def key_to_dict(k):
    y_ul, x_ul, y_dr, x_dr = k.coords()
    return {"y_ul": int(y_ul), "x_ul": int(x_ul), "y_dr": int(y_dr), "x_dr": int(x_dr), "name": str(k)}

def main():
    os.makedirs("outputs/keys", exist_ok=True)

    # load your chosen background image
    image = skimage.io.imread("outputs/frame_without_hands2.png")

    extractor = KeysExtractorThroughLines()
    white_keys_coords, black_keys_coords, angle = extractor(image)

    # convert to JSON-serializable format
    white_out = {name: key_to_dict(k) for name, k in white_keys_coords.items()}
    black_out = {name: key_to_dict(k) for name, k in black_keys_coords.items()}

    with open("outputs/keys/white_keys.json", "w") as f:
        json.dump(white_out, f, indent=2)

    with open("outputs/keys/black_keys.json", "w") as f:
        json.dump(black_out, f, indent=2)

    with open("outputs/keys/rotation_angle.txt", "w") as f:
        f.write(str(angle))

    print("Saved: outputs/keys/white_keys.json")
    print("Saved: outputs/keys/black_keys.json")
    print("Saved: outputs/keys/rotation_angle.txt")


    with open("outputs/keys/white_keys.json","r") as f:
        d = json.load(f)

    xs = []
    ws = []
    for _, v in d.items():
        x1, x2 = v["x_ul"], v["x_dr"]
        xs += [x1, x2]
        ws.append(x2 - x1)

    print("x range:", min(xs), max(xs))
    print("min width:", min(ws), "median width:", sorted(ws)[len(ws)//2], "max width:", max(ws))


    print("image shape:", image.shape)

    # convert RGBA/RGB to BGR
    if image.shape[2] == 4:
        base = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        base = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    overlay = base.copy()

    # very thin lines for small image
    th = 2

    # draw white keys (green)
    for key in white_keys_coords.values():
        y1, x1, y2, x2 = key.coords()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), th)

    # draw black keys (red)
    for key in black_keys_coords.values():
        y1, x1, y2, x2 = key.coords()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), th)

    # blend to keep keys visible
    alpha = 0.35   # lower = more visible keys
    out = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

    cv2.imwrite("outputs/keys/keys_overlay.png", out)
    print("Saved: outputs/keys/keys_overlay.png")


if __name__ == "__main__":
    main()
