import os
import cv2
import mediapipe as mp

FRAME_PATH = "/Users/shuvroahmed/Documents/GitHub/IACV-Project/data/frames/piano_video_1/cleaned/cleaned_frame_1.png"
OUT_PATH = "outputs/taskB/step2_one_frame_fingertips.png"

# fingertip landmark indices in MediaPipe hand model
FINGERTIPS = {
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    img_bgr = cv2.imread(FRAME_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read: {FRAME_PATH}")

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
    )

    res = hands.process(img_rgb)

    if not res.multi_hand_landmarks:
        print("No hand detected in this frame.")
        cv2.imwrite(OUT_PATH, img_bgr)
        print("Saved (no detections):", OUT_PATH)
        return

    vis = img_bgr.copy()

    for hand_idx, hand_lms in enumerate(res.multi_hand_landmarks):
        # draw fingertips
        for name, lid in FINGERTIPS.items():
            lm = hand_lms.landmark[lid]
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                vis,
                f"{hand_idx}:{name}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(OUT_PATH, vis)
    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
