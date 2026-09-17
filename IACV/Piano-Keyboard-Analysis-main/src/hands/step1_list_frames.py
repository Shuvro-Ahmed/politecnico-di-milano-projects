import glob
import os

FRAMES_DIR = "/Users/shuvroahmed/Documents/GitHub/IACV-Project/data/frames/piano_video_1/cleaned"

paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "cleaned_frame_*.png")))

print("Frames dir:", FRAMES_DIR)
print("Found frames:", len(paths))
print("First 3:")
for p in paths[:3]:
    print(" -", p)
