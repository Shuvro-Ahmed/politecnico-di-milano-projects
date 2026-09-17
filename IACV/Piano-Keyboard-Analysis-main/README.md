# IACV-Project

## Run (3 commands)

Run all commands from the project root.

### 1) Prepare (frames + frame-without-hands + keys)

```bash
python -m iacv prepare --video piano_video_1
```

Expected outputs:
- `data/frames/piano_video_1/cleaned/cleaned_frame_*.png`
- `artifacts/piano_video_1/keys/frame_without_hands.png`
- `artifacts/piano_video_1/keys/white_keys.json`
- `artifacts/piano_video_1/keys/black_keys.json`
- `artifacts/piano_video_1/keys/keys_overlay.png`

Optional args:
- `--fps` (default `10`)
- `--maxframes` (default `600`)
- `--deviation` (default `1.0`)
- `--manual-roi` (only if you want to manually select ROI)

### 2) Hands (fingertips + press)

```bash
python -m iacv hands --video piano_video_1
```

Expected outputs:
- `artifacts/piano_video_1/hands/fingertips_tracked.csv`
- `artifacts/piano_video_1/hands/landmarks_for_press.csv`
- `artifacts/piano_video_1/hands/pressed_per_finger_clean.csv`

### 3) Render (mapping + final results)

```bash
python -m iacv render --video piano_video_1
```

Expected outputs:
- `results/piano_video_1/fingertips_to_keys.csv`
- `results/piano_video_1/pressed_keys_per_frame.csv`
- `results/piano_video_1/visualization.mp4`

Notes:
- Input video must exist at `data/raw_videos/<video>.mp4`.
