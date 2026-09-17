from __future__ import annotations

"""Encoder and decoder for the challenge q8rle mask format.

The submission stores each 2D mask as run-length encoded uint8 values. Scores are
quantized from [0, 1] to [0, 255] before encoding, matching the competition
format expected by the leaderboard.
"""

import numpy as np


def float_matrix_to_q8rle(x: np.ndarray) -> str:
    """Encode a 2D float score map in [0, 1] using challenge q8rle."""
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255), 0, 255).astype(np.uint8)
    h, w = q.shape
    flat = q.T.reshape(-1)
    if flat.size == 0:
        return f"q8rle {h} {w}"

    cuts = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, flat.size]
    parts = ["q8rle", str(h), str(w)]
    for value, run_length in zip(flat[starts], ends - starts):
        parts.extend([str(int(value)), str(int(run_length))])
    return " ".join(parts)


def q8rle_to_float_matrix(s: str) -> np.ndarray:
    """Decode a q8rle string into a 2D float score map in [0, 1]."""
    tokens = s.split()
    if len(tokens) < 3 or tokens[0] != "q8rle":
        raise ValueError("Invalid q8rle string")

    h, w = int(tokens[1]), int(tokens[2])
    vals = np.array(list(map(int, tokens[3::2])), dtype=np.uint8)
    lens = np.array(list(map(int, tokens[4::2])), dtype=np.int64)
    if vals.size == 0:
        return np.zeros((h, w), dtype=np.float32)

    flat = np.repeat(vals, lens)
    if flat.size != h * w:
        raise ValueError(f"Invalid run lengths: expected {h * w}, got {flat.size}")
    return flat.reshape(w, h).T.astype(np.float32) / 255.0
