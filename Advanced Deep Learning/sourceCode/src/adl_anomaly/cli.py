from __future__ import annotations

"""Small command-line interface used by the Colab reproduction notebook.

Only the commands needed for the final supervised segmentation result are kept
in this source submission: q8rle smoke test, segmentation training, segmentation
prediction, and submission blending.
"""

import argparse
import csv
import sys
import zipfile
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .q8rle import float_matrix_to_q8rle, q8rle_to_float_matrix


def smoke_q8rle(_args: argparse.Namespace) -> None:
    """Quick check for the submission encoder used by the challenge."""
    x = np.zeros((16, 16), dtype=np.float32)
    x[2:5, 3:9] = 0.5
    x[10:, 10:] = 1.0
    encoded = float_matrix_to_q8rle(x)
    decoded = q8rle_to_float_matrix(encoded)
    assert decoded.shape == x.shape
    # Account for slight precision loss due to 8-bit quantization (255 levels)
    assert np.abs(decoded - np.rint(x * 255) / 255).max() < 1e-6
    print("q8rle smoke test passed")


def train_seg(args: argparse.Namespace) -> None:
    from .segmentation import train_segmentation

    train_segmentation(args)


def evaluate_seg(args: argparse.Namespace) -> None:
    from .segmentation import evaluate_segmentation

    evaluate_segmentation(args)


def predict_seg(args: argparse.Namespace) -> None:
    from .segmentation import predict_segmentation

    predict_segmentation(args)


def _open_submission_csv(path: Path):
    """Open either a plain CSV or a zip containing exactly one CSV."""
    if path.suffix == ".zip":
        zf = zipfile.ZipFile(path)
        csv_names = [name for name in zf.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            zf.close()
            raise ValueError(f"{path} must contain exactly one CSV, found {len(csv_names)}")
        return zf, zf.open(csv_names[0], "r")
    return None, path.open("rb")


def _raise_csv_field_limit() -> None:
    """Allow very long q8rle strings when reading generated submissions."""
    # Mitigate C-level limits in Python's csv parser when dealing with massive run-length encoded strings
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _read_submission(path: Path) -> dict[str, str]:
    _raise_csv_field_limit()
    owner, handle = _open_submission_csv(path)
    try:
        text = (line.decode("utf-8") for line in handle)
        reader = csv.DictReader(text)
        if reader.fieldnames is None:
            raise ValueError(f"{path} must have a header")
        # Support both standard submission naming variants used across the pipeline
        id_col = "ID" if "ID" in reader.fieldnames else "image_id"
        label_col = "Label" if "Label" in reader.fieldnames else "score"
        if id_col not in reader.fieldnames or label_col not in reader.fieldnames:
            raise ValueError(f"{path} must have ID/Label or image_id/score columns")
        rows = {row[id_col]: row[label_col] for row in reader}
    finally:
        handle.close()
        if owner is not None:
            owner.close()
    return rows


def blend_submissions(args: argparse.Namespace) -> None:
    """Average several q8rle submissions pixel-by-pixel and write one CSV."""
    inputs = [Path(path) for path in args.inputs]
    weights = args.weights or [1.0] * len(inputs)
    if len(weights) != len(inputs):
        raise ValueError("--weights must have the same length as --inputs")

    # Normalize weights so they always sum up to exactly 1.0
    weights_arr = np.asarray(weights, dtype=np.float32)
    if np.any(weights_arr < 0) or float(weights_arr.sum()) <= 0:
        raise ValueError("--weights must be non-negative and sum to a positive value")
    weights_arr = weights_arr / weights_arr.sum()

    submissions = [_read_submission(path) for path in inputs]
    image_ids = list(submissions[0].keys())
    expected = set(image_ids)
    for path, rows in zip(inputs[1:], submissions[1:]):
        if set(rows.keys()) != expected:
            raise ValueError(f"{path} has different image_id values")

    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        for image_id in tqdm(image_ids, desc="Blend submissions"):
            blended = None
            for weight, rows in zip(weights_arr, submissions):
                # Decode from RLE format to raw float matrices before performing weighted average
                decoded = q8rle_to_float_matrix(rows[image_id])
                blended = decoded * float(weight) if blended is None else blended + decoded * float(weight)
            # Enforce 0.0 - 1.0 bounds to handle any floating-point arithmetic edge cases before encoding back to RLE
            writer.writerow([image_id, float_matrix_to_q8rle(np.clip(blended, 0.0, 1.0))])

    if args.zip:
        zip_path = out_csv.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_csv, arcname=out_csv.name)
        print(f"Wrote {zip_path}")
    else:
        print(f"Wrote {out_csv}")


def _add_seg_common(p: argparse.ArgumentParser) -> None:
    """Arguments shared by train/evaluate/predict segmentation commands."""
    p.add_argument("--data", default="data", help="Dataset root")
    p.add_argument("--work-dir", default="outputs_seg", help="Directory for segmentation weights and reports")
    p.add_argument(
        "--weights-dir",
        default=None,
        help="Optional fallback directory for prediction/evaluation checkpoints, useful for Drive backups.",
    )
    p.add_argument("--classes", nargs="*", default=None, help="Optional class subset")
    p.add_argument("--seg-models", nargs="+", choices=["A", "B", "S"], default=["A", "B"])
    p.add_argument("--segformer-name", default="nvidia/segformer-b2-finetuned-ade-512-512")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tta", choices=["none", "hflip", "hvflip"], default="hvflip")
    p.add_argument("--tta-scales", nargs="+", type=float, default=[1.0])
    p.add_argument("--fusion", choices=["mean", "uncertainty", "softmax", "geometric"], default="mean")


def _add_seg_postprocess(p: argparse.ArgumentParser) -> None:
    """Prediction-time cleanup parameters for the dense score maps."""
    p.add_argument("--blur-sigma", type=float, default=1.5)
    p.add_argument("--low-cut", type=float, default=0.01)
    p.add_argument("--component-threshold", type=float, default=0.1)
    p.add_argument("--min-component-area", type=int, default=16)
    p.add_argument("--foreground-mask", action="store_true")
    p.add_argument("--foreground-background-weight", type=float, default=0.0)
    p.add_argument("--multiview", action="store_true")
    p.add_argument("--multiview-method", choices=["none", "suppress_low", "reweight", "clip_low"], default="suppress_low")
    p.add_argument("--multiview-threshold", type=float, default=0.01)


def build_parser() -> argparse.ArgumentParser:
    """Build the reduced CLI used in the submitted notebook."""
    parser = argparse.ArgumentParser(description="Spacepresso supervised segmentation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    train_seg_p = sub.add_parser("train-seg")
    _add_seg_common(train_seg_p)
    train_seg_p.add_argument("--epochs", type=int, default=20)
    train_seg_p.add_argument("--seg-loss", choices=["hybrid", "bce_dice"], default="hybrid")
    train_seg_p.add_argument("--lr", type=float, default=3e-4)
    train_seg_p.add_argument("--min-lr", type=float, default=1e-6)
    train_seg_p.add_argument("--weight-decay", type=float, default=1e-3)
    train_seg_p.add_argument("--real-prob", type=float, default=0.20)
    train_seg_p.add_argument("--good-prob", type=float, default=0.15)
    train_seg_p.add_argument("--synthetic-prob", type=float, default=0.25)
    train_seg_p.add_argument("--missing-prob", type=float, default=0.20)
    train_seg_p.add_argument("--copypaste-prob", type=float, default=0.20)
    train_seg_p.add_argument("--epoch-multiplier", type=int, default=1)
    train_seg_p.add_argument(
        "--backup-dir",
        default=None,
        help="Optional directory where each best checkpoint is copied immediately after it is saved.",
    )
    train_seg_p.add_argument(
        "--initial-weights-dir",
        default=None,
        help="Optional directory with previous checkpoints used to initialize matching class/model training runs.",
    )
    train_seg_p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop training after this many epochs without validation improvement; 0 disables it.",
    )
    train_seg_p.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss improvement required to reset early stopping.",
    )
    train_seg_p.set_defaults(func=train_seg)

    eval_seg_p = sub.add_parser("evaluate-seg")
    _add_seg_common(eval_seg_p)
    _add_seg_postprocess(eval_seg_p)
    eval_seg_p.set_defaults(func=evaluate_seg)

    pred_seg_p = sub.add_parser("predict-seg")
    _add_seg_common(pred_seg_p)
    _add_seg_postprocess(pred_seg_p)
    pred_seg_p.add_argument("--output", default="outputs_seg/segmentation_submission.csv")
    pred_seg_p.add_argument("--zip", action="store_true")
    pred_seg_p.set_defaults(func=predict_seg)

    smoke = sub.add_parser("smoke-q8rle")
    smoke.set_defaults(func=smoke_q8rle)

    blend = sub.add_parser("blend-submissions")
    blend.add_argument("--inputs", nargs="+", required=True)
    blend.add_argument("--weights", nargs="+", type=float, default=None)
    blend.add_argument("--output", default="outputs/blended_submission.csv")
    blend.add_argument("--zip", action="store_true")
    blend.set_defaults(func=blend_submissions)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # Route execution to the function attached via set_defaults() above
    args.func(args)


if __name__ == "__main__":
    main()
