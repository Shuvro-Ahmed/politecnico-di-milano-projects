from __future__ import annotations

import argparse

from .prepare import run_prepare
from .hands import run_hands
from .render import run_render


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="iacv", description="IACV pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare", help="Extract frames, background, and keys")
    p_prepare.add_argument("--video", required=True, help="Video name without extension (e.g. piano_video_1)")
    p_prepare.add_argument("--fps", type=int, default=10)
    p_prepare.add_argument("--maxframes", type=int, default=600)
    p_prepare.add_argument("--deviation", type=float, default=1.0)
    p_prepare.add_argument("--save-sampled", action="store_true")
    p_prepare.add_argument("--manual-roi", action="store_true")

    p_hands = sub.add_parser("hands", help="Extract fingertips and press decisions")
    p_hands.add_argument("--video", required=True)

    p_render = sub.add_parser("render", help="Map fingertips to keys and render results")
    p_render.add_argument("--video", required=True)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "prepare":
        run_prepare(
            video=args.video,
            fps=args.fps,
            maxframes=args.maxframes,
            deviation=args.deviation,
            save_sampled=args.save_sampled,
            manual_roi=args.manual_roi,
        )
        return

    if args.cmd == "hands":
        run_hands(video=args.video)
        return

    if args.cmd == "render":
        run_render(video=args.video)
        return

    raise SystemExit(2)


if __name__ == "__main__":
    main()
