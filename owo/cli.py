"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import control


def _record(args):
    from .record import run_recorder

    run_recorder(out=args.out, seconds=args.seconds, camera=args.camera, width=args.width)
    return 0


def _train(args):
    from .dataset import Dataset, synthetic
    from .train import print_report, train

    if args.synthetic:
        data = synthetic(
            per_class=args.per_class, noise=args.noise,
            curl_jitter=args.curl_jitter, dropout=args.dropout,
        )
        print(f"  {len(data)} synthetic samples, {len(data.classes)} classes")
    else:
        paths = [Path(p) for p in args.data]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"  no such dataset: {missing[0]}\n  record one first: owo record")
            return 1
        data = Dataset.concat([Dataset.load(p) for p in paths])
        print(f"  {len(data)} recorded samples from {len(paths)} session(s)")
        for name, n in data.counts().items():
            print(f"    {name:<14} {n:>5}")

    model, report = train(data, epochs=args.epochs, seed=args.seed)
    print_report(report)
    model.save(args.out)
    print(f"\n  saved {args.out}")
    return 0


def _live(args):
    from .app import run_live

    if not Path(args.checkpoint).exists():
        print(f"  no model at {args.checkpoint}\n  train one first: owo train --synthetic")
        return 1
    if args.live:
        print(
            "\n  This will send real input to your Mac. It needs Accessibility\n"
            "  permission (System Settings > Privacy & Security > Accessibility),\n"
            "  and pynput fails silently without it.\n"
        )
    run_live(
        checkpoint=args.checkpoint, camera=args.camera, width=args.width,
        live_control=args.live, threshold=args.threshold, dwell=args.dwell,
        smoothing=args.smoothing, pointer=not args.no_pointer,
    )
    return 0


def _bindings(_args):
    print("\n  gesture bindings\n")
    for line in control.describe_bindings():
        print(line)
    print("\n  Nothing destructive is bound on purpose: a misfire should cost an")
    print("  annoyance, never lost work.\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="owo", description="Control a computer with bare hands.")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="record your own gestures")
    rec.add_argument("--out", default="data/session.npz")
    rec.add_argument("--seconds", type=float, default=4.0, help="capture length per gesture")
    rec.add_argument("--camera", type=int, default=0)
    rec.add_argument("--width", type=int, default=640)
    rec.set_defaults(func=_record)

    tr = sub.add_parser("train", help="train the gesture classifier")
    tr.add_argument("data", nargs="*", default=["data/session.npz"])
    tr.add_argument("--synthetic", action="store_true", help="train on invented hands instead")
    tr.add_argument("--per-class", type=int, default=400, dest="per_class")
    tr.add_argument("--noise", type=float, default=0.012)
    tr.add_argument("--curl-jitter", type=float, default=0.12, dest="curl_jitter")
    tr.add_argument("--dropout", type=float, default=0.0, help="fraction of landmarks lost")
    tr.add_argument("--epochs", type=int, default=120)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--out", default="checkpoints/static.pt")
    tr.set_defaults(func=_train)

    lv = sub.add_parser("live", help="run it against the webcam")
    lv.add_argument("--checkpoint", default="checkpoints/static.pt")
    lv.add_argument("--live", action="store_true", help="actually drive the machine")
    lv.add_argument("--camera", type=int, default=0)
    lv.add_argument("--width", type=int, default=640)
    lv.add_argument("--threshold", type=float, default=0.80)
    lv.add_argument("--dwell", type=int, default=5, help="frames a gesture must hold")
    lv.add_argument("--smoothing", type=float, default=0.6)
    lv.add_argument("--no-pointer", action="store_true", dest="no_pointer")
    lv.set_defaults(func=_live)

    sub.add_parser("bindings", help="show what each gesture does").set_defaults(func=_bindings)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
