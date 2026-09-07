"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import control

# Defaults resolve against the repo, not the working directory, so `ika live`
# works from anywhere rather than only from the project root.
ROOT = Path(__file__).resolve().parent.parent


def _default(*parts: str) -> str:
    return str(ROOT.joinpath(*parts))


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
            print(f"  no such dataset: {missing[0]}\n  record one first: ika record")
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
    if not Path(args.checkpoint).exists():
        print(f"  no model at {args.checkpoint}\n  train one first: ika train --synthetic")
        return 1
    if args.live:
        print(
            "\n  This will send real input to your Mac. It needs Accessibility\n"
            "  permission (System Settings > Privacy & Security > Accessibility),\n"
            "  and pynput fails silently without it.\n"
        )

    if args.window:
        from .app import run_live

        run_live(
            checkpoint=args.checkpoint, dynamic_checkpoint=args.dynamic,
            camera=args.camera, width=args.width,
            live_control=args.live, threshold=args.threshold, dwell=args.dwell,
            smoothing=args.smoothing, pointer=not args.no_pointer,
        )
        return 0

    if args.stream:
        from .stream import run_stream

        run_stream(
            checkpoint=args.checkpoint, camera=args.camera, width=args.width,
            live_control=args.live, threshold=args.threshold, dwell=args.dwell,
            smoothing=args.smoothing, max_hands=args.hands,
            seconds=args.seconds, pointer=not args.no_pointer,
        )
        return 0

    from .tui import run_terminal

    run_terminal(
        checkpoint=args.checkpoint, camera=args.camera, width=args.width,
        live_control=args.live, threshold=args.threshold, dwell=args.dwell,
        smoothing=args.smoothing, max_hands=args.hands,
        pointer=not args.no_pointer,
    )
    return 0


def _train_dynamic(args):
    from . import trajectory
    from .dynamic import train_motion
    from .train import format_confusion

    vectors, labels, classes, windows = trajectory.realistic_dataset(
        per_class=args.per_class, seed=args.seed
    )
    print(f"  {len(labels)} sliding windows, {len(classes)} classes")
    if args.gru:
        from .sequence import train_sequence

        model, report = train_sequence(windows, labels, classes, epochs=args.epochs,
                                       seed=args.seed, verbose=True)
        print(f"\n  accuracy {report['accuracy']:.1%}")
    else:
        model, report = train_motion(vectors, labels, classes, epochs=args.epochs,
                                     seed=args.seed, verbose=True)
        print(f"\n  accuracy {report['accuracy']:.1%}   "
              f"balanced recall {report['balanced']:.1%}")
    print(format_confusion(report["confusion"], classes))
    print(f"\n  idle-class recall is the number that matters: "
          f"{report['recall'].get('none', 0):.1%}")
    print("  every point below 100% there is a firing you did not ask for.")
    model.save(args.out)
    print(f"  saved {args.out}")
    return 0


def _compare(args):
    """Landmarks against pixels, on the same real images and the same split."""
    import numpy as np
    import torch

    from .model import GestureNet
    from .vision import (finetune, landmark_features, load_crops, load_hagrid,
                         split_by_user)

    if not Path(args.root).exists():
        print(f"  no dataset at {args.root}. See the README for the download.")
        return 1

    records, classes = load_hagrid(args.root)
    labels = np.array([classes.index(r.label) for r in records], dtype=np.int64)
    train_idx, val_idx, held, total = split_by_user(records, fraction=0.25, seed=args.seed)
    print(f"  {len(records)} real hand crops, {len(classes)} classes")
    print(f"  split by PERSON: {len(train_idx)} train / {len(val_idx)} val, "
          f"{held}/{total} users held out\n")

    x = landmark_features(records)
    torch.manual_seed(args.seed)
    mlp = GestureNet(x.shape[1], classes, hidden=(256, 128), dropout=0.3)
    mlp.fit_standardiser(x[train_idx])
    xt, yt = torch.tensor(x[train_idx]), torch.tensor(labels[train_idx])
    xv, yv = torch.tensor(x[val_idx]), torch.tensor(labels[val_idx])
    opt = torch.optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=1e-4)
    lf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    best = 0.0
    for _ in range(args.epochs_mlp):
        mlp.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(order), 64):
            b = order[start:start + 64]
            if len(b) < 2:
                continue
            opt.zero_grad(); lf(mlp(xt[b]), yt[b]).backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            best = max(best, float((mlp(xv).argmax(1) == yv).float().mean()))
    lm_params = sum(p.numel() for p in mlp.parameters())
    print(f"  landmarks -> features -> MLP: {best:.1%}  ({lm_params:,} params)")

    crops = load_crops(records, size=args.size)
    _, rep = finetune(crops, labels, classes, train_idx, val_idx,
                      backbone=args.backbone, epochs=args.epochs_cnn, verbose=True)
    print(f"  {args.backbone}: {rep['accuracy']:.1%}  ({rep['params']:,} params, "
          f"on {rep['device']})")
    return 0


def _bindings(_args):
    print("\n  gesture bindings\n")
    for line in control.describe_bindings():
        print(line)
    print("\n  Nothing destructive is bound on purpose: a misfire should cost an")
    print("  annoyance, never lost work.\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ika", description="Control a computer with bare hands.")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="record your own gestures")
    rec.add_argument("--out", default=_default("data", "session.npz"))
    rec.add_argument("--seconds", type=float, default=4.0, help="capture length per gesture")
    rec.add_argument("--camera", type=int, default=0)
    rec.add_argument("--width", type=int, default=640)
    rec.set_defaults(func=_record)

    tr = sub.add_parser("train", help="train the gesture classifier")
    tr.add_argument("data", nargs="*", default=[_default("data", "session.npz")])
    tr.add_argument("--synthetic", action="store_true", help="train on invented hands instead")
    tr.add_argument("--per-class", type=int, default=400, dest="per_class")
    tr.add_argument("--noise", type=float, default=0.012)
    tr.add_argument("--curl-jitter", type=float, default=0.12, dest="curl_jitter")
    tr.add_argument("--dropout", type=float, default=0.0, help="fraction of landmarks lost")
    tr.add_argument("--epochs", type=int, default=120)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--out", default=_default("checkpoints", "static.pt"))
    tr.set_defaults(func=_train)

    lv = sub.add_parser("live", help="run it against the webcam, in the terminal")
    lv.add_argument("--checkpoint", default=_default("checkpoints", "static.pt"))
    lv.add_argument("--live", action="store_true", help="actually drive the machine")
    lv.add_argument("--camera", type=int, default=0)
    lv.add_argument("--width", type=int, default=640)
    lv.add_argument("--threshold", type=float, default=0.80)
    lv.add_argument("--dwell", type=int, default=5, help="frames a gesture must hold")
    lv.add_argument("--smoothing", type=float, default=0.6)
    lv.add_argument("--no-pointer", action="store_true", dest="no_pointer")
    lv.add_argument("--hands", type=int, default=2, help="how many hands to track")
    lv.add_argument("--stream", action="store_true",
                    help="print events instead of drawing a screen. Works "
                         "anywhere, including through a pipe")
    lv.add_argument("--seconds", type=float, default=None,
                    help="stop after this long (stream mode)")
    lv.add_argument("--window", action="store_true",
                    help="use the OpenCV window instead of the terminal "
                         "(single hand, but shows the camera image)")
    lv.add_argument("--dynamic", nargs="?", const=_default("checkpoints", "dynamic_motion.pt"),
                    default=None, dest="dynamic",
                    help="enable swipes and snaps. Off by default: it fires about "
                         "3 times a minute at an idle hand (see the README)")
    lv.set_defaults(func=_live)

    td = sub.add_parser("train-dynamic", help="train the movement classifier")
    td.add_argument("--per-class", type=int, default=700, dest="per_class")
    td.add_argument("--epochs", type=int, default=250)
    td.add_argument("--gru", action="store_true", help="use the sequence model instead")
    td.add_argument("--seed", type=int, default=0)
    td.add_argument("--out", default=_default("checkpoints", "dynamic_motion.pt"))
    td.set_defaults(func=_train_dynamic)

    cmp = sub.add_parser("compare", help="landmarks vs a fine-tuned backbone, on HaGRID")
    cmp.add_argument("--root", default=_default("data", "hagrid"))
    cmp.add_argument("--backbone", default="mobilenet_v3_small",
                     choices=("mobilenet_v3_small", "resnet18", "efficientnet_b0"))
    cmp.add_argument("--size", type=int, default=128)
    cmp.add_argument("--epochs-cnn", type=int, default=18, dest="epochs_cnn")
    cmp.add_argument("--epochs-mlp", type=int, default=300, dest="epochs_mlp")
    cmp.add_argument("--seed", type=int, default=0)
    cmp.set_defaults(func=_compare)

    sub.add_parser("bindings", help="show what each gesture does").set_defaults(func=_bindings)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
