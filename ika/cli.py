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
    _preflight()
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


def _preflight() -> None:
    """Check the landmarker before anything takes over the screen.

    The tracker is built inside the capture loop, and by then curses owns the
    terminal, so a missing-model error gets wiped by the teardown and the user
    sees a bare curses traceback instead of the one command that fixes it.
    """
    from .hands import MODEL, ModelMissing

    if not MODEL.exists():
        raise ModelMissing(MODEL)


def _live(args):
    _preflight()
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
        pointer=not args.no_pointer, commit=not args.confirm,
        body=args.body,
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


def _tell(args):
    """Can we find a habit, and call the next move in time to use it?"""
    import numpy as np

    from .tell import Fighter, Habit, lead_times, mine, summarise
    from .tell.actions import names

    habits = (
        Habit(("jab", "jab"), "drop_guard", 0.70),
        Habit(("low_kick",), "step_back", 0.45),
    )

    if args.null:
        print(f"\n  {args.trials} fighters with NO habits. Anything found is invented.\n")
        counts = [len(mine(names(Fighter(habits=(), seed=1000 + s).sequence(args.actions))))
                  for s in range(args.trials)]
        counts = np.array(counts)
        print(f"    a false find in {100 * (counts > 0).mean():.0f}% of fighters")
        print(f"    {counts.mean():.2f} false finds per fighter on average\n")
        return 0

    train = Fighter(habits=habits, seed=args.seed).sequence(args.actions)
    found = mine(names(train))
    print(f"\n  planted {len(habits)} habits, learned from {args.actions} past actions")
    print(f"  the miner reports {len(found)}:\n")
    for finding in found:
        print(f"    {finding.describe()}")

    unseen = Fighter(habits=habits, seed=args.seed + 92).sequence(args.actions // 2)
    print(f"\n  replayed against {len(unseen)} unseen actions "
          f"({unseen[-1].end / 60:.0f} min of action)\n")
    print(f"  {'recognition delay':>18}  {'right':>6}  {'in time':>8}  "
          f"{'useful':>7}  {'median lead':>12}")
    for cost in (0.0, 0.10, 0.25, 0.40):
        result = summarise(lead_times(unseen, found, detection_cost=cost))
        print(f"  {cost * 1000:>15.0f} ms  {result['precision']:>5.0%}  "
              f"{result['in_time']:>7.0%}  {result['useful']:>6.0%}  "
              f"{result['median_lead'] * 1000:>9.0f} ms")

    gaps = np.array([unseen[i + 1].start - unseen[i].end for i in range(len(unseen) - 1)])
    print(f"\n  the window being shot into: {np.median(gaps) * 1000:.0f} ms median "
          f"between one action ending and the next starting")
    print("  so recognition delay, not prediction quality, is the constraint.\n")
    return 0


def _early(args):
    """How much of a movement do we need to see before we can call it?"""
    import numpy as np
    import torch

    from .model import GestureNet
    from .sequence import _stratified
    from .tell.early import sweep, training_set

    x, y, classes = training_set(per_class=args.per_class,
                                 feint_rate=args.feints, seed=args.seed)
    print(f"\n  {len(y)} prefix samples, {args.feints:.0%} of movements are feints")

    tr, va = _stratified(y, 0.2, args.seed)
    torch.manual_seed(args.seed)
    model = GestureNet(x.shape[1], classes, hidden=(96, 48), dropout=0.25)
    model.fit_standardiser(x[tr])
    xt, yt = torch.tensor(x[tr]), torch.tensor(y[tr])
    xv, yv = torch.tensor(x[va]), torch.tensor(y[va])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    best, state = 0.0, None
    for _ in range(args.epochs):
        model.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(order), 64):
            batch = order[start : start + 64]
            if len(batch) < 2:
                continue
            opt.zero_grad(); loss_fn(model(xt[batch]), yt[batch]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            score = float((model(xv).argmax(1) == yv).float().mean())
        if score > best:
            best, state = score, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    print(f"  held-out accuracy over all prefix lengths: {best:.1%}")
    model.save(args.out)

    for label, rate in ((f"{args.feints:.0%} feints", args.feints), ("no feints", 0.0)):
        print(f"\n  -- against an opponent with {label} --")
        print(f"  {'commit at':>10} {'committed':>10} {'accuracy':>9} "
              f"{'seen':>6} {'latency':>10}")
        for row in sweep(model, classes, per_class=args.trials, feint_rate=rate):
            print(f"  {row['threshold']:>10.2f} {row['committed']:>9.0%} "
                  f"{row['accuracy']:>8.0%} {row['median_fraction']:>5.0%} "
                  f"{row['median_latency'] * 1000:>8.0f} ms")
    print("\n  Certainty is bought with time, and time is what there is none of.\n")
    return 0


def _bodyearly(args):
    """Can a camera read a body early, and what does it fail to see?"""
    import torch

    from .model import GestureNet
    from .sequence import _stratified
    from .tell.bodyearly import sweep, training_set

    for label, occlude in (("legs hidden, as a laptop webcam sees", True),
                           ("legs visible, tripod and a wider shot", False)):
        if args.wide and occlude:
            continue
        x, y, classes = training_set(per_class=args.per_class,
                                     occlude_legs=occlude, seed=args.seed)
        tr, va = _stratified(y, 0.2, args.seed)
        torch.manual_seed(args.seed)
        model = GestureNet(x.shape[1], classes, hidden=(128, 64), dropout=0.25)
        model.fit_standardiser(x[tr])
        xt, yt = torch.tensor(x[tr]), torch.tensor(y[tr])
        xv, yv = torch.tensor(x[va]), torch.tensor(y[va])
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()
        best, state = 0.0, None
        for _ in range(args.epochs):
            model.train()
            order = torch.randperm(len(xt))
            for start in range(0, len(order), 64):
                batch = order[start : start + 64]
                if len(batch) < 2:
                    continue
                opt.zero_grad(); loss_fn(model(xt[batch]), yt[batch]).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                score = float((model(xv).argmax(1) == yv).float().mean())
            if score > best:
                best, state = score, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        model.eval()

        rows = sweep(model, classes, per_class=args.trials, occlude_legs=occlude)
        at = min(rows, key=lambda r: abs(r["threshold"] - 0.85))
        print(f"\n  == {label} ==")
        print(f"  {len(y)} prefix samples, held-out {best:.1%}")
        print(f"  at 0.85 confidence: {at['accuracy']:.0%} accurate, "
              f"{at['median_latency'] * 1000:.0f} ms, "
              f"{at['median_fraction']:.0%} of the movement seen")
        print("  per action:")
        for name, value in sorted(at["per_action"].items(), key=lambda kv: kv[1]):
            print(f"    {name:<14} {value:>5.0%}")
        if occlude:
            model.save(args.out)
    print("\n  Punches and guard survive an occluded camera. Level changes do not.\n")
    return 0


def _watch(args):
    """Run a clip through the whole pipeline and report what it read."""
    from collections import Counter

    import numpy as np

    from .body import MODELS
    from .hands import ModelMissing
    from .watch import watch

    model = MODELS / f"pose_landmarker_{args.variant}.task"
    if not model.exists():
        from .body import POSE_URLS

        raise ModelMissing(model, POSE_URLS[args.variant])
    if not Path(args.clip).exists():
        print(f"  no such clip: {args.clip}")
        return 1

    print(f"\n  {args.clip}")
    seen, identities, drift, compensated, source = [], [], [], 0, "unknown"
    for observation in watch(args.clip, max_width=args.width, stride=args.stride,
                             max_frames=args.max_frames, variant=args.variant,
                             max_bodies=args.bodies, stabilise=not args.no_stabilise):
        seen.append(len(observation.seen))
        identities.append(tuple(sorted(s.id for s in observation.sightings)))
        if observation.camera.confidence > 0.25:
            drift.append(observation.camera.dx)
        compensated += observation.compensated
        source = observation.timeline_source

    if not seen:
        print("  no frames decoded")
        return 1

    print(f"  {len(seen)} frames, timed by {source}")
    print(f"  bodies per frame: " +
          ", ".join(f"{n} in {c}" for n, c in sorted(Counter(seen).items())))
    stable = Counter(identities).most_common(1)[0]
    print(f"  identities: {stable[0]} held for {stable[1]}/{len(seen)} frames")
    print(f"  camera compensated on {compensated}/{len(seen)} frames")
    if drift:
        print(f"  median per-frame camera dx {np.median(drift):+.4f}")
    print()
    return 0


def _drill(args):
    """Watch someone drill and report the habit, then call it as it happens."""
    if args.close:
        _preflight()     # standing back, the hand landmarker is never used
    from .body import MODELS, POSE_URLS
    from .hands import ModelMissing

    from .drill_app import pose_variant, run_drill

    framing = "close" if args.close else "stand"
    variant = pose_variant(framing)
    model = MODELS / f"pose_landmarker_{variant}.task"
    if not model.exists():
        # One command from a fresh clone: fetch what is missing, then drill.
        print("\n  first run: fetching the pose models.")
        import argparse as _argparse
        if _setup(_argparse.Namespace(camera=args.camera, no_camera=True)) != 0:
            raise ModelMissing(model, POSE_URLS[variant])

    source = args.source if args.source is not None else args.camera
    if framing == "stand":
        print("\n  Stand back until the camera sees you from head to hips, hands included.")
    else:
        print("\n  Sit at arm's length. The camera needs your head, shoulders and hands.")
    print(f"  Hold your guard still for {args.calibration:.0f}s while it learns it.")
    print("  Then drill. It reports the habit, not the punches, and once it knows")
    print("  your habit it says so out loud just before you do it.")
    if source == args.camera:
        print("  Press Ctrl-C to end the round; it is summarised and saved either way.")
    print()
    run_drill(camera=source, width=args.width,
              calibration=args.calibration, profile=args.profile,
              subject=args.subject, seconds=args.seconds, framing=framing,
              voice=not args.silent, save=not args.no_save, quiet=args.quiet)
    return 0


def _history(args):
    """Each habit's rate, session by session, so a fading habit is visible."""
    import time as _time
    from pathlib import Path

    from .profile import Profile

    store = Path(args.profile) / f"{args.subject}.json"
    if not store.exists():
        print(f"\n  no sessions for {args.subject} yet. Run `ika drill` first.")
        return 1
    record = Profile.load(store)
    print(f"\n  {args.subject}: {len(record.log)} logged session(s)\n")
    for row in record.log[-args.last:]:
        when = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(row["at"]))
        counts = row["counts"]
        punches = sum(v for k, v in counts.items() if k.startswith("punch_"))
        drops = sum(v for k, v in counts.items() if k.startswith("guard_down_"))
        print(f"  {when}  {row['seconds'] / 60:4.1f} min  {punches:4d} punches  "
              f"{drops:3d} guard drops")
    series = record.history()
    if not series:
        print("\n  no habit has been found yet.")
        return 0
    # Guard habits first: they are the read. Punch patterns are true too, but
    # "after a jab you throw a cross" is how most people box, not a flaw.
    series.sort(key=lambda item: (not item[0].split(" -> ")[1].startswith("guard"), item[0]))
    shown_header = None
    width = max(52, *(len(key) for key, _points in series))
    for key, points in series:
        header = "guard" if key.split(" -> ")[1].startswith("guard") else "punches"
        if header != shown_header:
            print("\n  where your hands go:" if header == "guard" else "\n  punch patterns:")
            shown_header = header
        cells = [f"{h / s:4.0%}" if s else "   -" for _at, h, s in points[-args.last:]]
        rated = [h / s for _a, h, s in points if s]
        trend = ""
        if len(rated) >= 2:
            trend = ("  falling" if rated[-1] < rated[0] - 0.1
                     else "  rising" if rated[-1] > rated[0] + 0.1 else "  steady")
        print(f"    {key:{width}s} {' '.join(cells)}{trend}")
    standing = record.habits(now=_time.time())
    if standing:
        print("\n  standing now:")
        for habit in standing[:6]:
            print(f"    {habit.describe(now=_time.time())}")
    return 0


def _setup(args):
    """Fetch what the drill needs, then check the camera. One command."""
    import urllib.request

    from .body import MODELS, POSE_URLS

    MODELS.mkdir(parents=True, exist_ok=True)
    for variant in ("full", "lite"):
        target = MODELS / f"pose_landmarker_{variant}.task"
        if target.exists() and target.stat().st_size > 1_000_000:
            print(f"  pose model present: {target.name}")
            continue
        print(f"  fetching the {variant} pose model (Google MediaPipe)...")
        partial = target.with_suffix(".part")
        urllib.request.urlretrieve(POSE_URLS[variant], partial)
        # Size, not existence: a truncated download passes every existence
        # check and fails much later, where the cause is invisible.
        if partial.stat().st_size < 1_000_000:
            partial.unlink()
            print("  the download came back truncated. Try again.")
            return 1
        partial.rename(target)
        print(f"  saved {target.name}")

    from .strike import WEIGHTS
    print(f"  punch model {'present' if WEIGHTS.exists() else 'MISSING'}: {WEIGHTS}")
    if not WEIGHTS.exists():
        return 1

    if args.no_camera:
        print("\n  ready. Run `ika drill`.")
        return 0
    import cv2

    capture = cv2.VideoCapture(args.camera)
    ok = capture.isOpened() and capture.read()[0]
    capture.release()
    if not ok:
        print(f"\n  camera {args.camera} did not give a frame. On macOS your terminal "
              "needs camera access:\n  System Settings > Privacy & Security > Camera.")
        return 1
    print(f"  camera {args.camera} works.\n\n  ready. Run `ika drill`.")
    return 0


def _bindings(_args):
    print("\n  gesture bindings\n")
    for line in control.describe_bindings():
        print(line)
    print("\n  Nothing destructive is bound on purpose: a misfire should cost an")
    print("  annoyance, never lost work.\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ika",
        description="Shadowbox in front of a camera; it finds your habit and calls it "
                    "before you do it.")
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
    lv.add_argument("--body", action="store_true",
                    help="also read posture: guard height, elbow extension, "
                         "reach. Costs 9.5 ms a frame")
    lv.add_argument("--confirm", action="store_true",
                    help="wait for a gesture to be held instead of committing "
                         "early. Slower by about 130 ms, measured")
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

    tl = sub.add_parser("tell", help="can we find a habit and call the next move in time?")
    tl.add_argument("--actions", type=int, default=2000, help="length of the history")
    tl.add_argument("--null", action="store_true",
                    help="run against fighters with no habits, to count invented ones")
    tl.add_argument("--trials", type=int, default=30)
    tl.add_argument("--seed", type=int, default=7)
    tl.set_defaults(func=_tell)

    er = sub.add_parser("early", help="how early can we commit to a call?")
    er.add_argument("--per-class", type=int, default=300, dest="per_class")
    er.add_argument("--feints", type=float, default=0.35,
                    help="fraction of movements that lie about their opening")
    er.add_argument("--trials", type=int, default=50)
    er.add_argument("--epochs", type=int, default=200)
    er.add_argument("--seed", type=int, default=0)
    er.add_argument("--out", default=_default("checkpoints", "early.pt"))
    er.set_defaults(func=_early)

    be = sub.add_parser("body", help="can a camera read a body early, and what does it miss?")
    be.add_argument("--per-class", type=int, default=200, dest="per_class")
    be.add_argument("--trials", type=int, default=40)
    be.add_argument("--epochs", type=int, default=180)
    be.add_argument("--wide", action="store_true",
                    help="only the tripod case, with legs visible")
    be.add_argument("--seed", type=int, default=0)
    be.add_argument("--out", default=_default("checkpoints", "body_early.pt"))
    be.set_defaults(func=_bodyearly)

    wa = sub.add_parser("watch", help="run a video clip through the whole pipeline")
    wa.add_argument("clip")
    wa.add_argument("--width", type=int, default=640)
    wa.add_argument("--stride", type=int, default=1)
    wa.add_argument("--max-frames", type=int, default=None, dest="max_frames")
    wa.add_argument("--bodies", type=int, default=2)
    wa.add_argument("--variant", default="lite", choices=("lite", "full"))
    wa.add_argument("--no-stabilise", action="store_true", dest="no_stabilise")
    wa.set_defaults(func=_watch)

    dr = sub.add_parser("drill", help="drill, find the habit, and call it out loud")
    dr.add_argument("source", nargs="?", default=None,
                    help="a video file to replay instead of the camera")
    dr.add_argument("--camera", type=int, default=0)
    dr.add_argument("--close", action="store_true",
                    help="sitting at arm's length rather than standing back")
    dr.add_argument("--width", type=int, default=640)
    dr.add_argument("--calibration", type=float, default=3.0)
    dr.add_argument("--subject", default="me")
    dr.add_argument("--profile", default=_default("data", "profiles"),
                    help="where habits accumulate between sessions")
    dr.add_argument("--seconds", type=float, default=None,
                    help="stop after this long, for a bounded session")
    dr.add_argument("--silent", action="store_true", help="print cues, do not speak them")
    dr.add_argument("--quiet", action="store_true", help="no per-event lines")
    dr.add_argument("--no-save", action="store_true", dest="no_save",
                    help="do not add this session to the profile")
    dr.set_defaults(func=_drill)

    hi = sub.add_parser("history", help="each habit's rate, session by session")
    hi.add_argument("--subject", default="me")
    hi.add_argument("--profile", default=_default("data", "profiles"))
    hi.add_argument("--last", type=int, default=12)
    hi.set_defaults(func=_history)

    se = sub.add_parser("setup", help="fetch the models and check the camera")
    se.add_argument("--camera", type=int, default=0)
    se.add_argument("--no-camera", action="store_true", dest="no_camera")
    se.set_defaults(func=_setup)

    sub.add_parser("bindings", help="show what each gesture does").set_defaults(func=_bindings)
    return parser


def main(argv: list[str] | None = None) -> int:
    # MediaPipe's native logging prints telemetry failures to the terminal in
    # the middle of a session. Quiet it before anything imports mediapipe.
    import os
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        from .hands import ModelMissing

        if isinstance(exc, ModelMissing):
            # A missing model is a setup step, not a crash. Print the fix.
            print(f"\n  {exc}")
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
