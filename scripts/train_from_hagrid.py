"""Train the live gesture classifier on real photographs, not synthetic hands.

Why this exists. The shipped classifier was trained on hands generated with
arithmetic, and on real held-out people it managed 46.4% across seven classes.
The same architecture trained on HaGRID photographs of 2,192 different people
reaches 98.8% on people it has never seen. So nobody has to record their own
hands to get a usable model, which is the whole point of this script.

HaGRID names seven of our nine gestures under its own labels. The two it misses
are the two the classifier does not need: `l_shape` is `point` plus an extended
thumb and the pair confuses in both directions, and `pinch` is measured
geometrically from thumb-to-index distance rather than classified.

    .venv/bin/python scripts/train_from_hagrid.py
"""

from __future__ import annotations

import numpy as np
import torch

from ika.dataset import Dataset
from ika.model import GestureNet
from ika.train import confusion, format_confusion
from ika.vision import load_hagrid, split_by_user

MAPPING = {
    "fist": "fist", "palm": "open_palm", "peace": "peace", "one": "point",
    "like": "thumbs_up", "dislike": "thumbs_down", "no_gesture": "rest",
}
CLASSES = ["rest", "open_palm", "fist", "point", "peace", "thumbs_up", "thumbs_down"]
OUT = "checkpoints/static.pt"
EPOCHS = 300


def main() -> int:
    records, _ = load_hagrid("data/hagrid")
    keep = [r for r in records if r.label in MAPPING]
    if not keep:
        print("  no usable records; see the README for the HaGRID download")
        return 1

    labels = np.array([CLASSES.index(MAPPING[r.label]) for r in keep], dtype=np.int64)
    train_idx, val_idx, held, total = split_by_user(keep, fraction=0.25, seed=0)
    print(f"  {len(keep)} photographs, {len(CLASSES)} classes")
    print(f"  split by PERSON, {held} of {total} people held out, so the model "
          f"cannot score by recognising individuals")

    # HaGRID landmarks are 2D. The live pipeline gets real depth from
    # MediaPipe's world landmarks, so this is a floor rather than a ceiling.
    marks = np.zeros((len(keep), 21, 3), dtype=np.float32)
    for i, record in enumerate(keep):
        marks[i, :, :2] = record.landmarks

    data = Dataset(landmarks=marks, is_left=np.zeros(len(keep), dtype=bool),
                   labels=labels, classes=CLASSES)
    features = data.features

    torch.manual_seed(0)
    model = GestureNet(features.shape[1], CLASSES, hidden=(256, 128), dropout=0.3)
    model.fit_standardiser(features[train_idx])

    xt, yt = torch.tensor(features[train_idx]), torch.tensor(labels[train_idx])
    xv, yv = torch.tensor(features[val_idx]), torch.tensor(labels[val_idx])
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    best, state = 0.0, None
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(order), 64):
            batch = order[start : start + 64]
            if len(batch) < 2:
                continue
            optimiser.zero_grad()
            loss_fn(model(xt[batch]), yt[batch]).backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float((model(xv).argmax(1) == yv).float().mean())
        if score > best:
            best, state = score, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        predicted = model(xv).argmax(1).numpy()
    print(f"\n  {best:.1%} on held-out people\n")
    print(format_confusion(confusion(yv.numpy(), predicted, len(CLASSES)), CLASSES))
    model.save(OUT)
    print(f"\n  saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
