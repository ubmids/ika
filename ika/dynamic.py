"""The dynamic lane, assembled around the model that actually won.

`sequence.py` builds a GRU over raw per-frame sequences, and a data-efficiency
sweep across three seeds found it never beats the 16 hand-picked motion
features: a tie from 20 examples per class upward, and a loss at 8 per class
(97.6% against 100%). So the shipped dynamic classifier is the small MLP over
`motion.features`, at roughly a quarter of the parameters.

That is the sweep's verdict on synthetic data, and it deserves a caveat: real
swipes will be messier than generated ones, and a sequence model has more
headroom to exploit if the extra mess turns out to carry signal. The GRU stays
in the tree for exactly that reason, and `ika train-dynamic --gru` will train
it, but the default follows the evidence that exists rather than the evidence
someone might wish for.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .model import GestureNet
from .motion import MOTION_DIM


def train_motion(
    vectors: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    epochs: int = 250,
    batch_size: int = 64,
    learning_rate: float = 3e-3,
    hidden: tuple[int, ...] = (64, 32),
    seed: int = 0,
    verbose: bool = False,
) -> tuple[GestureNet, dict]:
    """Fit the motion-feature classifier.

    Class weights are applied because the honest sliding-window dataset is
    heavily skewed toward `none`, which is correct for a live system (most
    windows contain no gesture) but would otherwise let the model score well by
    predicting nothing forever.
    """
    from .sequence import _stratified
    from .train import confusion

    torch.manual_seed(seed)
    train_idx, val_idx = _stratified(labels, 0.2, seed)

    model = GestureNet(vectors.shape[1], classes, hidden=hidden)
    model.fit_standardiser(vectors[train_idx])

    xt = torch.tensor(vectors[train_idx])
    yt = torch.tensor(labels[train_idx])
    xv = torch.tensor(vectors[val_idx])
    yv = torch.tensor(labels[val_idx])

    counts = np.bincount(labels[train_idx], minlength=len(classes)).astype(np.float32)
    weights = torch.tensor(len(train_idx) / np.maximum(counts, 1) / len(classes))

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            if len(batch) < 2:
                continue
            optimiser.zero_grad()
            loss_fn(model(xt[batch]), yt[batch]).backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            predicted = model(xv).argmax(1)
            # Balanced accuracy: mean per-class recall. Plain accuracy would be
            # flattered by the dominant `none` class.
            recalls = [
                float((predicted[yv == c] == c).float().mean())
                for c in range(len(classes))
                if (yv == c).any()
            ]
            score = float(np.mean(recalls))
        if score > best:
            best = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and epoch % 50 == 0:
            print(f"    epoch {epoch:>3}  balanced recall {score:.1%}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = model(xv).argmax(1).numpy()
    matrix = confusion(yv.numpy(), predicted, len(classes))

    return model, {
        "accuracy": float((predicted == yv.numpy()).mean()),
        "balanced": best,
        "confusion": matrix,
        "classes": classes,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "recall": {
            name: float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else 0.0
            for i, name in enumerate(classes)
        },
    }
