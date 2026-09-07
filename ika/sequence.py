"""Two ways to classify a movement, and a fair fight between them.

The hand-crafted route summarises a window into sixteen numbers I chose:
displacement, straightness, peak speed, how much the pose drifted. It encodes
what I believe distinguishes a swipe from a drag.

The learned route is a GRU over the raw per-frame sequence. It gets no such
hints and has to find the structure itself.

Both are built here because the comparison is the interesting part. Hand-picked
features usually win on small data, since they start with the answer, and lose
once there is enough data for a sequence model to discover something better
than what I thought of. Which side of that line a project sits on is worth
knowing rather than guessing, and there is a script below that measures it.

Windows are resampled to a fixed number of steps rather than padded. Gestures
have genuinely different durations, a snap is quicker than a drag, and padding
would leave the model reading trailing zeros as part of the shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .motion import Sample

STEPS = 20
# per frame: displacement 2, velocity 2, pinch 1, reach 5, curls 5
FRAME_DIM = 15

REACH = slice(72, 77)
CURLS = slice(82, 87)


def sequence_tensor(window: list[Sample], steps: int = STEPS) -> np.ndarray:
    """One window as a fixed-length (steps, FRAME_DIM) array.

    Position is relative to the start of the window and divided by hand size,
    so what the model sees is "how far, in hand widths" rather than absolute
    frame coordinates. Absolute position is not a gesture.
    """
    times = np.array([s.at for s in window], dtype=np.float64)
    points = np.stack([s.position for s in window])
    span = max(float(np.median([s.span for s in window])), 1e-4)

    scaled = (points - points[0]) / span
    gaps = np.maximum(np.diff(times, prepend=times[0] - 1e-3), 1e-4)
    velocity = np.vstack([np.zeros((1, 2)), np.diff(scaled, axis=0)]) / gaps[:, None]

    poses = np.stack([s.pose for s in window])
    per_frame = np.hstack(
        [
            scaled,
            velocity,
            np.array([s.pinch for s in window])[:, None],
            poses[:, REACH],
            poses[:, CURLS],
        ]
    ).astype(np.float32)

    # Resample onto a common timeline, so duration becomes a property the model
    # can use rather than an alignment problem it has to solve.
    source = np.linspace(0.0, 1.0, len(window))
    target = np.linspace(0.0, 1.0, steps)
    return np.stack(
        [np.interp(target, source, per_frame[:, c]) for c in range(per_frame.shape[1])],
        axis=1,
    ).astype(np.float32)


class SequenceNet(nn.Module):
    """A GRU over per-frame features. Small: it runs inside the capture loop."""

    def __init__(self, classes: list[str], frame_dim: int = FRAME_DIM, hidden: int = 64):
        super().__init__()
        self.classes = list(classes)
        self.frame_dim = frame_dim
        self.hidden = hidden
        self.register_buffer("mean", torch.zeros(frame_dim))
        self.register_buffer("std", torch.ones(frame_dim))
        self.gru = nn.GRU(frame_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, len(classes)))

    def fit_standardiser(self, x: np.ndarray) -> None:
        flat = x.reshape(-1, x.shape[-1])
        self.mean.copy_(torch.tensor(flat.mean(0), dtype=torch.float32))
        self.std.copy_(torch.tensor(np.maximum(flat.std(0), 1e-3), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, last = self.gru((x - self.mean) / self.std)
        return self.head(last[-1])

    @torch.no_grad()
    def predict(self, sequence: np.ndarray) -> tuple[int, float, np.ndarray]:
        self.eval()
        x = torch.tensor(sequence[None, ...], dtype=torch.float32)
        p = torch.softmax(self(x), dim=1)[0].numpy()
        best = int(p.argmax())
        return best, float(p[best]), p

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "classes": self.classes,
             "frame_dim": self.frame_dim, "hidden": self.hidden},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> SequenceNet:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(blob["classes"], blob["frame_dim"], blob["hidden"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model


def _stratified(labels: np.ndarray, fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * fraction)))
        val.extend(idx[:cut])
        train.extend(idx[cut:])
    return np.array(train), np.array(val)


def train_sequence(
    windows: list[list[Sample]],
    labels: np.ndarray,
    classes: list[str],
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 3e-3,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[SequenceNet, dict]:
    torch.manual_seed(seed)
    x = np.stack([sequence_tensor(w) for w in windows])
    train_idx, val_idx = _stratified(labels, 0.2, seed)

    model = SequenceNet(classes)
    model.fit_standardiser(x[train_idx])

    xt = torch.tensor(x[train_idx])
    yt = torch.tensor(labels[train_idx])
    xv = torch.tensor(x[val_idx])
    yv = torch.tensor(labels[val_idx])

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(xt[batch]), yt[batch])
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            accuracy = float((model(xv).argmax(1) == yv).float().mean())
        if accuracy > best:
            best = accuracy
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and epoch % 20 == 0:
            print(f"    epoch {epoch:>3}  loss {loss.item():.4f}  val {accuracy:.1%}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = model(xv).argmax(1).numpy()

    from .train import confusion

    return model, {
        "accuracy": best,
        "confusion": confusion(yv.numpy(), predicted, len(classes)),
        "classes": classes,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "recall": {
            classes[i]: float(
                confusion(yv.numpy(), predicted, len(classes))[i, i]
                / max(1, (yv.numpy() == i).sum())
            )
            for i in range(len(classes))
        },
    }
