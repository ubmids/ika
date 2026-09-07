"""The gesture classifier: features in, gesture out.

Small on purpose. It runs once per frame inside a loop that also has to do
landmark detection and drive a cursor, so a large model would cost frames, and
frames are what make hand control feel alive rather than sluggish.

The standardiser is a buffer inside the model rather than a separate object
saved alongside it. Feature blocks here live on genuinely different scales,
canonical coordinates run to a few units while curls sit in [-1, 1], so
standardising matters, and the classic way to get it wrong is to normalise with
one set of statistics during training and another at inference. Keeping the
statistics in the checkpoint makes that mistake impossible to make.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class GestureNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        classes: list[str],
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.classes = list(classes)
        self.in_dim = in_dim

        # Standardisation, fitted on training data only and carried in the
        # state dict. Buffers, not parameters: they are measured, not learned.
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))

        layers: list[nn.Module] = []
        previous = in_dim
        for width in hidden:
            layers += [nn.Linear(previous, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout)]
            previous = width
        layers.append(nn.Linear(previous, len(classes)))
        self.net = nn.Sequential(*layers)

    def fit_standardiser(self, x: np.ndarray) -> None:
        """Measure feature statistics. Training split only, never validation."""
        self.mean.copy_(torch.tensor(x.mean(axis=0), dtype=torch.float32))
        # Clamped so a feature that never varies cannot divide by ~0 and turn
        # a constant into a huge spurious signal.
        self.std.copy_(torch.tensor(np.maximum(x.std(axis=0), 1e-3), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> tuple[int, float, np.ndarray]:
        """Best class, its probability, and the full distribution.

        The distribution is returned because the state machine needs to know
        how *undecided* the model is, not just what it picked. A confident
        wrong answer and an uncertain right one call for different behaviour.
        """
        self.eval()
        x = torch.tensor(np.atleast_2d(features), dtype=torch.float32)
        probabilities = torch.softmax(self(x), dim=1)[0].numpy()
        best = int(probabilities.argmax())
        return best, float(probabilities[best]), probabilities

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "classes": self.classes,
                "in_dim": self.in_dim,
                "hidden": tuple(
                    m.out_features for m in self.net if isinstance(m, nn.Linear)
                )[:-1],
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> GestureNet:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(blob["in_dim"], blob["classes"], hidden=blob["hidden"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model
