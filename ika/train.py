"""Training, and reporting the result honestly.

Overall accuracy on a balanced synthetic set is close to meaningless: it can
sit at 95% while one gesture never works at all, and that one gesture is the
one you will try first. So this always prints per-class recall and a confusion
matrix, because what matters for control is *which* pairs get mixed up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dataset import Dataset
from .model import GestureNet


def confusion(true: np.ndarray, predicted: np.ndarray, n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(true, predicted):
        matrix[t, p] += 1
    return matrix


def format_confusion(matrix: np.ndarray, classes: list[str]) -> str:
    width = max(len(c) for c in classes) + 1
    head = " " * (width + 2) + " ".join(f"{c[:4]:>4}" for c in classes)
    lines = [head]
    for i, name in enumerate(classes):
        row = " ".join(f"{v:>4}" for v in matrix[i])
        total = matrix[i].sum()
        recall = matrix[i, i] / total if total else 0.0
        lines.append(f"  {name:<{width}}{row}   {recall:5.1%}")
    return "\n".join(lines)


def train(
    data: Dataset,
    epochs: int = 120,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    hidden: tuple[int, ...] = (128, 64),
    seed: int = 0,
    patience: int = 25,
    verbose: bool = True,
) -> tuple[GestureNet, dict]:
    """Fit a classifier and return it with a report.

    Early stopping keeps the weights from the best validation epoch rather than
    the last, so a model that starts overfitting does not quietly get worse
    while the loss curve still looks busy.
    """
    torch.manual_seed(seed)
    train_set, val_set = data.split(0.2, seed=seed)

    x_train = train_set.features
    x_val = val_set.features
    y_train = torch.tensor(train_set.labels, dtype=torch.long)
    y_val = torch.tensor(val_set.labels, dtype=torch.long)

    model = GestureNet(x_train.shape[1], data.classes, hidden=hidden)
    model.fit_standardiser(x_train)  # train split only: validation must stay unseen

    xt = torch.tensor(x_train, dtype=torch.float32)
    xv = torch.tensor(x_val, dtype=torch.float32)

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_accuracy, best_state, best_epoch, waited = -1.0, None, 0, 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            if len(batch) < 2:
                continue  # BatchNorm needs more than one sample
            optimiser.zero_grad()
            loss = loss_fn(model(xt[batch]), y_train[batch])
            loss.backward()
            optimiser.step()
        schedule.step()

        model.eval()
        with torch.no_grad():
            accuracy = float((model(xv).argmax(1) == y_val).float().mean())
        if accuracy > best_accuracy:
            best_accuracy, best_epoch, waited = accuracy, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:>3}  loss {loss.item():.4f}  val {accuracy:.1%}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        predicted = model(xv).argmax(1).numpy()
    matrix = confusion(y_val.numpy(), predicted, len(data.classes))

    report = {
        "accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "train_size": len(x_train),
        "val_size": len(x_val),
        "confusion": matrix,
        "classes": data.classes,
        "recall": {
            name: (float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else 0.0)
            for i, name in enumerate(data.classes)
        },
    }
    return model, report


def print_report(report: dict) -> None:
    print(f"\n  {report['train_size']} train / {report['val_size']} val")
    print(f"  best epoch {report['best_epoch']}, accuracy {report['accuracy']:.1%}\n")
    print(format_confusion(report["confusion"], report["classes"]))
    worst = min(report["recall"].items(), key=lambda kv: kv[1])
    print(f"\n  weakest gesture: {worst[0]} at {worst[1]:.1%} recall")
