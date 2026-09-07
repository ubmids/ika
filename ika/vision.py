"""Fine-tuning a pretrained vision backbone on hand crops, and racing it
against the landmark classifier on identical data.

This is the comparison the project is really about. The landmark route throws
the image away and keeps 21 points; the backbone route throws the points away
and keeps the pixels. Both are defensible, both are used in real products, and
the interesting question is which actually wins on a laptop.

Three details decide whether the answer means anything:

**Split by person, not by image.** HaGRID gives a `user_id`, and the same
person appears in many photos. Split randomly and the same hands land in both
train and validation, so the model is rewarded for recognising *people* and
the accuracy is inflated. Grouping by user is the difference between a number
you can quote and one you cannot.

**Pick the right hand.** An image labelled `peace` often contains a second hand
labelled `no_gesture`. Taking the first box would label that other hand as the
gesture, quietly poisoning a fraction of every class.

**HaGRID landmarks are 2D.** The z coordinate our own pipeline gets from
MediaPipe's world landmarks is simply absent here, so the landmark side of this
comparison runs with less information than it does live. That handicap is worth
stating rather than hiding: it makes the landmark result a floor, not a ceiling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Record:
    path: Path
    label: str
    bbox: tuple[float, float, float, float]   # normalised x, y, w, h
    landmarks: np.ndarray | None              # (21, 2) normalised, or None
    user_id: str


def load_hagrid(
    root: str | Path,
    classes: list[str] | None = None,
    require_landmarks: bool = True,
) -> tuple[list[Record], list[str]]:
    """Read the exported subset into flat records.

    Keeps only the box whose label matches the folder the image sits in.
    `require_landmarks` drops the ~8% of hands with no landmark annotation, so
    the backbone and the landmark classifier are compared on exactly the same
    examples. Without it the backbone would get 3,391 images and the landmark
    model 3,109, and the difference between them would be partly a difference
    in dataset.
    """
    root = Path(root)
    annotations = json.loads((root / "annotations.json").read_text())

    available = sorted(p.name for p in root.iterdir() if p.is_dir())
    classes = classes or available

    records: list[Record] = []
    for label in classes:
        entries = annotations.get(label, {})
        for image_id, entry in entries.items():
            path = root / label / f"{image_id}.jpg"
            if not path.exists():
                continue
            labels = entry.get("labels") or []
            if label not in labels:
                continue
            which = labels.index(label)          # the hand actually performing it
            boxes = entry.get("bboxes") or []
            if which >= len(boxes):
                continue
            marks = entry.get("hand_landmarks") or []
            landmarks = (
                np.array(marks[which], dtype=np.float32)
                if which < len(marks) and marks[which]
                else None
            )
            if require_landmarks and landmarks is None:
                continue
            records.append(
                Record(
                    path=path,
                    label=label,
                    bbox=tuple(boxes[which]),
                    landmarks=landmarks,
                    # `or` not a default: the key exists with a null value
                    # on some entries, which would break the group split.
                    user_id=entry.get("user_id") or image_id,
                )
            )
    return records, classes


def split_by_user(records: list[Record], fraction: float = 0.25, seed: int = 0):
    """Group split: a person is wholly in train or wholly in validation.

    Without this the same hands appear on both sides and every accuracy below
    would be measuring memory of individuals rather than knowledge of gestures.
    """
    users = sorted({r.user_id for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    held = set(users[: max(1, int(round(len(users) * fraction)))])
    train = [i for i, r in enumerate(records) if r.user_id not in held]
    val = [i for i, r in enumerate(records) if r.user_id in held]
    return np.array(train), np.array(val), len(held), len(users)


def crop_box(width: int, height: int, bbox, pad: float = 0.25):
    """Pixel crop around a normalised box, padded and squared off.

    Padding matters: a tight box cuts the wrist off, and the angle of the wrist
    is part of what distinguishes several of these gestures. Squaring avoids
    the aspect-ratio squash that a plain resize to 128x128 would apply.
    """
    x, y, w, h = bbox
    cx, cy = (x + w / 2) * width, (y + h / 2) * height
    side = max(w * width, h * height) * (1.0 + 2.0 * pad)
    half = side / 2.0
    left = int(max(0, cx - half))
    top = int(max(0, cy - half))
    right = int(min(width, cx + half))
    bottom = int(min(height, cy + half))
    return left, top, right, bottom


def load_crops(records: list[Record], size: int = 128) -> np.ndarray:
    """Every hand crop as one uint8 array, (n, size, size, 3).

    Cached in memory as uint8 rather than float: 3,400 crops at 128px is about
    170 MB this way and would be 670 MB as float32, and the conversion per
    batch costs nothing.
    """
    import cv2

    out = np.zeros((len(records), size, size, 3), dtype=np.uint8)
    for i, record in enumerate(records):
        image = cv2.imread(str(record.path))
        if image is None:
            continue
        h, w = image.shape[:2]
        left, top, right, bottom = crop_box(w, h, record.bbox)
        patch = image[top:bottom, left:right]
        if patch.size == 0:
            continue
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
        out[i] = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    return out


def landmark_features(records: list[Record]) -> np.ndarray:
    """Our own feature extractor, on HaGRID's 2D landmarks.

    Padded to three dimensions with a zero z, which is the honest thing to do:
    the depth our live pipeline gets from MediaPipe's world landmarks is not in
    this dataset, so the palm basis here is built inside the image plane.
    """
    from . import features

    out = np.zeros((len(records), features.FEATURE_DIM), dtype=np.float32)
    for i, record in enumerate(records):
        if record.landmarks is None:
            continue
        marks = np.zeros((21, 3), dtype=np.float64)
        marks[:, :2] = record.landmarks
        out[i] = features.extract(marks)
    return out


def build_backbone(name: str, n_classes: int):
    """A pretrained classifier with its head swapped for ours."""
    import torchvision
    from torch import nn

    if name == "mobilenet_v3_small":
        model = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_classes)
    elif name == "resnet18":
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    else:
        raise ValueError(f"unknown backbone {name!r}")
    return model


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(crops: np.ndarray, augment: bool = False, rng=None):
    """uint8 HWC crops to normalised NCHW float, optionally jittered.

    Augmentation is deliberately mild and never mirrors horizontally: several
    HaGRID classes are distinguished by orientation, and `peace` versus
    `peace_inverted` is exactly the pair a random flip would merge.
    """
    import torch

    x = crops.astype(np.float32) / 255.0
    if augment and rng is not None:
        x = x * rng.uniform(0.75, 1.25, (len(x), 1, 1, 1)).astype(np.float32)
        x = np.clip(x + rng.normal(0, 0.02, x.shape).astype(np.float32), 0.0, 1.0)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


def pick_device(prefer: str | None = None) -> str:
    import torch

    if prefer:
        return prefer
    return "mps" if torch.backends.mps.is_available() else "cpu"


def finetune(
    crops: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    backbone: str = "mobilenet_v3_small",
    epochs: int = 18,
    batch_size: int = 48,
    learning_rate: float = 6e-4,
    device: str | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[object, dict]:
    """Fine-tune a pretrained backbone on hand crops.

    Whole-network fine-tuning rather than a frozen backbone with a new head.
    ImageNet features transfer to hands reasonably well, but a tightly cropped
    hand at 128px is far enough from ImageNet's subject matter that letting the
    early layers move is worth the extra epochs. A low learning rate keeps the
    pretrained weights from being washed away by 2,300 images.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device(device)

    model = build_backbone(backbone, len(classes)).to(device)
    y_train = torch.tensor(labels[train_idx], dtype=torch.long, device=device)
    y_val = torch.tensor(labels[val_idx], dtype=torch.long, device=device)
    val_x = to_tensor(crops[val_idx]).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            if len(batch) < 2:
                continue
            x = to_tensor(crops[train_idx[batch]], augment=True, rng=rng).to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(x), y_train[batch])
            loss.backward()
            optimiser.step()
        schedule.step()

        model.eval()
        with torch.no_grad():
            predicted = torch.cat(
                [model(val_x[i : i + 128]).argmax(1) for i in range(0, len(val_x), 128)]
            )
            accuracy = float((predicted == y_val).float().mean())
        if accuracy > best:
            best = accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"    epoch {epoch:>2}  loss {loss.item():.3f}  val {accuracy:.1%}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval().to(device)
    with torch.no_grad():
        predicted = torch.cat(
            [model(val_x[i : i + 128]).argmax(1) for i in range(0, len(val_x), 128)]
        ).cpu().numpy()

    from .train import confusion

    matrix = confusion(labels[val_idx], predicted, len(classes))
    return model, {
        "accuracy": best,
        "confusion": matrix,
        "classes": classes,
        "device": device,
        "backbone": backbone,
        "params": sum(p.numel() for p in model.parameters()),
        "recall": {
            name: float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else 0.0
            for i, name in enumerate(classes)
        },
    }
