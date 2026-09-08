"""What a fresh clone is missing, reported before anything crashes to find out.

Why this exists. Nothing large is in the repo, and that is deliberate: the
landmarkers are Google releases, the HaGRID subset is 384 MB of someone else's
CC BY-SA photographs, and checkpoints are reproducible from the code, so
`.gitignore` keeps all three out. Good for the repo, bad for the first ten
minutes after a clone, because every absent piece surfaces somewhere else as a
different error. `ika live` raises `ModelMissing` from `hands.py`, `ika compare`
dies on a path inside `vision.py`, `ika watch` fails on a pose model nobody
mentioned, and the classifier is simply never trained. This answers the whole
question in one place, as data, so both a human and a script can act on it.

Two rules here were learned the hard way.

Size is checked, not only existence. A truncated file is worse than an absent
one: it passes every existence check and then fails much further downstream,
where the cause is invisible. The UT-Interaction host in this project returned
7.5 MB of a 13.2 MB archive with no error and a zero exit code, which is why
`scripts/ut_interaction.py` counts bytes and why this does too.

Downloadable and buildable are different problems. A landmarker is a URL. The
classifier checkpoint is the output of training and no URL will ever produce
it, so its `url` is None and its note names the command that makes it.

Nothing in here imports numpy, torch or mediapipe, and nothing in here touches
the network. It has to give a useful answer in exactly the situation where the
rest of the project cannot import or connect at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Resolved against the package, not the working directory, so a survey means
# the same thing from anywhere. `ika/cli.py` picks its defaults the same way.
ROOT = Path(__file__).resolve().parent.parent

# The names are constants because scripts dispatch on them. A caller matching
# on a literal string would break silently the first time wording changed.
HAND_LANDMARKER = "hand landmarker"
POSE_LITE = "pose landmarker lite"
POSE_FULL = "pose landmarker full"
HAGRID_ANNOTATIONS = "HaGRID annotations"
HAGRID_IMAGES = "HaGRID photographs"
STATIC_CLASSIFIER = "gesture classifier"

# These URLs live next to the code that loads each file, in `ika/hands.py` and
# `ika/body.py`. They are repeated rather than imported to keep this module
# free of numpy, which both of those import at module level and which a fresh
# clone may not have installed yet. A test asserts the two copies agree, so the
# duplication cannot drift.
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker"
    "/hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_LITE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
    "/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_FULL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
    "/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

# The HaGRID subset the README attributes and every real number in this project
# was measured on. One zip, unpacked into `data/hagrid`.
HAGRID_URL = (
    "https://huggingface.co/datasets/GestureDetectionConnoisseurs"
    "/hagrid_subsets/resolve/main/hagrid-export_100_images.zip"
)
HAGRID_ARCHIVE_NAME = "export_100.zip"

# Byte counts of the complete files, taken from the copies in this working
# tree. Those copies are known whole rather than assumed whole: MediaPipe
# parses a `.task` flatbuffer when it opens it, so a truncated model fails to
# load, and all three load. The annotations file is the one HaGRID export that
# `ika compare` and `scripts/train_from_hagrid.py` read to 3,109 usable
# records. Every URL above pins a release version (`/float16/1/`) and the
# HaGRID zip is a fixed dataset revision, so the bytes behind them do not move.
HAND_LANDMARKER_BYTES = 7_819_105
POSE_LITE_BYTES = 5_777_746
POSE_FULL_BYTES = 9_398_198
HAGRID_ANNOTATIONS_BYTES = 4_877_228

# The floor for a file whose exact size cannot be known in advance, which here
# means anything produced by training. Zero bytes is what an interrupted write
# leaves behind, and `torch.load` on that raises a corruption error a long way
# from the cause.
MIN_USEFUL_BYTES = 1

# The seven folders `scripts/train_from_hagrid.py` maps onto our vocabulary,
# mirroring its MAPPING. A dataset missing one of these trains a classifier
# missing a gesture, which is better caught here than in a confusion matrix.
HAGRID_TRAINING_CLASSES = (
    "fist", "palm", "peace", "one", "like", "dislike", "no_gesture",
)

IMAGE_SUFFIX = "*.jpg"


@dataclass(frozen=True)
class Requirement:
    """One thing a fresh clone needs, and what happens without it.

    `present` is the only question a caller usually asks, and it means complete
    rather than merely there: a file of the wrong size is not present, because
    treating it as present is how a truncated download gets to fail later and
    somewhere else.

    `bytes_found` is not part of the question, it is the evidence, and it is
    here so a wrong-size file can be reported precisely enough for someone to
    decide to delete it. Nothing in this project deletes it for them.
    """

    name: str
    path: Path
    url: str | None           # None means it cannot be downloaded, only built
    present: bool
    bytes_expected: int | None
    note: str                 # what breaks without it, in plain words
    bytes_found: int | None = None   # actual size on disk, None if absent

    @property
    def exists(self) -> bool:
        """Whether anything is at the path, complete or not."""
        return self.bytes_found is not None or self.path.is_dir()

    @property
    def wrong_size(self) -> bool:
        """Present on disk but the wrong length, so the download failed part way.

        Kept separate from plain absence because the two need opposite
        handling: an absent file should be fetched, a wrong-size one must not
        be, since re-fetching over it would destroy the evidence and could
        just as easily be a file the user put there on purpose.
        """
        return (
            self.bytes_found is not None
            and self.bytes_expected is not None
            and self.bytes_found != self.bytes_expected
        )


def _file(
    name: str,
    path: Path,
    url: str | None,
    bytes_expected: int | None,
    note: str,
) -> Requirement:
    """Build a requirement for a single file, checking length where known."""
    found = path.stat().st_size if path.is_file() else None
    if found is None:
        complete = False
    elif bytes_expected is not None:
        # Exact equality, not a minimum. A file longer than expected is as
        # wrong as a short one: it is not the file this project was measured
        # against, and guessing which direction is safe helps nobody.
        complete = found == bytes_expected
    else:
        complete = found >= MIN_USEFUL_BYTES
    return Requirement(
        name=name, path=path, url=url, present=complete,
        bytes_expected=bytes_expected, note=note, bytes_found=found,
    )


def _photographs(folder: Path) -> Requirement:
    """Build the requirement for the unpacked image tree.

    No expected size: the tree is thousands of jpegs whose total bytes are a
    property of the export rather than a number worth pinning, and a partial
    unzip shows up as a missing class folder long before it shows up as a byte
    count. So completeness here means every gesture the trainer needs actually
    has photographs in it.
    """
    complete = all(
        any((folder / label).glob(IMAGE_SUFFIX)) for label in HAGRID_TRAINING_CLASSES
    )
    return Requirement(
        name=HAGRID_IMAGES,
        path=folder,
        url=HAGRID_URL,
        present=complete,
        bytes_expected=None,
        note=("scripts/train_from_hagrid.py finds no usable records, so the "
              "classifier that ika live loads can only be trained on invented "
              "hands, which scores 46.4% on real people instead of 98.8%"),
    )


def survey(root: Path | None = None) -> list[Requirement]:
    """Look at the tree and report what is there, in the order to fix it.

    Reads only. It stats files and lists two directories, and it neither
    creates nor removes anything, so it is safe to call from anywhere,
    including from inside a failing command.
    """
    base = Path(root) if root is not None else ROOT
    models = base / "models"
    hagrid = base / "data" / "hagrid"

    return [
        _file(
            HAND_LANDMARKER, models / "hand_landmarker.task", HAND_MODEL_URL,
            HAND_LANDMARKER_BYTES,
            "ika live and ika record raise ModelMissing before the camera opens",
        ),
        _file(
            POSE_LITE, models / "pose_landmarker_lite.task", POSE_LITE_URL,
            POSE_LITE_BYTES,
            "ika watch and ika live --body cannot read a body at all",
        ),
        _file(
            POSE_FULL, models / "pose_landmarker_full.task", POSE_FULL_URL,
            POSE_FULL_BYTES,
            "ika watch --variant full has no model, so the lite against full "
            "trade cannot be measured, only guessed at",
        ),
        _file(
            HAGRID_ANNOTATIONS, hagrid / "annotations.json", HAGRID_URL,
            HAGRID_ANNOTATIONS_BYTES,
            "ika compare and scripts/train_from_hagrid.py stop on a missing "
            "path inside ika/vision.py",
        ),
        _photographs(hagrid),
        _file(
            STATIC_CLASSIFIER, base / "checkpoints" / "static.pt", None, None,
            "ika live cannot start, it prints no model and exits. Build it "
            "with: .venv/bin/python scripts/train_from_hagrid.py",
        ),
    ]


def missing(root: Path | None = None) -> list[Requirement]:
    """Just the incomplete ones, absent and wrong-size alike."""
    return [r for r in survey(root) if not r.present]


def ready(root: Path | None = None) -> bool:
    """Whether every command in the README quick start can actually run."""
    return not missing(root)


def _where(path: Path) -> str:
    """Path relative to the repo when it is inside it, since that is shorter."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


STATE_WIDTH = 10
NAME_WIDTH = 22


def describe(requirements: list[Requirement]) -> str:
    """Render a survey for a person to read.

    Every incomplete requirement gets three things, because each answers a
    question the reader is about to ask anyway: what breaks, and then either
    where to get it or what to run to build it. A wrong-size file gets told
    plainly that the fix is to delete it, since nothing in this project will.
    """
    lines: list[str] = []
    for r in requirements:
        if r.present:
            state = "ok"
        elif r.wrong_size:
            state = "WRONG SIZE"
        else:
            state = "MISSING"
        lines.append(f"  {state:<{STATE_WIDTH}} {r.name:<{NAME_WIDTH}} {_where(r.path)}")
        if r.present:
            continue
        pad = " " * (STATE_WIDTH + NAME_WIDTH + 4)
        lines.append(f"{pad}without it: {r.note}")
        if r.wrong_size:
            lines.append(
                f"{pad}found {r.bytes_found} bytes, expected {r.bytes_expected}. "
                f"Delete it yourself and run again, nothing here removes your files."
            )
        elif r.url is not None:
            lines.append(f"{pad}fetch: {r.url}")

    short = [r for r in requirements if not r.present]
    if short:
        lines.append("")
        lines.append(f"  {len(short)} of {len(requirements)} still needed")
    else:
        lines.append("")
        lines.append(f"  all {len(requirements)} present")
    return "\n".join(lines)
