"""Get a fresh clone from nothing to a working `ika live`, in one command.

    .venv/bin/python scripts/first_run.py

Why this exists. Four things this project needs are deliberately not in the
repo, and each one announces itself later, separately, as a different error in
a different file: the hand landmarker, the two pose landmarkers, the HaGRID
photographs, and the classifier checkpoint. The README has the commands, but
following four of them by hand is four chances to fetch the wrong thing into
the wrong place, and the trained checkpoint is not a download at all. So this
fetches what can be fetched, trains what cannot, and says plainly what is left.

Three rules it keeps.

It verifies byte counts. `ika/setup_check.py` holds the expected size of every
file with a known one, and a download that does not match it is not put into
place. That is not paranoia: the UT-Interaction host in this project returned
7.5 MB of a 13.2 MB archive with a zero exit code and no error, and a
half-written model fails much later, inside MediaPipe, with a flatbuffer error
that names nothing.

It never destroys anything. Downloads land in a `.part` file next to the
target and are moved into place only once the size checks out, so an
interrupted run cannot leave a plausible looking ruin. A file that is already
there but the wrong size is reported and left alone, because removing it is a
decision for whoever put it there. Unpacking skips any file that already
exists rather than overwriting it.

It is safe to run again. Anything already complete is skipped, not refetched,
so the second run of this script on a working clone downloads nothing and
takes a second.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Importable from source, so this script works before `pip install -e .` has
# been run and can therefore be the thing that tells you to run it.
sys.path.insert(0, str(ROOT))

from ika import setup_check  # noqa: E402
from ika.setup_check import (HAGRID_ANNOTATIONS, HAGRID_ARCHIVE_NAME,
                             HAGRID_IMAGES, HAGRID_URL, HAND_LANDMARKER,
                             POSE_FULL, POSE_LITE, STATIC_CLASSIFIER,
                             Requirement, describe, missing, ready, survey)

# The runtime pieces that pip installs. Checked before anything is downloaded,
# because 384 MB of photographs is a waste of somebody's bandwidth if the
# interpreter running this cannot import torch to train on them.
RUNTIME_MODULES = ("numpy", "torch", "mediapipe", "cv2")
INSTALL_COMMAND = "pip install -e '.[dev]'"

# Downloads are written here first and moved into place only when the size is
# right. The suffix is this script's own scratch space, so it is the one thing
# here that may be deleted without asking.
PART_SUFFIX = ".part"

# -f so an HTTP error is a failure rather than an error page written to disk,
# -L because both hosts redirect, --retry for a dropped connection, and a
# progress bar because a 384 MB fetch with no output looks like a hang.
CURL = ("curl", "-fL", "--retry", "3", "--progress-bar", "-o")

TRAINER = "scripts/train_from_hagrid.py"


def _dependencies_installed() -> bool:
    """Whether the interpreter running this can import what the project needs."""
    absent = [m for m in RUNTIME_MODULES if importlib.util.find_spec(m) is None]
    if not absent:
        return True
    print(f"  {', '.join(absent)} not importable by {sys.executable}")
    print(f"  install the project first:  {INSTALL_COMMAND}")
    return False


def _download(url: str, target: Path) -> bool:
    """Fetch a URL to a scratch file beside the target. True if curl succeeded."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + PART_SUFFIX)
    try:
        subprocess.run([*CURL, str(part), url], check=True)
    except subprocess.CalledProcessError as failure:
        print(f"  download failed, curl exited {failure.returncode}")
        part.unlink(missing_ok=True)
        return False
    return True


def _fetch_file(requirement: Requirement) -> bool:
    """Download one file and put it in place only if it is the right length."""
    target = requirement.path
    part = target.with_name(target.name + PART_SUFFIX)
    expected = requirement.bytes_expected
    size_note = f" ({expected / 1e6:.1f} MB)" if expected is not None else ""
    print(f"  fetching {requirement.name}{size_note}")
    if not _download(requirement.url, target):
        return False

    size = part.stat().st_size
    if expected is not None and size != expected:
        # Discard our own scratch file rather than move a broken model into
        # the place where every later failure will look like a code bug.
        part.unlink()
        print(f"  TRUNCATED: got {size} bytes, expected {expected}. Nothing written.")
        return False
    os.replace(part, target)
    print(f"  {requirement.name} in place")
    return True


def _unpack(archive: Path, into: Path) -> bool:
    """Extract an archive without overwriting anything already there."""
    if not zipfile.is_zipfile(archive):
        # What a redirect to a login page or an error body looks like once it
        # has been saved with the name of a zip.
        print(f"  {archive.name} is not a zip archive. Delete it and try again.")
        return False

    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name = Path(member.filename)
            if name.is_absolute() or ".." in name.parts:
                # A third party archive does not get to choose paths outside
                # the folder we asked for.
                print(f"  refusing suspicious archive entry: {member.filename}")
                return False
            if (into / name).exists():
                continue        # never overwrite a file the user already has
            bundle.extract(member, into)
    return True


def _fetch_hagrid(_requirement: Requirement) -> bool:
    """Download the HaGRID subset and unpack it into data/hagrid.

    One zip carries both the annotations and the photographs, so both
    requirements are satisfied by one fetch. Its size is not published
    anywhere, which is why it is checked by opening it as an archive and by
    resurveying afterwards rather than by counting bytes.
    """
    folder = ROOT / "data" / "hagrid"
    archive = folder / HAGRID_ARCHIVE_NAME
    if not archive.exists():
        print("  fetching the HaGRID subset (384 MB, this is the slow one)")
        if not _download(HAGRID_URL, archive):
            return False
        os.replace(archive.with_name(archive.name + PART_SUFFIX), archive)
    else:
        print(f"  {HAGRID_ARCHIVE_NAME} already downloaded, unpacking it")

    print(f"  unpacking into {folder}")
    if not _unpack(archive, folder):
        return False
    print(f"  unpacked. {archive} is no longer needed, delete it to reclaim the space")
    return True


def _train_classifier(_requirement: Requirement) -> bool:
    """Run the trainer, which is the only way this checkpoint can exist.

    Deliberately the real script rather than a copy of it, so the checkpoint
    this produces is the same one the README quotes 98.8% for.
    """
    if not ready_for_training():
        print("  cannot train yet, the HaGRID photographs are not in place")
        return False
    print(f"  training the classifier: {TRAINER}")
    print("  this takes a few minutes and prints its own held-out score\n")
    finished = subprocess.run([sys.executable, TRAINER], cwd=ROOT)
    return finished.returncode == 0


def ready_for_training() -> bool:
    """Whether the dataset the trainer reads is complete."""
    needed = {HAGRID_ANNOTATIONS, HAGRID_IMAGES}
    return not needed & {r.name for r in missing(ROOT)}


# What to do about each requirement. Dispatch is by name rather than by
# guessing from the URL, so a requirement that needs its own handling gets one
# instead of being quietly mishandled by a generic path.
HANDLERS = {
    HAND_LANDMARKER: _fetch_file,
    POSE_LITE: _fetch_file,
    POSE_FULL: _fetch_file,
    HAGRID_ANNOTATIONS: _fetch_hagrid,
    HAGRID_IMAGES: _fetch_hagrid,
    STATIC_CLASSIFIER: _train_classifier,
}


def main() -> int:
    print(f"\n  ika first run\n  {ROOT}\n")
    print(describe(survey(ROOT)))
    print()

    if ready(ROOT):
        print("  nothing to do. Run the app with ./run.sh, which is there\n"
              "  because curses needs a real terminal and cannot start without one.\n")
        return 0

    if not _dependencies_installed():
        return 1

    done: list[str] = []
    refused: list[str] = []
    attempted = set()
    for requirement in missing(ROOT):
        if requirement.wrong_size:
            # Reported, not repaired. Overwriting it would destroy the only
            # evidence of what went wrong, and it may be a file put there on
            # purpose.
            print(f"  {requirement.name}: {requirement.path} is "
                  f"{requirement.bytes_found} bytes, expected "
                  f"{requirement.bytes_expected}.")
            print("  Delete it and run this again. Nothing here deletes your files.")
            refused.append(requirement.name)
            continue

        handler = HANDLERS[requirement.name]
        if handler in attempted:
            # One HaGRID zip satisfies both HaGRID requirements, so its
            # handler runs once however many of them are missing.
            continue
        attempted.add(handler)
        # One name for one action, so the closing summary reads as what was
        # actually done rather than once per requirement it happened to cover.
        label = ("HaGRID annotations and photographs"
                 if handler is _fetch_hagrid else requirement.name)
        if handler(requirement):
            done.append(label)
        else:
            refused.append(label)

    print("\n  ---\n")
    if done:
        print("  done this run:")
        for name in done:
            print(f"    {name}")
        print()
    if refused:
        print("  did not finish:")
        for name in refused:
            print(f"    {name}")
        print()

    left = missing(ROOT)
    print(describe(survey(ROOT)))
    print()
    if left:
        print("  still to do, see each note above for what it stops.\n")
        return 1

    print("  ready. Start the app with:\n\n    ./run.sh\n\n"
          "  which exists because curses has to own a real terminal, so the app\n"
          "  cannot be started from a pipe or a tool call.\n")
    return 0


if __name__ == "__main__":
    if shutil.which("curl") is None:
        print("  curl is not installed, and every download here uses it")
        raise SystemExit(1)
    raise SystemExit(main())
