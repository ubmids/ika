"""The hand: landmark names, and the gesture vocabulary.

MediaPipe returns 21 points in a fixed order. Naming them once here means the
rest of the codebase never indexes a magic number, and the finger groupings are
what `features` uses to build a canonical frame.
"""

from __future__ import annotations

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

N_LANDMARKS = 21

# Tip and knuckle of each finger, thumb first. Used for extension tests.
FINGERS = {
    "thumb": (THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}
TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# The knuckle row across the palm. It is rigid, unlike anything with a joint in
# it, which is why the canonical frame is built from these three points.
PALM = (WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP)

# The bones, for drawing a hand on screen.
CONNECTIONS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (INDEX_MCP, MIDDLE_MCP),
    (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (MIDDLE_MCP, RING_MCP),
    (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (RING_MCP, PINKY_MCP), (WRIST, PINKY_MCP),
    (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
]

# The static poses we train on. Deliberately small and visually distinct: a
# vocabulary you cannot perform accidentally is worth more than a large one,
# because every gesture here competes with whatever your hands do while you
# think. `rest` and `open_palm` exist so the model has somewhere to put a hand
# that means nothing.
STATIC_GESTURES = [
    "rest",        # hand relaxed, doing nothing in particular
    "open_palm",   # all five extended, palm to camera
    "fist",        # everything curled
    "point",       # index only
    "pinch",       # thumb and index tips together, others curled
    "peace",       # index and middle extended
    "thumbs_up",
    "thumbs_down",
    "l_shape",     # thumb and index extended at a right angle
]

# Gestures that only exist as movement, so a still frame cannot express them.
# These need the temporal model, not the pose classifier.
DYNAMIC_GESTURES = [
    "none",
    "swipe_left",
    "swipe_right",
    "swipe_up",
    "swipe_down",
    "pinch_drag",
    "snap",
]
