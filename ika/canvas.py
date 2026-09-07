"""A drawing surface made of braille characters.

Terminals give you one character per cell, which is far too coarse to show a
hand. Braille solves it: each character encodes an independent 2x4 grid of
dots, so a 60x20 box of text becomes a 120x80 pixel canvas. That is enough to
recognise your own hand moving, which is the whole point of showing it.

No dependencies, and nothing here knows about hands or curses.
"""

from __future__ import annotations

import math

# Dot bit for each (row, column) inside one character cell.
#   1 4        0x01 0x08
#   2 5   ->   0x02 0x10
#   3 6        0x04 0x20
#   7 8        0x40 0x80
_DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
_BASE = 0x2800


class Braille:
    """A dot canvas, optionally with a layer id per cell for colouring.

    Layers exist so a hand can be drawn as a palm, fingers and fingertips in
    different colours without three passes over three canvases. Each cell
    remembers the highest layer that touched it, so later, more important
    strokes win the colour of a cell they share.
    """

    def __init__(self, columns: int, rows: int):
        self.columns = columns
        self.rows = rows
        self.width = columns * 2       # in dots
        self.height = rows * 4
        self._cells = [[0] * columns for _ in range(rows)]
        self._layers = [[0] * columns for _ in range(rows)]

    def clear(self) -> None:
        for cells, layers in zip(self._cells, self._layers):
            for i in range(len(cells)):
                cells[i] = 0
                layers[i] = 0

    def set(self, x: int, y: int, layer: int = 0) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            cx, cy = x >> 1, y >> 2
            self._cells[cy][cx] |= _DOTS[y & 3][x & 1]
            if layer > self._layers[cy][cx]:
                self._layers[cy][cx] = layer

    def dot(self, x: int, y: int, size: int = 1, layer: int = 0) -> None:
        """A filled disc, for joints, since a single dot is easy to lose."""
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                if dx * dx + dy * dy <= size * size:
                    self.set(x + dx, y + dy, layer)

    def stroke(
        self, x0: float, y0: float, x1: float, y1: float,
        start_width: float = 1.0, end_width: float = 1.0, layer: int = 0,
    ) -> None:
        """A line with thickness, tapering from `start_width` to `end_width`.

        Fingers are thicker at the knuckle than at the tip, and drawing them
        that way is most of the difference between something that reads as a
        hand and something that reads as a stick figure. Width is a half-width
        in dots: each step lays down a short perpendicular run.
        """
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            self.dot(int(round(x0)), int(round(y0)), int(round(start_width)), layer)
            return
        steps = int(length) + 1
        # Unit normal, so the perpendicular run is actually perpendicular
        # rather than axis-aligned.
        nx, ny = -dy / length, dx / length
        for i in range(steps + 1):
            t = i / steps
            cx, cy = x0 + dx * t, y0 + dy * t
            half = start_width + (end_width - start_width) * t
            spread = int(round(half))
            for j in range(-spread, spread + 1):
                self.set(int(round(cx + nx * j)), int(round(cy + ny * j)), layer)

    def polygon(self, points: list[tuple[float, float]], layer: int = 0) -> None:
        """Fill a polygon by scanline.

        Used for the palm. A filled palm is what stops the hand looking like a
        bundle of sticks: it gives the fingers something to belong to.
        """
        if len(points) < 3:
            return
        top = int(min(p[1] for p in points))
        bottom = int(max(p[1] for p in points))
        count = len(points)
        for y in range(top, bottom + 1):
            crossings = []
            for i in range(count):
                ax, ay = points[i]
                bx, by = points[(i + 1) % count]
                if ay == by:
                    continue
                if min(ay, by) <= y < max(ay, by):
                    crossings.append(ax + (bx - ax) * (y - ay) / (by - ay))
            crossings.sort()
            for i in range(0, len(crossings) - 1, 2):
                for x in range(int(crossings[i]), int(crossings[i + 1]) + 1):
                    self.set(x, y, layer)

    def line(self, x0: int, y0: int, x1: int, y1: int, layer: int = 0) -> None:
        """Bresenham, so bones look continuous rather than dotted."""
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx - dy
        while True:
            self.set(x0, y0, layer)
            if x0 == x1 and y0 == y1:
                return
            doubled = error * 2
            if doubled > -dy:
                error -= dy
                x0 += sx
            if doubled < dx:
                error += dx
                y0 += sy

    def text_rows(self) -> list[str]:
        return [
            "".join(chr(_BASE + cell) if cell else " " for cell in row)
            for row in self._cells
        ]

    def runs(self) -> list[list[tuple[int, str, int]]]:
        """Each row as runs of (start column, text, layer).

        Batched rather than per-cell so the caller can colour the drawing with
        a handful of writes per row instead of one per character, which at
        160 columns by 60 rows is the difference between free and a visible
        frame-rate cost.
        """
        out: list[list[tuple[int, str, int]]] = []
        for cells, layers in zip(self._cells, self._layers):
            row: list[tuple[int, str, int]] = []
            start, current, buffer = 0, None, []
            for i, cell in enumerate(cells):
                layer = layers[i] if cell else -1
                if layer != current:
                    if buffer and current is not None and current >= 0:
                        row.append((start, "".join(buffer), current))
                    start, current, buffer = i, layer, []
                if cell:
                    buffer.append(chr(_BASE + cell))
            if buffer and current is not None and current >= 0:
                row.append((start, "".join(buffer), current))
            out.append(row)
        return out
