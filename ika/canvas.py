"""A drawing surface made of braille characters.

Terminals give you one character per cell, which is far too coarse to show a
hand. Braille solves it: each character encodes an independent 2x4 grid of
dots, so a 60x20 box of text becomes a 120x80 pixel canvas. That is enough to
recognise your own hand moving, which is the whole point of showing it.

No dependencies, and nothing here knows about hands or curses.
"""

from __future__ import annotations

# Dot bit for each (row, column) inside one character cell.
#   1 4        0x01 0x08
#   2 5   ->   0x02 0x10
#   3 6        0x04 0x20
#   7 8        0x40 0x80
_DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
_BASE = 0x2800


class Braille:
    def __init__(self, columns: int, rows: int):
        self.columns = columns
        self.rows = rows
        self.width = columns * 2       # in dots
        self.height = rows * 4
        self._cells = [[0] * columns for _ in range(rows)]

    def clear(self) -> None:
        for row in self._cells:
            for i in range(len(row)):
                row[i] = 0

    def set(self, x: int, y: int) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self._cells[y >> 2][x >> 1] |= _DOTS[y & 3][x & 1]

    def dot(self, x: int, y: int, size: int = 1) -> None:
        """A blob, for joints, since a single dot is easy to lose."""
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                if dx * dx + dy * dy <= size * size:
                    self.set(x + dx, y + dy)

    def line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Bresenham, so bones look continuous rather than dotted."""
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx - dy
        while True:
            self.set(x0, y0)
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
