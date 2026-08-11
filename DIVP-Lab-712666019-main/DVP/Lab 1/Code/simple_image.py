"""Small dependency-free text-image helper for DVP labs.

The helper stores datasets as ASCII PPM (P3) files and exports results as SVG.
Both formats are plain text, so they are friendly to GitHub PR diffs that reject
binary files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

Pixel = tuple[int, int, int]


def clamp(value: float) -> int:
    return max(0, min(255, int(round(value))))


@dataclass
class Image:
    width: int
    height: int
    pixels: list[list[Pixel]]

    @classmethod
    def gradient(cls, width: int = 96, height: int = 72) -> "Image":
        pixels: list[list[Pixel]] = []
        for y in range(height):
            row: list[Pixel] = []
            for x in range(width):
                r = int(255 * x / max(1, width - 1))
                g = int(255 * y / max(1, height - 1))
                b = int(255 * (x + y) / max(1, width + height - 2))
                if 22 < x < 55 and 16 < y < 50:
                    r, g, b = 245, 90, 60
                if (x - 73) ** 2 + (y - 36) ** 2 < 14 ** 2:
                    r, g, b = 60, 160, 245
                row.append((r, g, b))
            pixels.append(row)
        return cls(width, height, pixels)

    @classmethod
    def read_ppm(cls, path: str | Path) -> "Image":
        tokens: list[str] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                tokens.extend(line.split())
        if len(tokens) < 4 or tokens[0] != "P3":
            raise ValueError("Only ASCII PPM (P3) files are supported")
        width, height, max_value = int(tokens[1]), int(tokens[2]), int(tokens[3])
        if max_value != 255:
            raise ValueError("Only max value 255 PPM files are supported")
        values = [int(token) for token in tokens[4:]]
        expected = width * height * 3
        if len(values) != expected:
            raise ValueError(f"Expected {expected} channel values, got {len(values)}")
        pixels: list[list[Pixel]] = []
        index = 0
        for _ in range(height):
            row: list[Pixel] = []
            for _ in range(width):
                row.append((values[index], values[index + 1], values[index + 2]))
                index += 3
            pixels.append(row)
        return cls(width, height, pixels)

    def save_ppm(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["P3", f"# Generated DVP lab image: {self.width}x{self.height}", f"{self.width} {self.height}", "255"]
        for row in self.pixels:
            lines.append(" ".join(str(channel) for pixel in row for channel in pixel))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def save_svg(self, path: str | Path, title: str = "DVP output") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rects = []
        for y, row in enumerate(self.pixels):
            run_start = 0
            run_color = row[0]
            for x, pixel in enumerate(row[1:], start=1):
                if pixel != run_color:
                    rects.append((run_start, y, x - run_start, run_color))
                    run_start, run_color = x, pixel
            rects.append((run_start, y, self.width - run_start, run_color))
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" shape-rendering="crispEdges">',
            f"  <title>{escape(title)}</title>",
        ]
        for x, y, width, (r, g, b) in rects:
            lines.append(f'  <rect x="{x}" y="{y}" width="{width}" height="1" fill="rgb({r},{g},{b})"/>')
        lines.append("</svg>")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def grayscale(self) -> "Image":
        return self.map_pixels(lambda r, g, b: (gray := clamp(0.299 * r + 0.587 * g + 0.114 * b), gray, gray))

    def brightness(self, amount: int = 40) -> "Image":
        return self.map_pixels(lambda r, g, b: (clamp(r + amount), clamp(g + amount), clamp(b + amount)))

    def contrast(self, factor: float = 1.4) -> "Image":
        return self.map_pixels(lambda r, g, b: (clamp((r - 128) * factor + 128), clamp((g - 128) * factor + 128), clamp((b - 128) * factor + 128)))

    def negative(self) -> "Image":
        return self.map_pixels(lambda r, g, b: (255 - r, 255 - g, 255 - b))

    def resize_half(self) -> "Image":
        new_w, new_h = max(1, self.width // 2), max(1, self.height // 2)
        return Image(new_w, new_h, [[self.pixels[min(self.height - 1, y * 2)][min(self.width - 1, x * 2)] for x in range(new_w)] for y in range(new_h)])

    def rotate_90_clockwise(self) -> "Image":
        return Image(self.height, self.width, [[self.pixels[self.height - 1 - x][y] for x in range(self.height)] for y in range(self.width)])

    def flip_horizontal(self) -> "Image":
        return Image(self.width, self.height, [list(reversed(row)) for row in self.pixels])

    def crop_center(self) -> "Image":
        x1, x2 = self.width // 4, self.width * 3 // 4
        y1, y2 = self.height // 4, self.height * 3 // 4
        return Image(x2 - x1, y2 - y1, [row[x1:x2] for row in self.pixels[y1:y2]])

    def map_pixels(self, func) -> "Image":
        return Image(self.width, self.height, [[func(*pixel) for pixel in row] for row in self.pixels])
