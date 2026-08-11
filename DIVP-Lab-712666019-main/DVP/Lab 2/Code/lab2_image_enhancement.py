"""Lab 2: image enhancement transformation functions."""
from pathlib import Path
from math import log
from simple_image import Image, clamp

BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / "Dataset" / "Lab 2" / "sample.ppm"
OUTPUT = BASE / "Lab 2" / "Image Output"
L = 256


def ensure_input() -> Image:
    if not INPUT.exists():
        Image.gradient().save_ppm(INPUT)
    return Image.read_ppm(INPUT)


def manual_negative(image: Image) -> Image:
    return Image(image.width, image.height, [[tuple(L - 1 - value for value in pixel) for pixel in row] for row in image.pixels])


def manual_gamma(image: Image, gamma: float = 0.55, c: float = 1.0) -> Image:
    return Image(image.width, image.height, [[tuple(clamp(c * 255 * ((value / 255) ** gamma)) for value in pixel) for pixel in row] for row in image.pixels])


def manual_log(image: Image) -> Image:
    c = 255 / log(1 + 255)
    return Image(image.width, image.height, [[tuple(clamp(c * log(1 + value)) for value in pixel) for pixel in row] for row in image.pixels])


def tool_negative(image: Image) -> Image:
    return image.negative()


def tool_gamma(image: Image, gamma: float = 0.55) -> Image:
    return image.map_pixels(lambda r, g, b: tuple(clamp(255 * ((value / 255) ** gamma)) for value in (r, g, b)))


def tool_log(image: Image) -> Image:
    c = 255 / log(256)
    return image.map_pixels(lambda r, g, b: tuple(clamp(c * log(1 + value)) for value in (r, g, b)))


def save(name: str, image: Image) -> None:
    path = OUTPUT / name
    image.save_svg(path, name)
    print(f"Saved: {path}")


def main() -> None:
    image = ensure_input()
    save("01_original.svg", image)
    save("02_manual_negative.svg", manual_negative(image))
    save("03_manual_gamma.svg", manual_gamma(image))
    save("04_manual_log.svg", manual_log(image))
    save("05_tool_negative.svg", tool_negative(image))
    save("06_tool_gamma.svg", tool_gamma(image))
    save("07_tool_log.svg", tool_log(image))


if __name__ == "__main__":
    main()
