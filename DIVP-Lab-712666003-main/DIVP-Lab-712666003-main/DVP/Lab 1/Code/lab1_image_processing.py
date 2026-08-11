"""Lab 1: menu-driven basic image processing."""
from pathlib import Path
from simple_image import Image

BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / "Dataset" / "Lab 1" / "sample.ppm"
OUTPUT = BASE / "Lab 1" / "Image Output"


def load_image() -> Image:
    if not INPUT.exists():
        Image.gradient().save_ppm(INPUT)
    return Image.read_ppm(INPUT)


def display_properties(image: Image) -> None:
    print(f"Image path: {INPUT}")
    print(f"Width: {image.width}px")
    print(f"Height: {image.height}px")
    print("Channels: 3 (RGB)")
    print("Format: ASCII PPM (P3) dataset, SVG outputs")


def save_result(name: str, image: Image) -> None:
    path = OUTPUT / name
    image.save_svg(path, name)
    print(f"Saved: {path}")


def run_demo() -> None:
    image = load_image()
    save_result("01_original.svg", image)
    save_result("02_grayscale.svg", image.grayscale())
    save_result("03_brightness.svg", image.brightness(50))
    save_result("04_contrast.svg", image.contrast(1.6))
    save_result("05_resized_half.svg", image.resize_half())
    save_result("06_rotated_90_clockwise.svg", image.rotate_90_clockwise())
    save_result("07_flipped_horizontal.svg", image.flip_horizontal())
    save_result("08_cropped_center.svg", image.crop_center())
    save_result("09_negative.svg", image.negative())
    display_properties(image)


def menu() -> None:
    image = load_image()
    while True:
        print("\n========= IMAGE PROCESSING LAB =========")
        print("1. Display Image")
        print("2. Display Image Properties")
        print("3. Convert to Grayscale")
        print("4. Increase Brightness")
        print("5. Increase Contrast")
        print("6. Resize Image")
        print("7. Rotate Image")
        print("8. Flip Image")
        print("9. Crop Image")
        print("10. Negative Image")
        print("11. Save Image")
        print("12. Exit")
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            save_result("display_original.svg", image)
        elif choice == "2":
            display_properties(image)
        elif choice == "3":
            save_result("grayscale.svg", image.grayscale())
        elif choice == "4":
            save_result("brightness.svg", image.brightness(50))
        elif choice == "5":
            save_result("contrast.svg", image.contrast(1.6))
        elif choice == "6":
            save_result("resized_half.svg", image.resize_half())
        elif choice == "7":
            save_result("rotated_90_clockwise.svg", image.rotate_90_clockwise())
        elif choice == "8":
            save_result("flipped_horizontal.svg", image.flip_horizontal())
        elif choice == "9":
            save_result("cropped_center.svg", image.crop_center())
        elif choice == "10":
            save_result("negative.svg", image.negative())
        elif choice == "11":
            save_result("saved_copy.svg", image)
        elif choice == "12":
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    run_demo()
