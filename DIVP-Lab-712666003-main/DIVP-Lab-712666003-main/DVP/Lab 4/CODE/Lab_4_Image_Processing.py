import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# AGV SPATIAL FILTERING LAB
# Input image:
# E:\Download\Nebula (1).png
#
# IMPORTANT:
# All filtering operations below are implemented from scratch.
# No cv2.filter2D or scipy.signal filtering is used.
# ============================================================

INPUT_PATH = r"E:\Download\Nebula (1).png"

# Output folder will be created on the same drive as the input image.
# If E: exists, outputs go to E:\Download\AGV_Spatial_Filtering_Outputs.
# If the script is run on another computer where E: is unavailable,
# it falls back to a folder beside this Python script.
OUTPUT_DIR = r"E:\Download\AGV_Spatial_Filtering_Outputs"

if not os.path.exists(os.path.dirname(INPUT_PATH)):
    OUTPUT_DIR = os.path.join(os.getcwd(), "AGV_Spatial_Filtering_Outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Subfolders
DEGRADED_DIR = os.path.join(OUTPUT_DIR, "01_Degraded_Images")
TASK1_DIR = os.path.join(OUTPUT_DIR, "02_Task1_Averaging")
TASK2_DIR = os.path.join(OUTPUT_DIR, "03_Task2_Laplacian")
TASK3_DIR = os.path.join(OUTPUT_DIR, "04_Task3_HighBoost")
TASK4_DIR = os.path.join(OUTPUT_DIR, "05_Task4_Metrics")
PIPELINE_DIR = os.path.join(OUTPUT_DIR, "06_Final_Pipeline")

for folder in [
    DEGRADED_DIR, TASK1_DIR, TASK2_DIR,
    TASK3_DIR, TASK4_DIR, PIPELINE_DIR
]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# BASIC IMAGE FUNCTIONS
# ============================================================

def load_grayscale(path):
    """Load image and convert to grayscale."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input image not found:\n{path}\n"
            "Check that the image exists at the specified location."
        )

    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)


def save_image(array, path):
    """Clip image to 0-255 and save as PNG."""
    array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def normalize_for_display(array):
    """Normalize an image/response map to 0-255 for visualization."""
    arr = np.asarray(array, dtype=np.float32)
    mn = np.min(arr)
    mx = np.max(arr)

    if mx - mn < 1e-12:
        return np.zeros_like(arr)

    return (arr - mn) * 255.0 / (mx - mn)


# ============================================================
# DEGRADATION: GAUSSIAN NOISE
# ============================================================

def add_gaussian_noise(image, sigma, seed=42):
    """
    Add zero-mean Gaussian noise with the requested sigma.
    Sigma is measured on the 0-255 intensity scale.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0, 255)


# ============================================================
# FROM-SCRATCH 2D CORRELATION
# ============================================================

def correlate2d(image, kernel):
    """
    2D correlation implemented from scratch.

    Uses edge padding so the output has the same dimensions
    as the input image.
    """
    image = np.asarray(image, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)

    kh, kw = kernel.shape
    ph = kh // 2
    pw = kw // 2

    padded = np.pad(
        image,
        ((ph, ph), (pw, pw)),
        mode="edge"
    )

    output = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)

    return output


# ============================================================
# FROM-SCRATCH 2D CONVOLUTION
# ============================================================

def convolve2d(image, kernel):
    """True convolution: flip kernel in both dimensions."""
    flipped_kernel = np.flipud(np.fliplr(kernel))
    return correlate2d(image, flipped_kernel)


# ============================================================
# AVERAGING FILTER
# ============================================================

def averaging_filter(image, size):
    """Box/averaging filter implemented using correlate2d."""
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    return correlate2d(image, kernel)


# ============================================================
# MOTION BLUR
# ============================================================

def motion_blur(image, length=9):
    """
    Horizontal linear motion blur.
    Length is 7-9 pixels as required by the problem statement.
    """
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2

    # Horizontal line through the center
    kernel[center, :] = 1.0 / length

    return correlate2d(image, kernel)


# ============================================================
# LAPLACIAN SHARPENING
# ============================================================

LAPLACIAN_4 = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=np.float32)

LAPLACIAN_8 = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)


def laplacian_response(image, kernel):
    """Calculate Laplacian response using correlation."""
    return correlate2d(image, kernel)


def laplacian_sharpen(image, kernel, alpha=1.0):
    """
    Sharpen image using:
        g = f + alpha * Laplacian(f)
    """
    response = laplacian_response(image, kernel)
    return np.clip(image + alpha * response, 0, 255)


# ============================================================
# UNSHARP MASKING / HIGH-BOOST
# ============================================================

def high_boost(image, k, blur_size=3):
    """
    High-boost filtering.

    Blurred image: B
    Mask: M = f - B
    Output: g = f + k*M

    k=1 gives unsharp masking.
    k>1 gives high-boost filtering.
    """
    blurred = averaging_filter(image, blur_size)
    mask = image - blurred
    return np.clip(image + k * mask, 0, 255)


# ============================================================
# METRICS
# ============================================================

def mse(reference, test):
    """Mean Squared Error."""
    reference = reference.astype(np.float32)
    test = test.astype(np.float32)
    return np.mean((reference - test) ** 2)


def psnr(reference, test):
    """
    Peak Signal-to-Noise Ratio.
    MAX_I = 255.
    """
    error = mse(reference, test)

    if error == 0:
        return float("inf")

    return 10 * math.log10((255.0 ** 2) / error)


def gradient_magnitude(image):
    """
    Mean gradient magnitude using simple central differences.

    The metric is:
        mean(sqrt(Gx^2 + Gy^2))
    """
    image = image.astype(np.float32)

    gx = np.zeros_like(image)
    gy = np.zeros_like(image)

    # Central difference in x direction
    gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2.0

    # Central difference in y direction
    gy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2.0

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    return float(np.mean(magnitude))


# ============================================================
# PLOTTING HELPERS
# ============================================================

def save_grid(images, titles, path, cols=3, figsize=(14, 9)):
    """
    Save a labeled image grid.
    """
    rows = math.ceil(len(images) / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        squeeze=False
    )

    axes = axes.ravel()

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(
            np.clip(image, 0, 255),
            cmap="gray",
            vmin=0,
            vmax=255
        )
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_metric_table(records, path):
    """
    Save metric results as CSV.
    """
    import csv

    if not records:
        return

    fields = list(records[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("AGV SPATIAL FILTERING LAB")
    print("=" * 70)

    # --------------------------------------------------------
    # Load clean image
    # --------------------------------------------------------

    clean = load_grayscale(INPUT_PATH)

    print(f"\nInput image: {INPUT_PATH}")
    print(f"Image size : {clean.shape[1]} x {clean.shape[0]} pixels")
    print(f"Output dir : {OUTPUT_DIR}")

    save_image(
        clean,
        os.path.join(DEGRADED_DIR, "00_Clean_Grayscale.png")
    )

    # --------------------------------------------------------
    # Generate Gaussian noise
    # sigma = 10 and sigma = 25
    # --------------------------------------------------------

    noisy10 = add_gaussian_noise(clean, sigma=10, seed=42)
    noisy25 = add_gaussian_noise(clean, sigma=25, seed=43)

    save_image(
        noisy10,
        os.path.join(DEGRADED_DIR, "01_Gaussian_Noise_sigma10.png")
    )

    save_image(
        noisy25,
        os.path.join(DEGRADED_DIR, "02_Gaussian_Noise_sigma25.png")
    )

    # --------------------------------------------------------
    # Motion blur length = 9
    # --------------------------------------------------------

    blurred = motion_blur(clean, length=9)

    save_image(
        blurred,
        os.path.join(DEGRADED_DIR, "03_Motion_Blur_Length9.png")
    )

    # Also create a blurred + noisy image for reference
    blurred_noisy10 = add_gaussian_noise(blurred, sigma=10, seed=44)

    save_image(
        blurred_noisy10,
        os.path.join(DEGRADED_DIR, "04_Motion_Blur_plus_Noise_sigma10.png")
    )

    # --------------------------------------------------------
    # TASK 1
    # Averaging filters: 3x3, 5x5, 9x9
    # --------------------------------------------------------

    print("\nTASK 1: Averaging Filter")

    task1_images = [noisy10, noisy25]
    task1_titles = [
        "Original noisy image - sigma=10",
        "Original noisy image - sigma=25"
    ]

    records = []

    for sigma, noisy in [(10, noisy10), (25, noisy25)]:

        for size in [3, 5, 9]:

            result = averaging_filter(noisy, size)

            filename = (
                f"Sigma_{sigma}_Average_{size}x{size}.png"
            )

            save_image(
                result,
                os.path.join(TASK1_DIR, filename)
            )

            task1_images.append(result)
            task1_titles.append(
                f"sigma={sigma}, average {size}x{size}"
            )

            records.append({
                "Task": "Task 1",
                "Configuration": f"sigma={sigma}, average={size}x{size}",
                "PSNR_dB": round(psnr(clean, result), 4),
                "Mean_Gradient_Magnitude": round(
                    gradient_magnitude(result), 4
                )
            })

    save_grid(
        task1_images,
        task1_titles,
        os.path.join(TASK1_DIR, "Task1_Averaging_Grid.png"),
        cols=4,
        figsize=(16, 9)
    )

    # --------------------------------------------------------
    # TASK 2
    # Laplacian sharpening: 4-neighbor and 8-neighbor
    # --------------------------------------------------------

    print("TASK 2: Laplacian Sharpening")

    response4 = laplacian_response(blurred, LAPLACIAN_4)
    response8 = laplacian_response(blurred, LAPLACIAN_8)

    sharpen4 = laplacian_sharpen(blurred, LAPLACIAN_4)
    sharpen8 = laplacian_sharpen(blurred, LAPLACIAN_8)

    save_image(
        normalize_for_display(response4),
        os.path.join(TASK2_DIR, "Laplacian_Response_4_Neighbor.png")
    )

    save_image(
        normalize_for_display(response8),
        os.path.join(TASK2_DIR, "Laplacian_Response_8_Neighbor.png")
    )

    save_image(
        sharpen4,
        os.path.join(TASK2_DIR, "Laplacian_Sharpened_4_Neighbor.png")
    )

    save_image(
        sharpen8,
        os.path.join(TASK2_DIR, "Laplacian_Sharpened_8_Neighbor.png")
    )

    save_grid(
        [blurred, response4, response8, sharpen4, sharpen8],
        [
            "Motion Blurred Image",
            "4-Neighbor Laplacian Response",
            "8-Neighbor Laplacian Response",
            "4-Neighbor Sharpened",
            "8-Neighbor Sharpened"
        ],
        os.path.join(TASK2_DIR, "Task2_Laplacian_Grid.png"),
        cols=3,
        figsize=(14, 9)
    )

    records.extend([
        {
            "Task": "Task 2",
            "Configuration": "4-neighbor Laplacian",
            "PSNR_dB": round(psnr(clean, sharpen4), 4),
            "Mean_Gradient_Magnitude": round(
                gradient_magnitude(sharpen4), 4
            )
        },
        {
            "Task": "Task 2",
            "Configuration": "8-neighbor Laplacian",
            "PSNR_dB": round(psnr(clean, sharpen8), 4),
            "Mean_Gradient_Magnitude": round(
                gradient_magnitude(sharpen8), 4
            )
        }
    ])

    # --------------------------------------------------------
    # TASK 3
    # Unsharp / high-boost: k = 1, 1.5, 2, 3
    # Applied to sigma=10 noisy image
    # --------------------------------------------------------

    print("TASK 3: Unsharp Masking / High-Boost")

    k_values = [1.0, 1.5, 2.0, 3.0]

    task3_images = [noisy10]
    task3_titles = ["Input: sigma=10"]

    sharpness_values = []

    for k in k_values:

        result = high_boost(
            noisy10,
            k=k,
            blur_size=3
        )

        filename = f"HighBoost_k_{str(k).replace('.', '_')}.png"

        save_image(
            result,
            os.path.join(TASK3_DIR, filename)
        )

        task3_images.append(result)
        task3_titles.append(f"High-Boost k={k}")

        sharpness = gradient_magnitude(result)
        sharpness_values.append(sharpness)

        records.append({
            "Task": "Task 3",
            "Configuration": f"High-Boost k={k}",
            "PSNR_dB": round(psnr(clean, result), 4),
            "Mean_Gradient_Magnitude": round(sharpness, 4)
        })

    save_grid(
        task3_images,
        task3_titles,
        os.path.join(TASK3_DIR, "Task3_HighBoost_Grid.png"),
        cols=3,
        figsize=(14, 8)
    )

    # --------------------------------------------------------
    # Sharpness vs k plot
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.plot(
        k_values,
        sharpness_values,
        marker="o"
    )
    plt.xlabel("Boost Factor k")
    plt.ylabel("Mean Gradient Magnitude")
    plt.title("Output Sharpness vs Boost Factor")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            TASK3_DIR,
            "Sharpness_vs_Boost_Factor.png"
        ),
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # TASK 4
    # Save all metric results
    # --------------------------------------------------------

    print("TASK 4: Objective Evaluation")

    csv_path = os.path.join(
        TASK4_DIR,
        "PSNR_and_Sharpness_Metrics.csv"
    )

    save_metric_table(records, csv_path)

    # --------------------------------------------------------
    # Create a readable TXT report of metrics
    # --------------------------------------------------------

    txt_path = os.path.join(
        TASK4_DIR,
        "PSNR_and_Sharpness_Metrics.txt"
    )

    with open(txt_path, "w", encoding="utf-8") as f:

        f.write("AGV SPATIAL FILTERING - TASK 4 METRICS\n")
        f.write("=" * 75 + "\n\n")

        f.write(
            "Sharpness metric: Mean Gradient Magnitude\n"
        )
        f.write(
            "Definition: mean(sqrt(Gx^2 + Gy^2)), using central differences.\n\n"
        )

        for record in records:

            f.write(
                f"{record['Task']:<10} | "
                f"{record['Configuration']:<40} | "
                f"PSNR = {record['PSNR_dB']} dB | "
                f"Sharpness = {record['Mean_Gradient_Magnitude']}\n"
            )

    # --------------------------------------------------------
    # TASK 5
    # Automatic recommendation based on metrics
    #
    # We select the configuration with the highest PSNR among
    # Task 1 denoising outputs, then sharpen that result using
    # the Laplacian variant that gives the better PSNR.
    #
    # This is a data-based starting point; inspect the actual
    # metrics and navigation relevance before deployment.
    # --------------------------------------------------------

    task1_records = [
        r for r in records
        if r["Task"] == "Task 1"
    ]

    best_denoise_record = max(
        task1_records,
        key=lambda r: r["PSNR_dB"]
    )

    # Parse selected sigma and kernel size
    selected_config = best_denoise_record["Configuration"]

    # Example:
    # sigma=10, average=3x3
    sigma_part = selected_config.split(",")[0]
    kernel_part = selected_config.split("=")[-1]

    selected_sigma = int(
        sigma_part.split("=")[1]
    )

    selected_kernel = int(
        kernel_part.split("x")[0]
    )

    selected_noisy = (
        noisy10 if selected_sigma == 10 else noisy25
    )

    selected_denoised = averaging_filter(
        selected_noisy,
        selected_kernel
    )

    # Select Laplacian variant by PSNR
    task2_records = [
        r for r in records
        if r["Task"] == "Task 2"
    ]

    best_laplacian = max(
        task2_records,
        key=lambda r: r["PSNR_dB"]
    )

    if "4-neighbor" in best_laplacian["Configuration"]:
        selected_laplacian = LAPLACIAN_4
        laplacian_name = "4-neighbor"
    else:
        selected_laplacian = LAPLACIAN_8
        laplacian_name = "8-neighbor"

    # Apply selected Laplacian to denoised image
    final_pipeline = laplacian_sharpen(
        selected_denoised,
        selected_laplacian
    )

    save_image(
        selected_denoised,
        os.path.join(
            PIPELINE_DIR,
            "Stage1_Selected_Denoised_Image.png"
        )
    )

    save_image(
        final_pipeline,
        os.path.join(
            PIPELINE_DIR,
            "Stage2_Final_Denoise_Sharpen_Image.png"
        )
    )

    save_grid(
        [
            clean,
            selected_noisy,
            selected_denoised,
            final_pipeline
        ],
        [
            "Clean Ground Truth",
            f"Noisy sigma={selected_sigma}",
            f"Denoised {selected_kernel}x{selected_kernel}",
            f"Final: {laplacian_name} Laplacian"
        ],
        os.path.join(
            PIPELINE_DIR,
            "Final_Pipeline_Grid.png"
        ),
        cols=4,
        figsize=(16, 5)
    )

    # --------------------------------------------------------
    # Pipeline report
    # --------------------------------------------------------

    final_report = os.path.join(
        PIPELINE_DIR,
        "Pipeline_Recommendation.txt"
    )

    with open(final_report, "w", encoding="utf-8") as f:

        f.write("AGV SPATIAL-FILTERING PIPELINE RECOMMENDATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            "The recommendation below is generated from the Task 4 "
            "PSNR measurements.\n\n"
        )

        f.write(
            f"Selected denoising configuration: "
            f"sigma={selected_sigma}, "
            f"{selected_kernel}x{selected_kernel} averaging filter\n"
        )

        f.write(
            f"Denoising PSNR: "
            f"{best_denoise_record['PSNR_dB']} dB\n"
        )

        f.write(
            f"Selected Laplacian: {laplacian_name}\n"
        )

        f.write(
            f"Selected Laplacian PSNR: "
            f"{best_laplacian['PSNR_dB']} dB\n\n"
        )

        f.write(
            "Two-stage pipeline:\n"
        )

        f.write(
            f"1. Denoise using {selected_kernel}x{selected_kernel} "
            f"averaging filter.\n"
        )

        f.write(
            f"2. Sharpen using the {laplacian_name} Laplacian.\n"
        )

        f.write(
            "\nImportant: The lab statement explicitly notes that there "
            "is no single correct kernel size or boost factor. "
            "The final deployment choice should therefore be defended "
            "using the measured metrics and the image's navigation "
            "requirements.\n"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETED")
    print("=" * 70)

    print(f"\nAll output files are saved in:\n{OUTPUT_DIR}")

    print("\nBest Task 1 configuration by PSNR:")
    print(
        f"  {best_denoise_record['Configuration']} "
        f"-> {best_denoise_record['PSNR_dB']} dB"
    )

    print("\nBest Task 2 Laplacian configuration by PSNR:")
    print(
        f"  {best_laplacian['Configuration']} "
        f"-> {best_laplacian['PSNR_dB']} dB"
    )

    print("\nOutput folders:")
    print(f"  01_Degraded_Images")
    print(f"  02_Task1_Averaging")
    print(f"  03_Task2_Laplacian")
    print(f"  04_Task3_HighBoost")
    print(f"  05_Task4_Metrics")
    print(f"  06_Final_Pipeline")

    print("\nDone.")


if __name__ == "__main__":
    main()
