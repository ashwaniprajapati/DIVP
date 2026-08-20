# ============================================================
# DVP LAB
# IMAGE ENHANCEMENT - TRANSFORMATION FUNCTIONS
#
# 1. Negative
#       S = L - 1 - r
#
# 2. Power Law / Gamma Correction
#       S = c(r^gamma)
#
# 3. Log Transformation
#       S = c log(1 + r)
#
# i)  Without library
# ii) With tool/library
#
# Input:
# E:\Download\butterflyimage.jfif
#
# Output:
# E:\Download\DVP_Lab_Image_Enhancement
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# 2. INPUT IMAGE
# ============================================================

input_path = r"E:\Download\butterflyimage.jfif"


# ============================================================
# 3. MAIN OUTPUT FOLDER
# ============================================================

main_folder = r"E:\Download\DVP_Lab_Image_Enhancement"

without_library_folder = os.path.join(
    main_folder,
    "01_Without_Library"
)

with_library_folder = os.path.join(
    main_folder,
    "02_With_Library"
)

comparison_folder = os.path.join(
    main_folder,
    "03_Comparison"
)


# ============================================================
# 4. CREATE FOLDERS
# ============================================================

os.makedirs(without_library_folder, exist_ok=True)
os.makedirs(with_library_folder, exist_ok=True)
os.makedirs(comparison_folder, exist_ok=True)


print("Output folders created successfully!")
print()
print("Main folder:")
print(main_folder)


# ============================================================
# 5. READ IMAGE
# ============================================================

img_pil = Image.open(input_path).convert("L")

image = np.array(img_pil)

print()
print("Image loaded successfully!")
print("Image shape:", image.shape)
print("Image size:", image.size)


# ============================================================
# 6. FUNCTION TO SAVE IMAGE
# ============================================================

def save_image(array, filename, folder, quality=90):

    path = os.path.join(folder, filename)

    image_to_save = Image.fromarray(
        np.uint8(np.clip(array, 0, 255))
    )

    image_to_save.save(
        path,
        "JPEG",
        quality=quality,
        optimize=True
    )

    return path


# ============================================================
# 7. NEGATIVE TRANSFORMATION
#
# Formula:
#
# S = L - 1 - r
#
# For 8-bit image:
#
# L = 256
#
# Therefore:
#
# S = 255 - r
# ============================================================

def negative_without_library(img):

    L = 256

    result = np.zeros_like(img)

    rows, cols = img.shape

    for i in range(rows):

        for j in range(cols):

            r = int(img[i, j])

            result[i, j] = L - 1 - r

    return result


# ============================================================
# 8. GAMMA CORRECTION WITHOUT LIBRARY
#
# Formula:
#
# S = c(r^gamma)
#
# Normalize r to [0,1]
# ============================================================

def gamma_without_library(img, gamma, c=1):

    result = np.zeros_like(img, dtype=np.float64)

    rows, cols = img.shape

    for i in range(rows):

        for j in range(cols):

            r = img[i, j] / 255.0

            s = c * (r ** gamma)

            result[i, j] = s * 255

    return np.uint8(
        np.clip(result, 0, 255)
    )


# ============================================================
# 9. LOG TRANSFORMATION WITHOUT LIBRARY
#
# Formula:
#
# S = c log(1 + r)
#
# r is normalized to [0,255]
#
# c = 255 / log(256)
# ============================================================

def log_without_library(img):

    result = np.zeros_like(
        img,
        dtype=np.float64
    )

    c = 255 / np.log(256)

    rows, cols = img.shape

    for i in range(rows):

        for j in range(cols):

            r = img[i, j]

            s = c * np.log(1 + r)

            result[i, j] = s

    return np.uint8(
        np.clip(result, 0, 255)
    )


# ============================================================
# 10. RUN WITHOUT LIBRARY
# ============================================================

negative_no_lib = negative_without_library(image)

gamma05_no_lib = gamma_without_library(
    image,
    gamma=0.5
)

gamma15_no_lib = gamma_without_library(
    image,
    gamma=1.5
)

log_no_lib = log_without_library(image)


# ============================================================
# 11. SAVE WITHOUT-LIBRARY OUTPUTS
# ============================================================

save_image(
    image,
    "Original.jpg",
    without_library_folder
)

save_image(
    negative_no_lib,
    "Negative.jpg",
    without_library_folder
)

save_image(
    gamma05_no_lib,
    "Gamma_0.5.jpg",
    without_library_folder
)

save_image(
    gamma15_no_lib,
    "Gamma_1.5.jpg",
    without_library_folder
)

save_image(
    log_no_lib,
    "Log_Transformation.jpg",
    without_library_folder
)


# ============================================================
# 12. DISPLAY WITHOUT-LIBRARY RESULTS
# ============================================================

plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(negative_no_lib, cmap="gray")
plt.title("Negative")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(gamma05_no_lib, cmap="gray")
plt.title("Gamma = 0.5")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(gamma15_no_lib, cmap="gray")
plt.title("Gamma = 1.5")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(log_no_lib, cmap="gray")
plt.title("Log Transformation")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        without_library_folder,
        "All_Results.jpg"
    ),
    dpi=100,
    quality=85
)

plt.show()


# ============================================================
# 13. WITH LIBRARY
# ============================================================
#
# Using NumPy/Pillow functions instead of manually
# processing every pixel.
# ============================================================


# ------------------------------------------------------------
# Negative using NumPy
# ------------------------------------------------------------

negative_library = 255 - image


# ------------------------------------------------------------
# Gamma using NumPy
# ------------------------------------------------------------

gamma05_library = np.uint8(
    255 * (
        image / 255.0
    ) ** 0.5
)


gamma15_library = np.uint8(
    255 * (
        image / 255.0
    ) ** 1.5
)


# ------------------------------------------------------------
# Log transformation using NumPy
# ------------------------------------------------------------

c = 255 / np.log(256)

log_library = np.uint8(
    c * np.log(
        1 + image
    )
)


# ============================================================
# 14. SAVE WITH-LIBRARY OUTPUTS
# ============================================================

save_image(
    image,
    "Original.jpg",
    with_library_folder
)

save_image(
    negative_library,
    "Negative.jpg",
    with_library_folder
)

save_image(
    gamma05_library,
    "Gamma_0.5.jpg",
    with_library_folder
)

save_image(
    gamma15_library,
    "Gamma_1.5.jpg",
    with_library_folder
)

save_image(
    log_library,
    "Log_Transformation.jpg",
    with_library_folder
)


# ============================================================
# 15. DISPLAY WITH-LIBRARY RESULTS
# ============================================================

plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(negative_library, cmap="gray")
plt.title("Negative")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(gamma05_library, cmap="gray")
plt.title("Gamma = 0.5")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(gamma15_library, cmap="gray")
plt.title("Gamma = 1.5")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(log_library, cmap="gray")
plt.title("Log Transformation")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        with_library_folder,
        "All_Results.jpg"
    ),
    dpi=100
)

plt.show()


# ============================================================
# 16. COMPARISON - NEGATIVE
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(
    negative_no_lib,
    cmap="gray"
)
plt.title("Negative - Without Library")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(
    negative_library,
    cmap="gray"
)
plt.title("Negative - With Library")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        comparison_folder,
        "Negative_Comparison.jpg"
    ),
    dpi=100
)

plt.show()


# ============================================================
# 17. COMPARISON - GAMMA
# ============================================================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(
    image,
    cmap="gray"
)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(
    gamma05_no_lib,
    cmap="gray"
)
plt.title("Gamma = 0.5")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(
    gamma15_no_lib,
    cmap="gray"
)
plt.title("Gamma = 1.5")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        comparison_folder,
        "Gamma_Comparison.jpg"
    ),
    dpi=100
)

plt.show()


# ============================================================
# 18. COMPARISON - LOG
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(
    log_no_lib,
    cmap="gray"
)
plt.title("Log - Without Library")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(
    log_library,
    cmap="gray"
)
plt.title("Log - With Library")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        comparison_folder,
        "Log_Comparison.jpg"
    ),
    dpi=100
)

plt.show()


# ============================================================
# 19. FUNCTION TO CALCULATE FOLDER SIZE
# ============================================================

def get_folder_size(folder):

    total = 0

    for root, dirs, files in os.walk(folder):

        for file in files:

            path = os.path.join(
                root,
                file
            )

            if os.path.exists(path):

                total += os.path.getsize(path)

    return total


def bytes_to_mb(size):

    return size / (1024 * 1024)


# ============================================================
# 20. CHECK EACH FOLDER SIZE
# ============================================================

print()
print("=" * 60)
print("OUTPUT FOLDER SIZE")
print("=" * 60)


folders = [
    without_library_folder,
    with_library_folder,
    comparison_folder
]


for folder in folders:

    size = get_folder_size(folder)

    print(
        os.path.basename(folder),
        " : ",
        f"{bytes_to_mb(size):.2f} MB"
    )


# ============================================================
# 21. CHECK WHETHER FOLDERS ARE BELOW 25 MB
# ============================================================

print()
print("=" * 60)
print("SIZE CHECK")
print("=" * 60)


LIMIT = 25 * 1024 * 1024


for folder in folders:

    size = get_folder_size(folder)

    folder_name = os.path.basename(folder)

    if size < LIMIT:

        print(
            f"✓ {folder_name} "
            f"is BELOW 25 MB"
        )

    else:

        print(
            f"⚠ {folder_name} "
            f"is ABOVE 25 MB"
        )


# ============================================================
# 22. CREATE README FILE
# ============================================================

readme_text = """

DVP LAB
IMAGE ENHANCEMENT - TRANSFORMATION FUNCTIONS
==============================================

Input Image:
butterflyimage.jfif


TRANSFORMATIONS
---------------

1. Negative Transformation

Formula:

S = L - 1 - r

For an 8-bit image:

L = 256

Therefore:

S = 255 - r


2. Power Law / Gamma Correction

Formula:

S = c(r^gamma)

Implemented gamma values:

gamma = 0.5
gamma = 1.5


3. Log Transformation

Formula:

S = c log(1 + r)

For an 8-bit image:

c = 255 / log(256)


IMPLEMENTATIONS
---------------

1. Without Library
   Pixel-by-pixel implementation.

2. With Library
   NumPy based implementation.


OUTPUT FOLDERS
--------------

01_Without_Library
Contains manually implemented transformations.

02_With_Library
Contains NumPy/library based transformations.

03_Comparison
Contains comparison images.


INPUT
-----

E:\\Download\\butterflyimage.jfif


All output folders are checked to ensure
their size remains below 25 MB.

"""


readme_path = os.path.join(
    main_folder,
    "README.txt"
)


with open(
    readme_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(readme_text)


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print()
print("=" * 60)
print("LAB COMPLETED SUCCESSFULLY")
print("=" * 60)

print()
print("Input:")
print(input_path)

print()
print("Output:")
print(main_folder)

print()
print("README:")
print(readme_path)

print()
print("Folders created:")

for folder in folders:

    size = get_folder_size(folder)

    print(
        f"  {os.path.basename(folder)}"
        f" -> {bytes_to_mb(size):.2f} MB"
    )

print()
print("✓ Negative transformation completed")
print("✓ Gamma correction completed")
print("✓ Log transformation completed")
print("✓ Without-library implementation completed")
print("✓ With-library implementation completed")
print("✓ Comparison images created")
print("✓ README created")
print("✓ Folder sizes checked")
