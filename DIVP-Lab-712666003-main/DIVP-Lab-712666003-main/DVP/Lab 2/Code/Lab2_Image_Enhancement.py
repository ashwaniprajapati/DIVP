import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# ============================================================
# IMAGE ENHANCEMENT - TRANSFORMATION FUNCTIONS
# ============================================================
#
# 1. Negative:
#       S = L - 1 - r
#
# 2. Power Law / Gamma:
#       S = c(r^gamma)
#
# 3. Log:
#       S = c log(1 + r)
#
# Two implementations:
#       i)  Without library
#       ii) With tool & library
# ============================================================


# ============================================================
# 1. INPUT IMAGE PATH
# ============================================================

image_path = r"E:\Download\REFERENCE IMAGE.jpg"


# ============================================================
# 2. CREATE MAIN OUTPUT FOLDER
# ============================================================

main_output_folder = r"E:\Download\Image_Enhancement_Lab"

os.makedirs(main_output_folder, exist_ok=True)


# ============================================================
# 3. CREATE SEPARATE OUTPUT FOLDERS
# ============================================================

without_library_folder = os.path.join(
    main_output_folder,
    "01_Without_Library"
)

with_library_folder = os.path.join(
    main_output_folder,
    "02_With_Library"
)

comparison_folder = os.path.join(
    main_output_folder,
    "03_Comparison"
)

os.makedirs(without_library_folder, exist_ok=True)
os.makedirs(with_library_folder, exist_ok=True)
os.makedirs(comparison_folder, exist_ok=True)


print("=" * 60)
print("IMAGE ENHANCEMENT LAB")
print("=" * 60)

print("\nOutput folder created at:")
print(main_output_folder)


# ============================================================
# 4. READ IMAGE
# ============================================================

image_pil = Image.open(image_path).convert("L")

image = np.array(image_pil, dtype=np.float32)

print("\nImage loaded successfully.")

print("Image size:")
print(image.shape)

print("Minimum intensity:", image.min())
print("Maximum intensity:", image.max())


# ============================================================
# PART I
# WITHOUT LIBRARY
# ============================================================

print("\n" + "=" * 60)
print("PART I - WITHOUT LIBRARY")
print("=" * 60)


# ------------------------------------------------------------
# 1. NEGATIVE TRANSFORMATION
# ------------------------------------------------------------
#
# Formula:
#
#       S = L - 1 - r
#
# For an 8-bit image:
#
#       L = 256
#
# Therefore:
#
#       S = 255 - r
# ------------------------------------------------------------

def negative_without_library(img):

    L = 256

    result = L - 1 - img

    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)


negative_manual = negative_without_library(image)


# ------------------------------------------------------------
# 2. POWER LAW / GAMMA CORRECTION
# ------------------------------------------------------------
#
# Formula:
#
#       S = c(r^gamma)
#
# Pixel values are first normalized:
#
#       r_normalized = r / 255
#
# Then:
#
#       S = c(r_normalized^gamma)
#
# Finally converted back to 0-255.
# ------------------------------------------------------------

def gamma_without_library(img, gamma, c=1):

    # Normalize pixel values
    r = img / 255.0

    # Apply power law
    s = c * (r ** gamma)

    # Convert back to 0-255
    s = s * 255

    # Keep values within valid range
    s = np.clip(s, 0, 255)

    return s.astype(np.uint8)


# Gamma < 1 makes image brighter
gamma_05_manual = gamma_without_library(
    image,
    gamma=0.5
)

# Gamma > 1 makes image darker
gamma_20_manual = gamma_without_library(
    image,
    gamma=2.0
)


# ------------------------------------------------------------
# 3. LOG TRANSFORMATION
# ------------------------------------------------------------
#
# Formula:
#
#       S = c log(1 + r)
#
# We normalize r first.
# ------------------------------------------------------------

def log_without_library(img, c=1):

    # Normalize
    r = img / 255.0

    # Apply log transformation
    s = c * np.log(1 + r)

    # Normalize result
    s = s / np.max(s)

    # Convert to 0-255
    s = s * 255

    s = np.clip(s, 0, 255)

    return s.astype(np.uint8)


log_manual = log_without_library(image)


# ============================================================
# SAVE WITHOUT-LIBRARY OUTPUTS
# ============================================================

Image.fromarray(negative_manual).save(
    os.path.join(
        without_library_folder,
        "01_Negative.png"
    )
)

Image.fromarray(gamma_05_manual).save(
    os.path.join(
        without_library_folder,
        "02_Gamma_0.5.png"
    )
)

Image.fromarray(gamma_20_manual).save(
    os.path.join(
        without_library_folder,
        "03_Gamma_2.0.png"
    )
)

Image.fromarray(log_manual).save(
    os.path.join(
        without_library_folder,
        "04_Log_Transformation.png"
    )
)


# ============================================================
# SAVE ORIGINAL IMAGE
# ============================================================

Image.fromarray(image.astype(np.uint8)).save(
    os.path.join(
        without_library_folder,
        "00_Original.png"
    )
)


# ============================================================
# DISPLAY WITHOUT-LIBRARY RESULTS
# ============================================================

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(negative_manual, cmap="gray")
plt.title("Negative")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(gamma_05_manual, cmap="gray")
plt.title("Gamma = 0.5")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(gamma_20_manual, cmap="gray")
plt.title("Gamma = 2.0")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(log_manual, cmap="gray")
plt.title("Log Transformation")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        without_library_folder,
        "05_All_Results_Without_Library.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PART II
# WITH TOOL & LIBRARY
# ============================================================

print("\n" + "=" * 60)
print("PART II - WITH LIBRARY / TOOL")
print("=" * 60)


# ============================================================
# READ IMAGE USING OPENCV
# ============================================================

cv_image = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)


# Check whether image loaded
if cv_image is None:

    raise FileNotFoundError(
        "Image could not be loaded. Check the image path."
    )


# ============================================================
# 1. NEGATIVE USING OPENCV
# ============================================================

negative_library = cv2.bitwise_not(
    cv_image
)


# ============================================================
# 2. GAMMA CORRECTION USING LUT
# ============================================================

def gamma_with_library(img, gamma):

    # Create lookup table
    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    # Apply lookup table
    result = cv2.LUT(
        img,
        table
    )

    return result


gamma_05_library = gamma_with_library(
    cv_image,
    0.5
)

gamma_20_library = gamma_with_library(
    cv_image,
    2.0
)


# ============================================================
# 3. LOG TRANSFORMATION USING NUMPY + OPENCV
# ============================================================

def log_with_library(img):

    # Convert to float
    img_float = np.float32(img)

    # Apply log
    log_img = np.log1p(img_float)

    # Normalize to 0-255
    log_img = cv2.normalize(
        log_img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return np.uint8(log_img)


log_library = log_with_library(
    cv_image
)


# ============================================================
# SAVE LIBRARY OUTPUTS
# ============================================================

cv2.imwrite(
    os.path.join(
        with_library_folder,
        "00_Original.png"
    ),
    cv_image
)

cv2.imwrite(
    os.path.join(
        with_library_folder,
        "01_Negative.png"
    ),
    negative_library
)

cv2.imwrite(
    os.path.join(
        with_library_folder,
        "02_Gamma_0.5.png"
    ),
    gamma_05_library
)

cv2.imwrite(
    os.path.join(
        with_library_folder,
        "03_Gamma_2.0.png"
    ),
    gamma_20_library
)

cv2.imwrite(
    os.path.join(
        with_library_folder,
        "04_Log_Transformation.png"
    ),
    log_library
)


# ============================================================
# DISPLAY LIBRARY RESULTS
# ============================================================

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv_image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(negative_library, cmap="gray")
plt.title("Negative - Library")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(gamma_05_library, cmap="gray")
plt.title("Gamma = 0.5")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(gamma_20_library, cmap="gray")
plt.title("Gamma = 2.0")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(log_library, cmap="gray")
plt.title("Log Transformation")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        with_library_folder,
        "05_All_Results_With_Library.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PART III
# COMPARISON OF WITHOUT LIBRARY VS LIBRARY
# ============================================================

print("\n" + "=" * 60)
print("CREATING COMPARISON RESULTS")
print("=" * 60)


# ============================================================
# NEGATIVE COMPARISON
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(negative_manual, cmap="gray")
plt.title("Negative - Without Library")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(negative_library, cmap="gray")
plt.title("Negative - With Library")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        comparison_folder,
        "01_Negative_Comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# GAMMA COMPARISON
# ============================================================

plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.imshow(gamma_05_manual, cmap="gray")
plt.title("Gamma 0.5 - Without Library")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gamma_05_library, cmap="gray")
plt.title("Gamma 0.5 - With Library")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(gamma_20_manual, cmap="gray")
plt.title("Gamma 2.0 - Without Library")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(gamma_20_library, cmap="gray")
plt.title("Gamma 2.0 - With Library")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        comparison_folder,
        "02_Gamma_Comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# LOG COMPARISON
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(log_manual, cmap="gray")
plt.title("Log - Without Library")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(log_library, cmap="gray")
plt.title("Log - With Library")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        comparison_folder,
        "03_Log_Comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FINAL COMPARISON
# ============================================================

plt.figure(figsize=(16, 10))

plt.subplot(2, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(negative_manual, cmap="gray")
plt.title("Negative")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(gamma_05_manual, cmap="gray")
plt.title("Gamma 0.5")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(gamma_20_manual, cmap="gray")
plt.title("Gamma 2.0")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(log_manual, cmap="gray")
plt.title("Log")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(negative_library, cmap="gray")
plt.title("Negative - Library")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(gamma_05_library, cmap="gray")
plt.title("Gamma - Library")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(log_library, cmap="gray")
plt.title("Log - Library")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        comparison_folder,
        "04_Final_Comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# CREATE README FILE
# ============================================================

readme_text = """
DVP LAB - IMAGE ENHANCEMENT
============================

Image:
REFERENCE IMAGE.jpg

Transformations implemented:

1. Negative
   S = L - 1 - r

2. Power Law / Gamma Correction
   S = c(r^gamma)

3. Log Transformation
   S = c log(1 + r)


PART I - WITHOUT LIBRARY
-------------------------
The mathematical transformation equations are implemented
manually using NumPy array operations.

Outputs:
- Negative
- Gamma = 0.5
- Gamma = 2.0
- Log Transformation


PART II - WITH LIBRARY / TOOL
------------------------------
OpenCV is used for image processing operations.

Outputs:
- Negative
- Gamma = 0.5
- Gamma = 2.0
- Log Transformation


Gamma correction:
- gamma < 1  -> brighter image
- gamma = 1  -> approximately unchanged
- gamma > 1  -> darker image


OUTPUT FOLDERS
--------------

01_Without_Library
    Manual mathematical implementation

02_With_Library
    OpenCV/library implementation

03_Comparison
    Comparison of results


All output images are saved automatically.
"""

with open(
    os.path.join(
        main_output_folder,
        "README.txt"
    ),
    "w"
) as file:

    file.write(readme_text)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("ALL PROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nYour outputs are saved here:")

print(
    "\n1. Without Library:"
)

print(without_library_folder)

print(
    "\n2. With Library:"
)

print(with_library_folder)

print(
    "\n3. Comparisons:"
)

print(comparison_folder)

print(
    "\n4. README:"
)

print(
    os.path.join(
        main_output_folder,
        "README.txt"
    )
)

print("\nDone!")
