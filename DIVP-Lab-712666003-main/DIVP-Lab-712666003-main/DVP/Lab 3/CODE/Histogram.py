import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------
# 1. Read image and convert to grayscale
# ---------------------------------------------------------

def read_gray(filename):
    img = Image.open(filename).convert("L")
    return np.array(img)


# ---------------------------------------------------------
# 2. Calculate histogram
# ---------------------------------------------------------

def histogram(img):
    hist = np.zeros(256, dtype=int)

    for pixel in img.ravel():
        hist[pixel] += 1

    return hist


# ---------------------------------------------------------
# 3. Calculate CDF
# ---------------------------------------------------------

def calculate_cdf(hist):

    cdf = np.cumsum(hist)

    # Normalize CDF to [0, 1]
    cdf = cdf / cdf[-1]

    return cdf


# ---------------------------------------------------------
# 4. Histogram Matching
# ---------------------------------------------------------

def histogram_matching(source, reference):

    source_hist = histogram(source)
    reference_hist = histogram(reference)

    source_cdf = calculate_cdf(source_hist)
    reference_cdf = calculate_cdf(reference_hist)

    # Mapping table
    mapping = np.zeros(256, dtype=np.uint8)

    for source_intensity in range(256):

        # CDF value of source intensity
        s = source_cdf[source_intensity]

        # Find reference intensity with closest CDF
        difference = np.abs(reference_cdf - s)

        mapping[source_intensity] = np.argmin(difference)

    # Apply mapping
    matched = mapping[source]

    return matched


# ---------------------------------------------------------
# 5. Display histogram and CDF
# ---------------------------------------------------------

def show_histogram_cdf(images, names):

    plt.figure(figsize=(14, 10))

    # Histograms
    plt.subplot(2, 1, 1)

    for img, name in zip(images, names):

        hist = histogram(img)

        plt.plot(hist, label=name)

    plt.title("Histograms")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()

    # CDFs
    plt.subplot(2, 1, 2)

    for img, name in zip(images, names):

        hist = histogram(img)
        cdf = calculate_cdf(hist)

        plt.plot(cdf, label=name)

    plt.title("CDFs")
    plt.xlabel("Intensity")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()



reference = read_gray("E:\Download\REFERENCE IMAGE.jpg")
frame1 = read_gray("E:\Download\FRAME1.jpg")
frame2 = read_gray("E:\Download\FRAME2.jpg")

matched1 = histogram_matching(frame1, reference)
matched2 = histogram_matching(frame2, reference)

plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(reference, cmap="gray")
plt.title("Reference")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(frame1, cmap="gray")
plt.title("Frame 1")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(frame2, cmap="gray")
plt.title("Frame 2")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(matched1, cmap="gray")
plt.title("Frame 1 Matched")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(matched2, cmap="gray")
plt.title("Frame 2 Matched")
plt.axis("off")

plt.tight_layout()
plt.show()

#Compare their histogram
show_histogram_cdf(
    [reference, frame1, frame2, matched1, matched2],
    [
        "Reference",
        "Original Frame 1",
        "Original Frame 2",
        "Matched Frame 1",
        "Matched Frame 2"
    ]
)

def create_target_histogram():

    x = np.arange(256)

    mean = 70
    sigma = 35

    target = np.exp(
        -((x - mean) ** 2) /
        (2 * sigma ** 2)
    )

    # Convert to integer histogram
    target = target / target.sum()

    return target


def match_to_target_histogram(source, target_probability):

    source_hist = histogram(source)
    source_cdf = calculate_cdf(source_hist)

    target_cdf = np.cumsum(target_probability)
    target_cdf = target_cdf / target_cdf[-1]

    mapping = np.zeros(256, dtype=np.uint8)

    for i in range(256):

        difference = np.abs(
            target_cdf - source_cdf[i]
        )

        mapping[i] = np.argmin(difference)

    return mapping[source]

target_hist = create_target_histogram()

moody = match_to_target_histogram(
    frame1,
    target_hist
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(frame1, cmap="gray")
plt.title("Original Frame")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(moody, cmap="gray")
plt.title("Stylized Moody Result")
plt.axis("off")

plt.tight_layout()
plt.show()

plt.plot(target_hist)
plt.title("Analytical Moody Target Histogram")
plt.xlabel("Intensity")
plt.ylabel("Probability")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------
# Read grayscale image
# ---------------------------------------------------------

def read_gray(reference):

    img = Image.open(reference).convert("L")

    return np.array(img)


# ---------------------------------------------------------
# Histogram
# ---------------------------------------------------------

def calculate_histogram(tile):

    hist = np.zeros(256, dtype=int)

    for pixel in tile.ravel():

        hist[pixel] += 1

    return hist


# ---------------------------------------------------------
# Clip histogram
# ---------------------------------------------------------

def clip_histogram(hist, clip_limit):

    excess = 0

    # Clip bins
    for i in range(256):

        if hist[i] > clip_limit:

            excess += hist[i] - clip_limit

            hist[i] = clip_limit

    # Redistribute excess pixels
    redistribution = excess // 256
    remainder = excess % 256

    hist += redistribution

    # Distribute remaining pixels
    for i in range(remainder):

        hist[i] += 1

    return hist


# ---------------------------------------------------------
# Create mapping from histogram
# ---------------------------------------------------------

def histogram_mapping(hist):

    cdf = np.cumsum(hist)

    # Find first non-zero CDF
    nonzero = np.nonzero(cdf)[0]

    if len(nonzero) == 0:

        return np.arange(256)

    cdf_min = cdf[nonzero[0]]
    total = cdf[-1]

    mapping = (
        (cdf - cdf_min) /
        (total - cdf_min + 1e-10)
        * 255
    )

    mapping = np.clip(mapping, 0, 255)

    return mapping.astype(np.uint8)

def clahe_basic(image, tile_size=(8, 8), clip_factor=2.0):

    height, width = image.shape

    tile_h, tile_w = tile_size

    output = np.zeros_like(image)

    for y in range(0, height, tile_h):

        for x in range(0, width, tile_w):

            y_end = min(y + tile_h, height)
            x_end = min(x + tile_w, width)

            tile = image[y:y_end, x:x_end]

            hist = calculate_histogram(tile)

            # Calculate clip limit
            tile_pixels = tile.size

            clip_limit = max(
                1,
                int(
                    clip_factor *
                    tile_pixels /
                    256
                )
            )

            hist = clip_histogram(
                hist,
                clip_limit
            )

            mapping = histogram_mapping(hist)

            output[y:y_end, x:x_end] = mapping[tile]

    return output

image = read_gray(r"E:\Download\xrayiamge.jpg")

clahe_result = clahe_basic(
    image,
    tile_size=(32, 32),
    clip_factor=2.0
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original X-ray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(clahe_result, cmap="gray")
plt.title("Tile-based CLAHE")
plt.axis("off")

plt.tight_layout()
plt.show()


#compare global equalization and clahe
def global_equalization(image):

    hist = calculate_histogram(image)

    mapping = histogram_mapping(hist)

    return mapping[image]


global_result = global_equalization(image)

clahe_result = clahe_basic(
    image,
    tile_size=(32, 32),
    clip_factor=2.0
)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(global_result, cmap="gray")
plt.title("Global Histogram Equalization")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(clahe_result, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

plt.tight_layout()
plt.show()

#Full CLAHE implementation with interpolation
def clahe(image, tile_rows=8, tile_cols=8, clip_factor=2.0):

    height, width = image.shape

    tile_h = int(np.ceil(height / tile_rows))
    tile_w = int(np.ceil(width / tile_cols))

    # Mapping for every tile
    mappings = []

    for i in range(tile_rows):

        row_maps = []

        for j in range(tile_cols):

            y1 = i * tile_h
            y2 = min((i + 1) * tile_h, height)

            x1 = j * tile_w
            x2 = min((j + 1) * tile_w, width)

            tile = image[y1:y2, x1:x2]

            hist = calculate_histogram(tile)

            clip_limit = max(
                1,
                int(
                    clip_factor *
                    tile.size /
                    256
                )
            )

            hist = clip_histogram(
                hist,
                clip_limit
            )

            mapping = histogram_mapping(hist)

            row_maps.append(mapping)

        mappings.append(row_maps)

    output = np.zeros_like(image, dtype=np.uint8)

    # -----------------------------------------------------
    # Bilinear interpolation
    # -----------------------------------------------------

    for y in range(height):

        for x in range(width):

            # Position in tile grid
            gy = y / tile_h - 0.5
            gx = x / tile_w - 0.5

            y0 = int(np.floor(gy))
            x0 = int(np.floor(gx))

            dy = gy - y0
            dx = gx - x0

            # Clamp indices
            y0 = max(0, min(y0, tile_rows - 1))
            x0 = max(0, min(x0, tile_cols - 1))

            y1 = min(y0 + 1, tile_rows - 1)
            x1 = min(x0 + 1, tile_cols - 1)

            intensity = image[y, x]

            # Four neighboring mappings
            v00 = mappings[y0][x0][intensity]
            v01 = mappings[y0][x1][intensity]
            v10 = mappings[y1][x0][intensity]
            v11 = mappings[y1][x1][intensity]

            # Bilinear interpolation
            value = (
                (1 - dy) * (1 - dx) * v00 +
                (1 - dy) * dx       * v01 +
                dy       * (1 - dx) * v10 +
                dy       * dx       * v11
            )

            output[y, x] = np.clip(
                value,
                0,
                255
            )

    return output


image = read_gray(r"E:\Download\xrayiamge.jpg")

result = clahe(
    image,
    tile_rows=8,
    tile_cols=8,
    clip_factor=2.0
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original X-ray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("CLAHE with Bilinear Interpolation")
plt.axis("off")

plt.tight_layout()
plt.show()


image = read_gray(r"E:\Download\xrayiamge.jpg")

global_result = global_equalization(image)

clahe_result = clahe(
    image,
    tile_rows=8,
    tile_cols=8,
    clip_factor=2.0
)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(global_result, cmap="gray")
plt.title("Global HE")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(clahe_result, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

plt.hist(
    image.ravel(),
    bins=256,
    range=(0, 255),
    density=True,
    alpha=0.5,
    label="Original"
)

plt.hist(
    global_result.ravel(),
    bins=256,
    range=(0, 255),
    density=True,
    alpha=0.5,
    label="Global HE"
)

plt.hist(
    clahe_result.ravel(),
    bins=256,
    range=(0, 255),
    density=True,
    alpha=0.5,
    label="CLAHE"
)

plt.xlabel("Intensity")
plt.ylabel("Probability")
plt.title("Histogram Comparison")
plt.legend()
plt.show()
