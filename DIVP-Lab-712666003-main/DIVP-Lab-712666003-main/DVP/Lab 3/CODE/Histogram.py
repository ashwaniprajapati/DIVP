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
  
