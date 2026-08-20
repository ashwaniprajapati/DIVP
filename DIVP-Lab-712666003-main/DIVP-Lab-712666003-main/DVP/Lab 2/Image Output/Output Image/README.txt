

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

E:\Download\butterflyimage.jfif


All output folders are checked to ensure
their size remains below 25 MB.

