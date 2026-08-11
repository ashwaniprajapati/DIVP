Lab 2 - Image Enhancement Transformation Functions
==================================================

Aim:
Apply image-enhancement transformation functions manually and with reusable helper/tool functions.

Transformations:
1. Negative: S = L - 1 - r
2. Power Law / Gamma Correction: S = c(r^gamma)
3. Log Transformation: S = c log(1 + r)

How to run:
python3 "DVP/Lab 2/Code/lab2_image_enhancement.py"

Input dataset:
DVP/Dataset/Lab 2/sample.ppm

Output folder:
DVP/Lab 2/Image Output/

Implementation details:
- Manual implementation uses direct loops over every pixel value.
- Tool/library-style implementation uses reusable Image helper methods and map_pixels.
- The code is dependency-free and uses only Python standard library modules.

Result:
The program creates output files for original, manual negative, manual gamma, manual log, tool negative, tool gamma, and tool log transformations.

What I learned:
- Negative transformation reverses intensity levels and highlights bright/dark regions oppositely.
- Gamma correction changes image brightness non-linearly.
- Log transformation expands darker pixel values and compresses brighter values.
- Manual implementation helps understand formulas at pixel level.
- Helper functions reduce repeated code and make experiments easier.
