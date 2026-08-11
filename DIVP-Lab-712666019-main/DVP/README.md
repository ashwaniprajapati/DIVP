# Digital Image/Video Processing (DVP) Labs

This repository contains a GitHub-ready structure for DVP practical submissions.

## Structure

- `Dataset/` stores text-based ASCII PPM input images for each lab.
- `Lab 1/` contains basic image-processing operations and text-based SVG result images.
- `Lab 2/` contains image-enhancement transformation functions and text-based SVG result images.
- `Lab 3/` and `Lab 4/` are prepared for future practical work.

## What I learned

- How digital images are represented as width, height, channels, and pixel intensity values.
- How to perform basic image operations such as grayscale conversion, brightness/contrast adjustment, resize, rotate, flip, crop, negative conversion, and saving processed images.
- How image-enhancement transformation functions change pixel intensities.
- How to compare manual pixel-by-pixel implementation with reusable helper/tool-based implementation.
- How to organize lab code, datasets, output images, and README files for GitHub submission.

## PR compatibility

All generated datasets and result images are stored as plain-text `.ppm` or `.svg` files so GitHub PR creation does not fail because of binary image files.
