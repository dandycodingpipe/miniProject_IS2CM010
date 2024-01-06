# Medical Image Analysis Mini-Project: Automatic Aortic Vessel Segmentation Model

![image](https://github.com/dandycodingpipe/miniProject_IS2CM010/assets/123328325/14455c51-c23d-48af-a504-7212c8c4572f)

## Overview
This script is designed for processing and extracting data from the MM-WHS-2017 (https://zmiclab.github.io/zxh/0/mmwhs/data.html) dataset, but we encourage trying other CT/CT-A datasets, specifically tailored for segmenting the ascending and descending aorta.

## Prerequisites
Before running this script, ensure you have the following dependencies installed:
- `nibabel`
- `matplotlib`
- `numpy`
- `scikit-image`
- `scipy`
- `SimpleITK`
- `pycad`
- `pandas`
- `fastASF` fast Aortic Segmentation Framework package (our custom functions available in the git)

The script also requires access to specific medical image files in NRRD (not currently supported) and NIfTI formats. Make sure these files are correctly placed in your project directory.

## Usage Instructions
1. **Modify File Paths**: Replace the `input_path` and `output_dir` variables with the correct path to your NIfTI files. The script currently uses the following placeholders:
   - NIfTI file: `"Project/Rider/R1 (AD)/R1.nii.gz"`

2. **Use Aortic Ground Truths**: The script requires aortic ground truth images for evaluation. These are available for download in the git. These ground truths were generated using TotalSegmentator.

3. **The script utilizes several functions from the `fastASF` package for image processing:
   - `ImageProcessor`
   - `scoreDice`
   - `separate_anatomy`
   - `diameter_stack`
   - `remove_close_maxima`

   Make sure the `fastASF` package is correctly installed and imported.

4. **Running the Script**: 
   - Processes images using `ImageProcessor`.
   - Calculates and compares Dice scores.
   - Saves processed images and results in specified directories.

5. **Output**: The script outputs:
   - Processed images in NIfTI format.
   - CSV file containing scores and diameters.
   - Performance graphs as PNG files.

6. **Customization**: Feel free to adjust the thresholds, slice numbers, and other parameters as per your dataset requirements.

## Note on Performance
The script’s performance and accuracy are dependent on the quality of the input images and the precision of the ground truths provided. Please ensure that the data is pre-processed as needed for optimal results.

## Authors
KAMOLE KHOMSI, Lloyd E.
HERNANDEZ-FAJARDO, Christian A.
