# Deep Learning Aortic Image Processing Script

## Overview
This script is designed for processing and extracting data from the rider dataset, specifically tailored for analyzing aortic images in medical imaging studies. It is an adaptation of the `12_19 Training Set Script`, focusing on multi-layer perceptron models and the backpropagation principle.

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
- `fastASF` package (custom package for specific image processing functions)

The script also requires access to specific medical image files in NRRD and NIfTI formats. Make sure these files are correctly placed in your project directory.

## Usage Instructions
1. **Modify File Paths**: Replace the `input_path` and `output_dir` variables with the correct path to your NRRD and NIfTI files. The script currently uses the following placeholders:
   - NRRD file: `"Project/Rider/R1 (AD)/R1.nrrd"`
   - NIfTI file: `"Project/Rider/R1 (AD)/R1.nii.gz"`

2. **Use Aortic Ground Truths**: The script requires aortic ground truth images for accurate processing. These ground truths should be provided in your Git repository. Load them using the appropriate file paths.

3. **Calling `fastASF` Functions**: The script utilizes several functions from the `fastASF` package for image processing:
   - `ImageProcessor`
   - `scoreDice`
   - `separate_anatomy`
   - `diameter_stack`
   - `remove_close_maxima`

   Make sure the `fastASF` package is correctly installed and imported.

4. **Running the Script**: Execute the script to process the images. The script performs the following key operations:
   - Converts NRRD files to NIfTI format.
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

## Contact
For any queries or issues related to the script, please contact [Your Contact Information].
