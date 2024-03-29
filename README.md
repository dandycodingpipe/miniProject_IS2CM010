# Medical Image Analysis Mini-Project: Automatic Aortic Vessel Segmentation Model

Aortic diseases like aneuryms and aortic disections can both lead to the dilation of arterial vessels. These diseases increase the risk of developing cardiovascular complications such as hemmorage (internal bleeding), stroke, and heart attacks to populations like smokers and the elderly. Radiologists and surgeons need to survey the condition of organs such as the heart and its vessels to evaluate health risks and develop strategies for treating patients. Here we give a simple experimental procedure for segmenting the aorta on python for diameter measurements as well as a qualitative 

<img src="https://github.com/dandycodingpipe/miniProject_IS2CM010/assets/123328325/c1e9245a-ef3f-4b7b-8d1d-6d341877b045" width="400" height="400">
<img src="https://github.com/dandycodingpipe/miniProject_IS2CM010/assets/123328325/114c2e58-9b21-4af0-9ea4-48b73048ffca" width="400" height="400">

## Overview
This script is designed for automatically processing and segmenting chest computed tomography (CT) data from the MM-WHS-2017 (https://zmiclab.github.io/zxh/0/mmwhs/data.html) dataset, but we encourage trying other CT/CT-A datasets, specifically tailored for segmenting the ascending and descending aorta. We combine quantile histogram thresholding with a recursive nearest centroid neighbor search (RNCNS) approach to track and create 3D masks of the ascending and descending aortas.

Our functions, testing script, and data analysis script are provided as is. No additional time was allocated to making it particularly easier on the end-reader other than what was necessary for our own understandinf of our implementation. This was due to time-constraints.

<img src="https://github.com/dandycodingpipe/miniProject_IS2CM010/assets/123328325/14455c51-c23d-48af-a504-7212c8c4572f" width="400" height="450">
(3D-mesh generation via ImageJ)

## Dependencies
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

2. **Ground Truths**: Save and copy the path of the ground truth data. These are available for download in the git. These ground truths were generated by uploading the training data to TotalSegmentator online and using the "aorta" quick-segmentation option. 

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

5. **Outputs**:
   - Processed images in NIfTI format.
   - CSV file containing scores and diameters.
   - Performance graphs as PNG files.

6. **Customization**: Feel free to adjust the thresholds, slice numbers, and other parameters as per your dataset requirements. We achieved much better results when overfitting our data according to individual quantile thresholding variances—which we now recognize could have been used as a feature to optimize using GridSearch CV.

## Note on Performance
The script’s performance and accuracy are dependent on the quality of the input images and the precision of the ground truths provided. Please ensure that the data is pre-processed as needed for optimal results.

![image](https://github.com/dandycodingpipe/miniProject_IS2CM010/assets/123328325/e2d7b032-3de3-445a-81c0-a3dba2782665)

This is an example of the performance output as a PNG for the first image in the dataset.

## Authors
KAMOLE KHOMSI, Lloyd E.
HERNANDEZ-FAJARDO, Christian A.
