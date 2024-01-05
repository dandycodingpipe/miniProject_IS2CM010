#This script is effectively a copy of the 12_19 script but tailored to processing and extracting the rider data set

#12/19 Training Set Script

#Dependencies:
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from skimage import morphology as MM
from scipy.spatial.distance import cdist
from scipy import ndimage as ndi

from fastASF import ImageProcessor, scoreDice, separate_anatomy, diameter_stack, remove_close_maxima

import time

start_time = time.time()

import SimpleITK as sitk
import nibabel as nib

img = 1

from pycad.converters import NrrdToNiftiConverter

# Create an instance of the converter
converter = NrrdToNiftiConverter()

# Define the path to the NRRD file or directory containing NRRD files
input_path = "Project/Rider/R1 (AD)/R1.nrrd"

# Define the destination directory for the converted NIFTI files
output_dir = "Project/Rider/R1 (AD)/R1.nii.gz"

# Run the conversion
converter.convert(input_path, output_dir)
# Load the NIfTI file using nibabel
raw = nib.load(output_dir)
data = raw.get_fdata()

# Create an ImageProcessor object
slice = ImageProcessor(data)

    
slicer, slicer_og1, slicer_og2 = slice.find_aorta(data.shape[2], threshold1=90, threshold2=94)

#ensure image quality
print(slicer.image_arrays)
plt.show()

#####################################################################################

#result data and total segmentator data must be identical in dimension for comparison
reshaped_results = np.flip(np.stack(slicer.image_arrays, axis=-1), axis = -1)
reshaped_results = reshaped_results[:,:,::-1]

#ensure image quality 2
plt.imshow(reshaped_results[:,:,89], cmap = 'gray')

#load Labels
# Load NRRD file
nrrd_filename = f"Project/Rider/R1 (AD)/R1.seg.nrrd"
# Read the NRRD file
image = sitk.ReadImage(nrrd_filename)

# Convert to NIfTI format
nifti_filename = f"Project/Rider/R1 (AD)/R1seg.nii.gz"
nifti_data = sitk.GetArrayFromImage(image)
nifti_affine = image.GetOrigin() + image.GetSpacing()
nifti_image = nib.Nifti1Image(nifti_data, nifti_affine)
nib.save(nifti_image, nifti_filename)


# Load the NIfTI file using nibabel
raw = nib.load(nifti_filename)
data = raw.get_fdata()

groundTruth = nib.load(filename2)
gt_data = ndi.rotate(groundTruth.get_fdata(), -90)

# find and order the masks features properly
binary_gt = np.where(gt_data > 1E-10, 1, 0).astype(float)
binary_gt = binary_gt[:,:,::-1]

####################################################################################

# Create a new "denoised" NIfTI image
new_nifti = nib.Nifti1Image(reshaped_results, raw.affine, header=raw.header)
output_file_path = f"Project/Results/Rider/{img}/wholeMask"
new_nifti.to_filename(output_file_path)

#check that slice information between results and ground truth agree...
n = 50
plt.subplot(1,2,1)
plt.imshow(reshaped_results[:,:,n], cmap = 'gray')

plt.subplot(1,2,2)
plt.imshow(binary_gt[:,:,n], cmap = 'gray')

print(reshaped_results.shape[2])


####################################################################################

descendAo, ascendAo, = separate_anatomy(reshaped_results)

new_nifti = nib.Nifti1Image(descendAo, raw.affine, header=raw.header)
output_file_path = f"Project/Results/Rider/{img}/descendingAorta"
new_nifti.to_filename(output_file_path)

new_nifti = nib.Nifti1Image(ascendAo, raw.affine, header=raw.header)
output_file_path = f"Project/Results/Rider/{img}/ascendingAorta"
new_nifti.to_filename(output_file_path)

print(reshaped_results.shape)
print(gt_data[:,:,0:reshaped_results.shape[2]].shape)
gtdAo, gtAAo = separate_anatomy(binary_gt[:,:,0:(reshaped_results.shape[2])])


#####################################################################################
plt.clf()
# Calculate Dice scores for ascending aorta
ascending_Scores = scoreDice(ascendAo, gtAAo)
plt.plot(range(ascendAo.shape[2]), ascending_Scores, label="Ascending Aorta")

# Calculate Dice scores for descending aorta
descending_Scores = scoreDice(descendAo, gtdAo)
plt.plot(range(descendAo.shape[2]), descending_Scores, label="Descending Aorta")

# Set common labels and title
plt.xlabel("Slice (anatomically superior to inferior)")
plt.ylabel("Dice score")
plt.title(f"Algorithm performance (image 00{img})")

# Add legend
plt.legend()
plt.savefig(f"Project/Results/{img}/algorithm_performance.png")

ascendingDiameters = diameter_stack(gtAAo)
descendingDiameters = diameter_stack(gtdAo)

import pandas as pd

# Determine the maximum length among all arrays
max_length = max(len(ascending_Scores), len(descending_Scores), len(ascendingDiameters), len(descendingDiameters))

# Pad the arrays with NaN to make them the same length
ascending_Scores += [np.nan] * (max_length - len(ascending_Scores))
descending_Scores += [np.nan] * (max_length - len(descending_Scores))
ascendingDiameters += [np.nan] * (max_length - len(ascendingDiameters))
descendingDiameters += [np.nan] * (max_length - len(descendingDiameters))

# Create a DataFrame
df = pd.DataFrame({
    'Slice': range(1, max_length + 1),  # Assuming slices start from 1
    'Ascending_Scores': ascending_Scores,
    'Descending_Scores': descending_Scores,
    'Ascending_Diameters': ascendingDiameters,
    'Descending_Diameters': descendingDiameters
})

# Save the DataFrame to a CSV file
df.to_csv(f'Project/Results/{img}/GT_scores_and_diameters.csv', index=False)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")