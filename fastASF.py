import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from skimage import morphology as MM
from scipy.spatial.distance import cdist
from scipy import ndimage as ndi

class ImageProcessor:

    def __init__(self, data):
        self.data = data
        self.z = data.shape[2] - 1
        self.slider = []
        self.slider_og1 = []
        self.slider_og2 = []

    def remove_close_maxima(self, maxima, distance_threshold=2):
        """
        Calculate distances between each pair of points in the given list.
        Optionally remove one of the two points that are too close to each other.

        Parameters:
        - points (list): List of points, where each point is represented as a list or array.
        - distance_threshold (float or None): If provided, remove one of the two points that are closer than this threshold.

        Returns:
        - distances (numpy.ndarray): Matrix of distances between each pair of points.
        - filtered_points (list): List of points after removing one of the two points that are too close (if distance_threshold is provided).
        """
        # Convert the list of points to a NumPy array
        points_array = np.array(maxima)

        # Calculate pairwise distances using cdist
        distances = cdist(points_array, points_array)

        # Optionally remove one of the two points that are too close
        indices_to_remove = set()
        if distance_threshold is not None:
            for i in range(len(maxima)):
                for j in range(len(maxima)):
                    if i != j and distances[i, j] < distance_threshold and j not in indices_to_remove:
                        # Add the indices to the set of indices to remove
                        indices_to_remove.add(i)

        # Use a list comprehension to create a new list with points to keep
        filtered_points = [point for i, point in enumerate(maxima) if i not in indices_to_remove]

        return distances, filtered_points

    def keep_farthest_points(self, points):
        """
        Keep only the two points that are farthest away from each other.

        Parameters:
        - points (list): List of points, where each point is represented as a list or array.

        Returns:
        - kept_points (list): List of the two points that are farthest away from each other.
        """
        # Convert the list of points to a NumPy array
        points_array = np.array(points)

        # Calculate pairwise distances using cdist
        distances = cdist(points_array, points_array)

        # Find the indices of the two farthest points
        max_distance_indices = np.unravel_index(np.argmax(distances), distances.shape)

        # Extract the two farthest points
        farthest_points = [points[max_distance_indices[0]], points[max_distance_indices[1]]]

        return farthest_points
    
    def generate_lung_mask(self, index, threshold_percentile=75):
        slice = ndi.rotate(self.data[:, :, index], -90)
        slice2 = slice > np.percentile(slice, threshold_percentile)
        lung_mask = slice * slice2
        return lung_mask

    def remove_artifacts(self, lung_mask):
        seed = MM.dilation(lung_mask, MM.disk(10))
        reconstruction = MM.reconstruction(seed, lung_mask, method='erosion')
        return reconstruction

    def preprocess_pipeline(self, index, reconstruction, threshold_percentile=95):
        slice_og = ndi.rotate(self.data[:, :, index], -90)
        slice2 = reconstruction > np.percentile(slice_og, threshold_percentile)
        preprocessed_data = reconstruction * slice2
        return preprocessed_data

    def distance_transform(self, lung_mask):
        return ndi.distance_transform_edt(lung_mask)

    def find_maxima(self, edt, threshold):
        maxima = MM.h_maxima(edt, threshold)
        return np.where(maxima == 1)

    def find_closest_maxima(self, ao_cent, list_max):
        list_dist = [np.sqrt(((ao_cent[0] - point[0]) ** 2) + ((ao_cent[1] - point[1]) ** 2)) for point in list_max]
        min_index = np.argmin(list_dist)
        ao_cent_new = list_max[min_index]
        return ao_cent_new

    def create_map_maxima(self, ao1_cent, ao2_cent, mono=0):
        map = np.zeros((512,512))
        if mono == 1:
            map[ao1_cent[0], ao1_cent[1]] = 1
        else:
            map[ao1_cent[0], ao1_cent[1]] = 1
            map[ao2_cent[0], ao2_cent[1]] = 1
        return map

    def find_aorta_beginning(self, threshold1, threshold2): #thresholdd

        lung_mask = self.generate_lung_mask(self.z)
        reconstruction = self.remove_artifacts(lung_mask)

        slice_og = self.preprocess_pipeline(self.z, reconstruction,threshold_percentile=threshold1)
        preprocessed_data = self.preprocess_pipeline(self.z, reconstruction,threshold_percentile=threshold2)
        
        edt = self.distance_transform(preprocessed_data)
        maxima_coords = self.find_maxima(edt, 9)
        list_max = np.array([maxima_coords[0], maxima_coords[1]]).T.tolist()
        distances, filtered_points = self.remove_close_maxima(list_max, distance_threshold=2)
        cent = self.keep_farthest_points(filtered_points)

        new_ao1_cent = cent[0]
    
        new_ao2_cent = cent[1]
    
        map_maxima_beg = self.create_map_maxima(new_ao1_cent, new_ao2_cent)

        return (new_ao1_cent, new_ao2_cent, map_maxima_beg, slice_og)

    def find_aorta_cyle(self, index, ao1_cent, ao2_cent, threshold=95, mono=0):
        lung_mask = self.generate_lung_mask(self.z - index)
        reconstruction = self.remove_artifacts(lung_mask)
        preprocessed_data = self.preprocess_pipeline(self.z - index, reconstruction, threshold_percentile=threshold)
        edt = self.distance_transform(preprocessed_data)
        maxima_coords = self.find_maxima(edt, 9)
        list_max = np.array([maxima_coords[0], maxima_coords[1]]).T.tolist()
        new_ao1_cent = self.find_closest_maxima(ao1_cent, list_max)
        new_ao2_cent = self.find_closest_maxima(ao2_cent, list_max)
        map_maxima = self.create_map_maxima(new_ao1_cent, new_ao2_cent, mono)
        return(new_ao1_cent, new_ao2_cent, map_maxima, preprocessed_data)

    def find_aorta(self, num, inte=100, threshold1=95, threshold2=95):
        new_ao1_cent, new_ao2_cent, map_maxima_beg, slice_og = self.find_aorta_beginning(threshold1, threshold2)
        self.slider.append(MM.reconstruction(map_maxima_beg, slice_og, method='dilation'))
        self.slider_og1.append(slice_og)
        self.slider_og2.append(ndi.rotate(self.data[:, :, self.z], -90))
        for index in range(1, num):
            print(f'Iteration number : {index}')
            mono = 0
            if index > inte:
                mono = 1
            new_ao1_cent, new_ao2_cent, map_maxima, preprocessed_data = self.find_aorta_cyle(index, new_ao1_cent, new_ao2_cent,threshold=threshold1, mono=mono)
            self.slider.append(MM.reconstruction(map_maxima, preprocessed_data, method='dilation'))
            self.slider_og1.append(preprocessed_data)
            self.slider_og2.append(ndi.rotate(self.data[:, :, self.z - index], -90))
        self.slider = ImageSlider(self.slider)
        self.slider_og1 = ImageSlider(self.slider_og1)
        self.slider_og2 = ImageSlider(self.slider_og2)
        return(self.slider, self.slider_og1, self.slider_og2)



class ImageSlider:
    def __init__(self, image_arrays):
        self.image_arrays = image_arrays
        self.current_index = 0

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        self.img_plot = self.ax.imshow(self.image_arrays[self.current_index], cmap='gray')

        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, 'Image', 0, len(self.image_arrays) - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.update_image)

    def update_image(self, val):
        self.current_index = int(self.slider.val)
        self.img_plot.set_array(self.image_arrays[self.current_index])
        self.fig.canvas.draw_idle()


def scoreDice(ourMask, theirMask, z = 0):
    dices = []

    if z == 0:
        z=ourMask.shape[2]

    for slice in range(z):
        # Flatten the 2D arrays
        n = slice
        flat_results = ourMask[:, :, n].flatten()
        flat_binary = theirMask[:, :, n].flatten()

        # Calculate intersection, union, and Dice score
        intersection = np.sum(flat_results * flat_binary)
        union = np.sum(flat_results) + np.sum(flat_binary)
        dice = 2 * intersection / union
        dices.append(dice)

        #print("Dice Score for index:",slice,"---", dice)
    
    print(len(dices))
    print("Mean dice:", np.sum(dices)/len(dices))
    return(dices)


def separate_anatomy(mask):
    """
    The goal of generate_diameters() is to separate the final masks into two separate files containing either ascending or descending aorta...
    This will help us calculate separate dice scores to pinpoint areas of improvement as well as easily run measurments on a particular anatomical feature.
    """
    # We will store the hyperstack of our individual features in their own 3D array 
    ascendAo = np.zeros_like(mask)
    descendAo = np.zeros_like(mask)

    #iterate in 3D
    for slice in range(mask.shape[2]):
        
        #generate maxima centroids
        rawMaxima = MM.h_maxima(ndi.distance_transform_edt(mask[:,:,slice]), 2)

        #extract centroid coordinates
        coordinates = np.where(rawMaxima == 1)
        coordinates = np.asarray(coordinates).T
    
        #this will ideally extract 2 coordinates. the first will be "higher" and we know thats the descending aorta :)
        distance, cent = remove_close_maxima(coordinates, distance_threshold = 9)
        #reformatting 
        cent = np.asarray(cent)
        #print(cent.shape)
   
        #store vessel coordinates in their respective 2d-arrays
        descendAo[cent[0][0],cent[0][1],slice] = 1
        #print(cent[0][0],cent[0][1], slice)

        #the ascending aorta will not always be present in the coordinates, therefore we need this conditional boundary preventing crashes
        if(cent.shape[0] > 1):
            ascendAo[cent[1][0],cent[1][1],slice] = 1
            #print(cent[1][0],cent[1][1], slice)
    
        #reconstruct vessels in their respective 2d-arrays (anatomical class separation)
        descendAo[:,:,slice] = MM.reconstruction(descendAo[:,:,slice], mask[:,:,slice], method = 'dilation')
        ascendAo[:,:,slice] = MM.reconstruction(ascendAo[:,:,slice], mask[:,:,slice], method = 'dilation')

    return(descendAo, ascendAo)
   
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt

def diameter_stack(mask):

    avg_diameters = []
    for slice in range(mask.shape[2]):
        contours = measure.find_contours(mask[:,:, slice], 0.5, positive_orientation='low')

        # Calculate diameters for each contour
        for contour in contours:
            # Calculate pairwise distances between points on the contour
            distances = np.linalg.norm(np.subtract(contour[:, None, :], contour[None, :, :]), axis=-1)
            
            # Exclude self-distances and find the maximum distance as the diameter
            diameter = np.mean(distances[~np.eye(len(contour), dtype=bool)])
            scaled_diameter = diameter * 0.75
            avg_diameters.append(round(diameter/10, 2))

    # Plot the contour for verification
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)

    return(avg_diameters)

def remove_close_maxima(points, distance_threshold=None):
    """
    Calculate distances between each pair of points in the given list.
    Optionally remove one of the two points that are too close to each other.

    Parameters:
    - points (list): List of points, where each point is represented as a list or array.
    - distance_threshold (float or None): If provided, remove one of the two points that are closer than this threshold.

    Returns:
    - distances (numpy.ndarray): Matrix of distances between each pair of points.
    - filtered_points (list): List of points after removing one of the two points that are too close (if distance_threshold is provided).
    """
    # Convert the list of points to a NumPy array
    points_array = np.array(points)

    # Calculate pairwise distances using cdist
    distances = cdist(points_array, points_array)

    # Optionally remove one of the two points that are too close
    indices_to_remove = set()
    if distance_threshold is not None:
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j and distances[i, j] < distance_threshold and j not in indices_to_remove :
                    # Add the indices to the set of indices to remove
                    indices_to_remove.add(i)

    # Use a list comprehension to create a new list with points to keep
    filtered_points = [point for i, point in enumerate(points) if i not in indices_to_remove]

    return distances, filtered_points



