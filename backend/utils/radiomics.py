import numpy as np
from skimage import io, measure


def extract_radiomic_features(image_path, mask_path):
   
    #Read and mask
    img = io.imread(image_path, as_gray=True)
    msk = io.imread(mask_path, as_gray=True) > 0

    #Tag
    labels = measure.label(msk.astype(int))
    props_list = measure.regionprops(labels, intensity_image=img)
    if not props_list:
        #If doesnt find any region return zeros vector
        return np.zeros(5, dtype=np.float32)

    #take the fist tagged region
    props = props_list[0]

    #extract
    area = props.area
    perimeter = props.perimeter
    eccentricity = props.eccentricity
    mean_int = props.mean_intensity
    #contrast
    contrast = float(np.var(img[msk]))

    return np.array([area, perimeter, eccentricity, mean_int, contrast], dtype=np.float32)
