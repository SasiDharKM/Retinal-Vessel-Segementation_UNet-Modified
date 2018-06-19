import numpy as np
from PIL import Image
import cv2

from help_functions import *


## Histogram Equalisation (not used, can be used)
def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)

# adaptive histogram equalization is used. 
# In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV).
# Then each of these blocks are histogram equalized as usual. 
# So in a small area, histogram would confine to a small region (unless there is noise).
# If noise is there, it will be amplified. 
# To avoid this, contrast limiting is applied. 
# If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), 
# those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. 
# After equalization, to remove artifacts in tile borders, bilinear interpolation is applied

def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized

