import numpy as np
import random
import ConfigParser

from help_functions import *

from pre_processing import my_PreProc

def get_data_training(DRIVE_train_imgs_original, DRIVE_train_groudTruth, patch_height, patch_width, N_subimgs, inside_FOV):
    train_imgs_original = load_hdf5(DRIVE_train_imgs_original)
    train_masks = load_hdf5(DRIVE_train_groudTruth) #masks always the same

    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks/255.

    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print "\ntrain images/masks shape:"
    print train_imgs.shape
    print "train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs))
    print "train masks are within 0-1\n"

    #extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train = extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    data_consistency_check(patches_imgs_train, patches_masks_train)

    print "\ntrain PATCHES images/masks shape:"
    print patches_imgs_train.shape
    print "train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train))

    return patches_imgs_train, patches_masks_train#, patches_imgs_test, patches_masks_test