'''
Mini-Project Main file
Julia Graham
20173309
'''

#import libraries
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

#import functions from Utils
from Utils import prostate_segmenter, resample_to_reference, plot_overlay, seg_eval_dice, pixel_extract

def main():
    '''This function utilizes several helper functions defined in Utils.py.
    It takes an MRI image and a radiologist defined prostate mask. It applies
    segmentation to create a new prostate mask, and plots the overlay of this
    segmentation against the MRI. It also plots the radiologist's mask against
    the MRI. It calculates the DICE coefficient of the segmentation vs the
    gold standard mask. It creates a boxplot of pixel intensity of a region
    of interest in the MRI'''

    # read in external files
    #replace filepaths with your own
    gs_mask = sitk.ReadImage("C:/Users/julia/PycharmProjects/pythonProject/10522_1000532_segmentation.nii")
    img = sitk.ReadImage("C:/Users/julia/PycharmProjects/pythonProject/10522_1000532_t2wMRI.mha")

    # resample gs_mask to have the same spacing as the MRI image
    gs_res = resample_to_reference(gs_mask, img)

    # perform CTIF region growing with fine tuned parameters
    seed_point = [(452, 5489, 13), (578, 523, 13), (518, 452, 13)]
    lower_thresh = 50
    upper_thresh = 150
    seg = prostate_segmenter(img, lower_thresh, upper_thresh, seed_point)

    # save segmentation
    #replace destination filepath with your own
    sitk.WriteImage(seg, 'C:/Users/julia/PycharmProjects/pythonProject/my_segmentation.nrrd')

    plot_overlay(img, seg, gs_res)

    # get dice coefficient
    dsc_coef = seg_eval_dice(seg, gs_res)
    print(f"The dice coefficient is: {dsc_coef}")

    # extract pixel
    point = (5, 25, 30)
    width = 6
    pixel_extract(img, point, width)

if __name__ == "__main__":
    main()