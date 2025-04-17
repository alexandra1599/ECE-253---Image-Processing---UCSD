#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:17:08 2023

@author: alexandramikhael
"""

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import cv2
import skimage.morphology as skim_morph
import scipy.ndimage.measurements as scipy_im_measure
import pandas as pd
from scipy import signal
import math

def canny_detector(im, thresh):
    im = im/255
    
    # smoothing
    smooth_filt_list = [[2,4,5,4,2],
                   [4,9,12,9,4],
                   [5,12,15,12,5],
                   [4,9,12,9,4],
                   [2,4,5,4,2]]
    smooth_filt = 1/159 * np.asarray(smooth_filt_list)
    smoothed = cv2.filter2D(im,-1,smooth_filt) # what exactly -1 does idk. Ask TA.
    
    
    # find gradients
    
    k_x_list = [[-1,0,1],
               [-2,0,2],
               [-1,0,1]]

    k_y_list = [[-1,-2,-1],
               [0,0,0],
               [1,2,1]]

    k_x = np.asarray(k_x_list)
    k_y = np.asarray(k_y_list)
    
    # we need to convolve image with k_x and k_y
    G_x = cv2.filter2D(im,-1,k_x)
    G_y = cv2.filter2D(im,-1,k_y)
    
    # gradient magnitude
    gradient_image = np.sqrt(G_x**2 + G_y**2)
    plt.figure(figsize = [10,10])
    plt.imshow(gradient_image)
    plt.title('Original Gradient Magnitude Image')
    # gradient angle
    G_x = np.where(G_x == 0, 0.00000001, G_x)
    cont_angles = np.arctan(G_y/G_x)
    
    # NMS
    
    # rounding angles
    disc_angles = np.zeros(cont_angles.shape)
    disc_angles = np.where(cont_angles<(-3*math.pi/8),(-math.pi/2),cont_angles)
    disc_angles = np.where((cont_angles>=(-3*math.pi)/8) & (cont_angles<(-math.pi/8)),-math.pi/4,disc_angles)
    disc_angles = np.where((cont_angles>=-math.pi/8) & (cont_angles<math.pi/8),0,disc_angles)
    disc_angles = np.where((cont_angles>=math.pi/8) & (cont_angles<3*math.pi/8),math.pi/4,disc_angles)
    disc_angles = np.where(cont_angles>=3*math.pi/8 ,math.pi/2,disc_angles)
    
    # pad mirroring
    gradient_padded = np.pad(gradient_image, 1, mode = 'symmetric')
    disc_angle_padded = np.pad(disc_angles, 1, mode = 'symmetric')
    
    supressed = np.zeros(gradient_padded.shape)


    for i in range(1,supressed.shape[0]-1):
        row_num = i
        for j in range(1,supressed.shape[1]-1):
            col_num = j

            # above is correct 100%

            angle = disc_angle_padded[row_num, col_num]
            magnitude = gradient_padded[row_num, col_num]

            if(angle == 0):
                if((magnitude<gradient_padded[row_num, col_num-1]) or (magnitude<gradient_padded[row_num, col_num+1])):
                    supressed[row_num, col_num] = 0
                else:
                    supressed[row_num, col_num] = gradient_padded[row_num, col_num]

            if(angle == math.pi/2 or angle == -math.pi/2):
                if(magnitude<gradient_padded[row_num-1, col_num] or magnitude<gradient_padded[row_num+1, col_num]):
                    supressed[row_num, col_num] = 0
                else:
                    supressed[row_num, col_num] = gradient_padded[row_num, col_num]

            if(angle == math.pi/4):
                if(magnitude<gradient_padded[row_num-1, col_num+1] or magnitude<gradient_padded[row_num+1, col_num-1]):
                    supressed[row_num, col_num] = 0
                else:
                    supressed[row_num, col_num] = gradient_padded[row_num, col_num]
            if(angle == -math.pi/4):
                if(magnitude<gradient_padded[row_num+1, col_num+1] or magnitude<gradient_padded[row_num-1, col_num-1]):
                    supressed[row_num, col_num] = 0
                else:
                    supressed[row_num, col_num] = gradient_padded[row_num, col_num]
                    
    plt.figure(figsize = [10,10])
    plt.imshow(supressed)
    plt.title('Image After NMS')
    # thresholding    
    thresholded_supressed = np.where(supressed<thresh,0,1)
    
    return(thresholded_supressed)

g_sel_color = cv2.imread('/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/geisel.jpg')
g_sel = cv2.cvtColor(g_sel_color, cv2.COLOR_BGR2GRAY)
canny_image_geisel = canny_detector(g_sel, 0.9)

plt.figure(figsize = [10,10])
plt.imshow(canny_image_geisel)
plt.title('Final edge image after thresholding')