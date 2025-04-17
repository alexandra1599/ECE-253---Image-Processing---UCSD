#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 19:59:18 2023

@author: alexandramikhael
"""

def MeanFilter(image, filter_size):
    output = np.zeros(image.shape, np.uint8)
    result = 0
    for j in range(1, image.shape[0]-1):
        for i in range(1, image.shape[1]-1):
            for y in range(-1, 2):
                for x in range(-1, 2):
                    result = result + image[j+y, i+x]
            output[j][i] = int(result / filter_size)
            result = 0

    return output


from scipy import ndimage
from scipy import signal
import matplotlib.pyplot as plt
import cv2
from numpy import asarray
import numpy as np
from PIL import Image, ImageFilter  


noisy1 = Image.open("/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/noisy1.png")
noisy2 = Image.open("/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/noisy2.png")
sungod = Image.open("/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/sungod.png")


"Use a 1x3 median filter to clean both images. What are the resulting MSEs?"

image1_1x3 = signal.medfilt2d(noisy1, kernel_size=(1,3))
E = Image.fromarray(image1_1x3)
E.save('/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/noisy1_median_filtered_1_3.png')
Y = np.square(np.subtract(noisy1,image1_1x3)).mean()
print("MSE1 median 1x3 :", Y)

image2_1x3 = signal.medfilt2d(noisy2, kernel_size=(1,3))
E2 = Image.fromarray(image2_1x3)
E2.save('/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/noisy2_median_filtered_1_3.png')
Y = np.square(np.subtract(noisy2,image2_1x3)).mean()
print("MSE2 median 1x3 :", Y)

"Use a 3x3 median filter to clean both images. What are the resulting MSEs?"

image1_3x3 = noisy1.filter(ImageFilter.MedianFilter(size = 3))  
"image_3x3.show() "
image1_3x3.save('/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/noisy1_median_filtered_3_3.png')
Y = np.square(np.subtract(noisy1,image1_3x3)).mean()
print("MSE1 median 3x3 :", Y)

image2_3x3 = noisy2.filter(ImageFilter.MedianFilter(size = 3))  
"image_3x3.show() "
image2_3x3.save('/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/noisy2_median_filtered_3_3.png')
Y = np.square(np.subtract(noisy2,image2_3x3)).mean()
print("MSE2 median 3x3:", Y)

"Use a 3x3 mean filter to clean both images. What are the resulting MSEs?"

noisy1 = cv2.imread('/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/noisy1.png', 0)
noisy2 = cv2.imread('/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/noisy2.png', 0)

mean1_3x3 = MeanFilter(noisy1, 9)
E2 = Image.fromarray(mean1_3x3)
E2.save('/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/noisy1_mean_filtered.png')
Y = np.square(np.subtract(noisy1,mean1_3x3)).mean()
print("MSE1 mean filter :", Y)

mean2_3x3 = MeanFilter(noisy2, 9)
E2 = Image.fromarray(mean2_3x3)
E2.save('/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/noisy2_mean_filtered.png')
Y = np.square(np.subtract(noisy1,mean2_3x3)).mean()
print("MSE2 mean filter :", Y)

