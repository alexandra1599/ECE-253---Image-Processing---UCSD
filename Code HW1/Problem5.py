#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 18:16:09 2023

@author: alexandramikhael
"""

from PIL import Image
import cv2
import skimage
import numpy as np 
from numpy import asarray
import scipy
import math
from skimage.morphology import disk
from PIL import Image, ImageChops


pcb = cv2.imread("/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/pcb.jpg")

"""The image pcb contains an image of a PCB (Printed Circuit Board) that 
needs to be inspected for the number of holes (in this example, 4) and 
their horizontal diameters.

Using binary morphology and other operations, design a system which 
filters the image, leaving only the holes."""

img = Image.open('/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/pcb.jpg')
"""inv_img = ImageChops.invert(img)
inv_img.show()"""

image_array = asarray(img);
kernel_erode = np.ones((1, 15), np.uint8) 
kernel_dilate = disk(4);
image_array = image_array.astype(np.uint8);
img_erosion = cv2.erode(image_array, kernel_erode, iterations=1) 
img_dilation = cv2.dilate(img_erosion, kernel_dilate, iterations=1)
erosion = Image.fromarray(img_erosion);
erosion.show()
dilation = Image.fromarray(img_dilation)
dilation.show()