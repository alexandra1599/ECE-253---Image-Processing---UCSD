#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:10:14 2023

@author: alexandramikhael
"""

"Download any color image from the Internet or use one of the given images."
"Read this image and call it A."

import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image

"Transform the color image to gray-scale. Verify the values are between 0 and 255."
" If not, please normalize your image from 0 to 255. Call this image B."

A = Image.open("/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/geisel.jpg")
B = A.convert('L')
B.save("/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/geisel_gray.jpg")
"B.show()"

B = np.asarray(B)

"Add 15 to each value of image B. Set all pixel values greater than 255 to 255. Call this image C."

C = B + 15
C[C > 255] = 255

" Flip image B along both the horizontal and vertical axis. Call this image D."

D = np.flip(np.flip(C, 0),1)

"Calculate the median of all values in image B. "

median = np.median(B)
E = np.zeros(B.shape)
smaller = B <= median
larger = B > median
E[smaller] = 1
E[larger] = 0

"Plot the images "

fig1 = plt.figure(figsize=(12,6))
fig1_ax1 = fig1.add_subplot(2,3,1)
fig1_ax2 = fig1.add_subplot(2,3,2)
fig1_ax3 = fig1.add_subplot(2,3,3)
fig1_ax4 = fig1.add_subplot(2,3,4)
fig1_ax5 = fig1.add_subplot(2,3,5)

fig1_ax1.imshow(A)
fig1_ax1.title.set_text("Image A")
fig1_ax2.imshow(B, cmap = plt.get_cmap('gray'))
fig1_ax2.title.set_text("Image B")
fig1_ax3.imshow(C, cmap = plt.get_cmap('gray'))
fig1_ax3.title.set_text("Image C")
fig1_ax4.imshow(D, cmap = plt.get_cmap('gray'))
fig1_ax4.title.set_text("Image D")
fig1_ax5.imshow(E, cmap = plt.get_cmap('gray'))
fig1_ax5.title.set_text("Image E")
plt.show()
plt.savefig("Poblem2_images.png")
 

 

