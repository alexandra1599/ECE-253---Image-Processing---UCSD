#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 22:43:16 2023

@author: alexandramikhael
"""

import numpy as np
import imageio
import matplotlib.pyplot as plt


def ChromaKeying(img):
    s1 = np.zeros(img.shape[0:2])
    s2 = np.zeros(img.shape)
    s3 = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i][j][1] > 120 and ((img[i][j][0] < 180 and img[i][j][2] < 100)
                    or (img[i][j][0] < 100 and img[i][j][2] < 180))):
                s1[i][j] = 0
                s2[i][j] = [0, 0, 0]
                s3[i][j] = [255, 2, 90]
            else:
                s1[i][j] = 255
                s2[i][j] = img[i][j]
                s3[i][j] = img[i][j]
    return s1.astype(int), s2.astype(int), s3.astype(int)


trav = imageio.imread('/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/travolta.jpg',pilmode="RGB")

trav1, trav2, trav3 = ChromaKeying(trav)

f2 = plt.figure(figsize=(14,8))
f2_ax1 = f2.add_subplot(231)
f2_ax2 = f2.add_subplot(232)
f2_ax3 = f2.add_subplot(233)

f2_ax1.imshow(trav1, cmap = plt.get_cmap('gray'))
f2_ax1.title.set_text("(i)")
f2_ax2.imshow(trav2)
f2_ax2.title.set_text("(ii)")
f2_ax3.imshow(trav3)
f2_ax3.title.set_text("(iii)")
plt.show()
plt.savefig("Problem4_Travolta.png")