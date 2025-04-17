#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:55:31 2023

@author: alexandramikhael
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2



def AHE(img, win_size):
    height = img.shape[1]
    width = img.shape[0]
    
    pad_w = int(win_size / 2)
    pad_h = win_size - pad_w - 1
    img_pad = np.pad(img, (pad_w, pad_h), mode = 'symmetric')
    
    output = np.zeros((width, height))
    for x in range(0, width):
        for y in range(0, height):
            rank = 0
            pixel = img_pad[x + pad_w][y + pad_h]
            context_reg = img_pad[x : x + win_size, y : y + win_size]
            rank = np.sum(context_reg < pixel)
            output[x][y] = int(rank * 255 / win_size ** 2)
            
    return output

img = plt.imread("data/beach.png")
win_sizes = [33, 65, 129]
i = 1

for win_size in win_sizes:
    output = AHE(img, win_size)

    plt.imshow(output, cmap = 'gray')
    plt.savefig("q1_AHE_%s.jpg"%i)
    i += 1
    plt.show()
    

img = cv2.imread("data/beach.png",0)
equ = cv2.equalizeHist(img)
print(equ)
plt.imshow(equ, cmap='gray')
plt.savefig("p1_HE.jpg")
