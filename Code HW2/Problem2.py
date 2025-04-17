#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:58:50 2023

@author: alexandramikhael
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from lloyds import lloyds
import glob


path = "/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW2/*.tif"
imgs = []

for file in glob.glob(path):
    imgs.append(cv2.imread(file, 0))
    
plt.imshow(imgs[1], cmap = 'gray')
plt.show()


def bit_change(img, original, out):
    change = 2 ** (original - out)
    img_new = img // change * change
    return img_new


def lloyds_img(img, original, out):
    [M, N] = img.shape;
    img = img.astype(float)
    img_new_l = np.zeros(img.shape)
    training_set = np.reshape(img,N*M,1);
    partition, codebook = lloyds(training_set, [2**out])
    for i in range(M):
        for j in range(N):
            if (img[i][j] < partition[0]):
                img_new_l[i][j] = codebook[0];
            else:
                cdist = [img[i][j] - p for p in partition]
                cmin = min(i for i in cdist if i >= 0)
                minIndex = cdist.index(cmin)
                img_new_l[i][j] = codebook[minIndex + 1]
    return img_new_l


def MSE(img, img_new):
    size = img.shape[0] * img.shape[1]
    return np.linalg.norm(img_new.astype(float) - img.astype(float), 2)  / size


def mse_compute(img):

    ss = np.arange(1, 8, 1)
    mses_bit_change = []
    mses_lloyds = []

    for s in ss:
        img_new_bit_change = bit_change(img, 8, s)
        mse_bit_change = MSE(img, img_new_bit_change)
        mses_bit_change.append(mse_bit_change)

        img_new_l = lloyds_img(img, 8, s)
        mse_lloyds = MSE(img, img_new_l)
        mses_lloyds.append(mse_lloyds)
    
    return mses_bit_change, mses_lloyds



msess_bit_change = []
msess_lloyds = []

for img in imgs:    
    mses_bit_change, mses_lloyds = mse_compute(img)
    msess_bit_change.append(mses_bit_change)
    msess_lloyds.append(mses_lloyds)



f = plt.figure(figsize = (12, 4))
ss = np.arange(1, 8, 1)

ax1 = f.add_subplot(121)
ax1.title.set_text("MSE of lena512.tif")
ax1.plot(ss, msess_bit_change[0], color='orange', label='bit_change')
ax1.plot(ss, msess_lloyds[0], color='green', label='lloyds')
ax1.legend() 
ax1.set_xlabel('bit')
ax1.set_ylabel('MSE')

ax2 = f.add_subplot(122)
ax2.title.set_text("MSE of diver.tif")
ax2.plot(ss, msess_bit_change[1], color='orange', label='bit_change')
ax2.plot(ss, msess_lloyds[1], color='green', label='lloyds')
ax2.legend() 
ax2.set_xlabel('bit')
ax2.set_ylabel('MSE')
plt.savefig("p3_noHE.jpg")

plt.show()



imgs_histequs = []
for img in imgs:
    imgs_histequs.append(cv2.equalizeHist(img))
    
msess_histequ_bit_change = []
msess_histequ_lloyds = []

for img in imgs_histequs:    
    mses_bit_change, mses_lloyds = mse_compute(img)
    msess_histequ_bit_change.append(mses_bit_change)
    msess_histequ_lloyds.append(mses_lloyds)



f = plt.figure(figsize = (12, 4))
ss = np.arange(1, 8, 1)

ax1 = f.add_subplot(121)
ax1.title.set_text("MSE of lena512.tif after histgram equalization")
ax1.plot(ss, msess_histequ_bit_change[0], color='orange', label='bit_change')
ax1.plot(ss, msess_histequ_lloyds[0], color='green', label='lloyds')
ax1.legend() 
ax1.set_xlabel('bit')
ax1.set_ylabel('MSE')

ax2 = f.add_subplot(122)
ax2.title.set_text("MSE of diver.tif after histgram equalization")
ax2.plot(ss, msess_histequ_bit_change[1], color='orange', label='bit_change')
ax2.plot(ss, msess_histequ_lloyds[1], color='green', label='lloyds')
ax2.legend() 
ax2.set_xlabel('bit')
ax2.set_ylabel('MSE')
plt.savefig("p3_HE.jpg")

plt.show()
