#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:33:24 2023

@author: alexandramikhael
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import sys


img = cv2.imread("/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW3/white-tower.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.title("input image")
plt.imshow(img)
plt.show()


def createDataset(img):
    h = img.shape[0]
    w = img.shape[1]
    img_reshape = img.reshape((h * w, 3))
    return img_reshape



def kMeansCluster(features, centers):
    idx = np.zeros((features.shape[0],))
    next_centers = np.zeros(centers.shape)
    for iteration in range(100):
        for i in range(features.shape[0]):
            feature = features[i]
            minDist = sys.float_info.max
            for j in range(centers.shape[0]):
                center = centers[j]
                dist = np.linalg.norm(center - feature, 2)
                if minDist > dist:
                    idx[i] = j
                    minDist = dist
        for i in range(next_centers.shape[0]):
            feature_in_i_cluster = features[np.where(idx == i)]
            next_centers[i] = np.mean(feature_in_i_cluster, axis = 0)
#         print(next_centers, centers)
        if np.array_equal(next_centers, centers) == True:
            break
        print(iteration)
        centers = next_centers.copy()
    return idx, centers



def kMeansCluster_new(features, centers):
    idx = np.zeros(features.shape[0])
    next_centers = np.zeros(centers.shape)
    for iteration in range(100):
        minDist = np.ones(features.shape[0]) * sys.float_info.max
        for j in range(centers.shape[0]):
            center = centers[j]
            diff = features - center
            dist = np.apply_along_axis(np.linalg.norm, 1, diff)
            idx_min = np.where(dist < minDist)
            minDist[idx_min] = dist[idx_min]
            idx[idx_min] = j
        for i in range(centers.shape[0]):
            feature_in_i_cluster = features[np.where(idx == i)]
            next_centers[i] = np.mean(feature_in_i_cluster, axis = 0)
        if np.array_equal(next_centers, centers) == True:
            break
        print(iteration)
        centers = next_centers.copy()
    return idx, centers



def mapValues(im, idx, centers):
    img_seg = np.zeros(im.shape).astype(int)
    idx_new = np.reshape(idx, (im.shape[0], im.shape[1]))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            img_seg[i, j, :] = centers[int(idx_new[i][j]), :]
    return img_seg



k = 7
features = createDataset(img)
idx_center = np.random.randint(features.shape[0], size = k)
centers = features[idx_center]


idx, centers = kMeansCluster_new(features, centers)


img_reg = mapValues(img, idx, centers)


print(centers)


plt.imshow(img_reg)
plt.title("image after segmentation")
plt.show()
