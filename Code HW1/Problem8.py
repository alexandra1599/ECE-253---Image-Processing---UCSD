#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 22:52:28 2023

@author: alexandramikhael
"""

import glob
import cv2
import matplotlib.pyplot as plt

image1 = cv2.imread("/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/geisel.jpg")
image2 = cv2.imread("/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/Vincent.jpg")

images = [image1 , image2]

"""
"1 "
scale_percent = [0.3, 0.5, 0.7]

i = 1
for im in images:
    reference = plt.figure(figsize=(14,16))
    referenceax0 = reference.add_subplot(431)
    referenceax0.imshow(im)   
    referenceax0.title.set_text("Original Image")
    n = 4;
    for s in scale_percent:
        w = int(im.shape[1] * s)
        h = int(im.shape[0] * s)
        dim = (w, h)
        nearest_method = cv2.resize(im, dim, interpolation = cv2.INTER_NEAREST)
        linear_method = cv2.resize(im, dim, interpolation = cv2.INTER_LINEAR)
        bicubic_method = cv2.resize(im, dim, interpolation = cv2.INTER_CUBIC)
        first = reference.add_subplot(4,3,n)
        second = reference.add_subplot(4,3,n+1)
        third = reference.add_subplot(4,3,n+2)
        n += 3
        first.title.set_text("NN Method : %s"%s)
        first.imshow(nearest_method)
        second.title.set_text("Bilinear Method : %s"%s)
        second.imshow(linear_method)
        third.title.set_text("Bicubic Method : %s"%s)
        third.imshow(bicubic_method)
    plt.savefig("question3_%s"%i)
    i += 1"""
    
"""    
scale_percent = [1.5, 1.7, 2.0]


i = 1
for im in images:
    reference = plt.figure(figsize=(12,16))
    referenceax0 = reference.add_subplot(431)
    referenceax0.imshow(im)   
    referenceax0.title.set_text("Original Image")
    n = 4;
    for s in scale_percent:
        w = int(im.shape[1] * s)
        h = int(im.shape[0] * s)
        dim = (w, h)
        nearest_method = cv2.resize(im, dim, interpolation = cv2.INTER_NEAREST)
        linear_method = cv2.resize(im, dim, interpolation = cv2.INTER_LINEAR)
        bicubic_method = cv2.resize(im, dim, interpolation = cv2.INTER_CUBIC)
        
        cropNd3 = nearest_method[800:900, 100:150]
        cropNd2 = linear_method[800:900, 100:150]
        cropNd = bicubic_method[800:900, 100:150]
        
        first = reference.add_subplot(4,3,n)
        second = reference.add_subplot(4,3,n+1)
        third = reference.add_subplot(4,3,n+2)
        n += 3
        first.title.set_text("NN Method : %s"%s)
        first.imshow(cropNd3)
        second.title.set_text("Bilinear Method : %s"%s)
        second.imshow(cropNd2)
        third.title.set_text("Bicubic Method : %s"%s)
        third.imshow(cropNd)
    plt.savefig("question4_%s"%i)
    i += 1
"""  
i = 1

image2 =  cv2.imread("/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/Vincent.jpg")
images = [image1 , image2]
for im in images:
    reference = plt.figure(figsize=(12,16))
    referenceax0 = reference.add_subplot(431)
    referenceax0.imshow(im)   
    referenceax0.title.set_text("original image")
    
    s = 0.1
    w = int(im.shape[1] * s)
    h = int(im.shape[0] * s)
    dim = (w, h)
    original_dim = (im.shape[1], im.shape[0])
    
    nearest_method = cv2.resize(im, dim, interpolation = cv2.INTER_NEAREST)
    linear_method = cv2.resize(im, dim, interpolation = cv2.INTER_LINEAR)
    bicubic_method = cv2.resize(im, dim, interpolation = cv2.INTER_CUBIC)
    
    renearest = cv2.resize(nearest_method, original_dim, interpolation = cv2.INTER_NEAREST)
    relinear = cv2.resize(linear_method, original_dim, interpolation = cv2.INTER_LINEAR)
    rebicubic = cv2.resize(bicubic_method, original_dim, interpolation = cv2.INTER_CUBIC)
    
    nearest_linear = cv2.resize(nearest_method, original_dim, interpolation = cv2.INTER_LINEAR)
    nearest_cubic = cv2.resize(nearest_method, original_dim, interpolation = cv2.INTER_CUBIC)
    linear_cubic = cv2.resize(linear_method, original_dim, interpolation = cv2.INTER_CUBIC)
    linear_nearest = cv2.resize(linear_method, original_dim, interpolation = cv2.INTER_NEAREST)
    cubic_linear = cv2.resize(bicubic_method, original_dim, interpolation = cv2.INTER_LINEAR)
    cubic_nearest = cv2.resize(bicubic_method, original_dim, interpolation = cv2.INTER_NEAREST)
    
    cropNd3 = nearest_linear[800:900, 100:150]
    cropNd2 = nearest_cubic[800:900, 100:150]
    cropNd = linear_cubic[800:900, 100:150]
    cropNd4 = linear_nearest[800:900, 100:150]
    cropNd5 = cubic_linear[800:900, 100:150]
    cropNd6 = cubic_nearest[800:900, 100:150]
    
    referenceax = reference.add_subplot(431)
    referenceax1 = reference.add_subplot(432)
    referenceax2 = reference.add_subplot(433)
    referenceax3 = reference.add_subplot(434)
    referenceax4 = reference.add_subplot(435)
    referenceax5 = reference.add_subplot(436)
    referenceax6 = reference.add_subplot(437)


    referenceax.title.set_text("Original Image:")
    referenceax.imshow(im)
    referenceax1.title.set_text("NN - Linear Method :")
    referenceax1.imshow(cropNd3)
    referenceax2.title.set_text("NN - Bicubic Method :")
    referenceax2.imshow(cropNd2)
    referenceax3.title.set_text("Linear - Bicubic Method :")
    referenceax3.imshow(cropNd)
    referenceax4.title.set_text("Linear - NN Method :")
    referenceax4.imshow(cropNd4)
    referenceax5.title.set_text("Bicubic - Linear Method :")
    referenceax5.imshow(cropNd5)
    referenceax6.title.set_text("Bicubic - M=NN Method :")
    referenceax6.imshow(cropNd6)
    plt.savefig("question5_%s.jpg"%i)
    i += 1

