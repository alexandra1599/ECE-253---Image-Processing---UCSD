#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 03:05:15 2023

@author: alexandramikhael
"""
import matplotlib.image as Image
import cv2
import skimage
import numpy as np 
from numpy import asarray
from skimage.morphology import disk
import scipy
import matplotlib.pyplot as plt
import pandas

""" For the binary image circles lines.jpg, your aim is to separate out the 
    circles in the image, and calculate certain attributes corresponding to 
    these circles.
    
    1) remove the lines from the image, opening operation
    
"""

cirlines = Image.imread("/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/circles_lines.jpg")
image_array = np.mean(cirlines,axis = 2);
plt.imshow(image_array, cmap='gray')
plt.title('Original Image')

kernel = disk(4);
im_open = skimage.morphology.opening(image_array, kernel);
plt.imshow(im_open, cmap='gray');
plt.title('Opened Image')

th = np.zeros(im_open.shape);
th = np.where(im_open>225.0, 1,0);
plt.imshow(th, cmap='gray')


""" 2) Once we have a binary image with just the circles, the individual 
regions need to be labeled to represent distinct objects in the image i.e. 
connected component labeling. """

label = np.ones((3,3));
th_label, num_features = scipy.ndimage.measurements.label(th,label)

fig = plt.figure();
ax = fig.add_subplot()
conn = ax.imshow(th_label, cmap = 'cool')
ax.set_title("Connected Component Labeling")
fig.colorbar(conn);

"""For each labeled circular region, calculate its centroid and area."""

fig1 = plt.figure();
ax1 = fig1.add_subplot();
conn2 = ax1.imshow(th_label, cmap = 'cool')
centroids = []; areas = [];

for i in range(1, num_features+1):
    p = np.asarray(np.where(th_label == i))
    centroid_p = np.mean(p, axis = 1)
    ax1.scatter(centroid_p[1], centroid_p[0], marker = '*', c = 'r')
    centroids.append(centroid_p)
    areas.append(p.shape[1])

plt.title("Labeled Components")
fig1.colorbar(conn2);

c_i = [i+1 for i in range(num_features)]
c_d = pandas.DataFrame(c_i, columns = ['Circle Index'])
df = pandas.DataFrame(centroids, columns = ['Row', 'Column'])
df2 = pandas.DataFrame(areas, columns = ['Area of Circle'])
df3 = pandas.concat([c_d,df,df2], axis=1)

   
""" In a similar manner to part (i) above, use the opening operation with a 
    suitable structuring element to remove the horizontal and diagonal
    lines in the lines.jpg image."""
fig2 = plt.figure();  
a1 = fig2.add_subplot();
lines = Image.imread('/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/lines.jpg')
image_array1 = np.mean(lines, axis = 2)
a1.imshow(image_array1, cmap='gray')
plt.title('Original Image')

fig3 = plt.figure(); 
a11 = fig3.add_subplot(); 
kernel1 = skimage.morphology.rectangle(10,3)
im1_open = skimage.morphology.opening(image_array1, kernel1)
a11.imshow(im1_open, cmap = 'gray')
plt.title('Opened Image')

th1 = np.where(im1_open>=100,1,0)
plt.imshow(th1, cmap = 'gray')

label1 = np.ones((3,3))
th_label1 , num_features1 = scipy.ndimage.measurements.label(th1,label1)
plt.imshow(th_label1, cmap = 'cool')
plt.colorbar()

line_centroids = []; line_len = [];
for i in range(1, num_features1+1):
    p1 =  p = np.asarray(np.where(th_label1 == i))
    lcentroid_p = np.mean(p, axis = 1)
    linelen = p1[0][-1] - p1[0][0]
    line_centroids.append(lcentroid_p)
    line_len.append(linelen)
    
line_i = [i+1 for i in range(6)]
dfl = pandas.DataFrame(line_i, columns=['Line Index'])
dfl1 =  pandas.DataFrame(line_centroids, columns = ['Row', 'Column'])
dfl2 = pandas.DataFrame(line_len, columns = ['Line Length'])   
dfl3 = pandas.concat([dfl,dfl1,dfl2], axis=1)
                
                