#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:07:46 2023

@author: alexandramikhael
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import axes_location
import math

## 2(i)
img = cv2.imread("Car.tif", 0)
img_pad = np.zeros((512, 512))
p = int((512 - img.shape[0]) / 2)
q = int((512 - img.shape[1]) / 2)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img_pad[i + p][j + q] = img[i][j]

f = np.fft.fft2(img_pad)
fshift = np.fft.fftshift(f)
log_m_spectrum = np.log(np.abs(fshift))

uk = [91, 176, 344, 429]
vk = [168, 166, 166, 169]
uk  = uk - np.ones((4, )) * 256
vk = vk - np.ones((4, )) * 256
print(uk, vk)
x_axis = np.linspace(-256,255,512)
y_axis = np.linspace(-256,255,512)
[v,u] = np.meshgrid(x_axis,y_axis)

img_h = np.zeros(img_pad.shape)
n = 2
d0 = 20
for i in range(img_pad.shape[0]):
    for j in range(img_pad.shape[1]):
        h = 1
        uc = u[i][j]
        vc = v[i][j]
        for k in range(4):
            dk = np.sqrt(np.square(uc - uk[k]) + np.square(vc - vk[k]))
            d_k = np.sqrt(np.square(uc + uk[k]) + np.square(vc + vk[k]))
            if (dk == 0 or d_k == 0):
                h *= 0
            else:
                h *= 1/(1 + math.pow(d0 / dk, 2 * n)) * 1/(1 + math.pow(d0 / d_k, 2 * n))
        img_h[i][j] = h

img_no_shift = np.fft.ifftshift(fshift * img_h)
img_b = np.abs(np.fft.ifft2(img_no_shift))
img_b -= img_b.min()
img_b = img_b[p : img_b.shape[0] - p, q : img_b.shape[1] - q] * 256 / img_b.max()

f = plt.figure(figsize=(14,14))
f_ax1 = f.add_subplot(221)
f_ax2 = f.add_subplot(222)
f_ax3 = f.add_subplot(223)
f_ax4 = f.add_subplot(224)

img1_1 = f_ax1.imshow(img, cmap = 'gray')
f_ax1.title.set_text("unpadded original image")
divider = axes_location(f_ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(img1_1, cax, orientation='vertical')

img1_2 = f_ax2.imshow(log_m_spectrum, cmap = 'gray')
f_ax2.title.set_text("the corresponding 2D DFT log-magnitude")
divider = axes_location(f_ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(img1_2, cax, orientation='vertical')

img2_1 = f_ax3.imshow(img_h, cmap = 'gray')
f_ax3.title.set_text("the butterworth Notch Reject Filter in frequency domain HNR(u, v)")
divider = axes_location(f_ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(img2_1, cax, orientation='vertical')

img2_2 = f_ax4.imshow(img_b, cmap = 'gray')
f_ax4.title.set_text("the final filtered image")
divider = axes_location(f_ax4)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(img2_2, cax, orientation='vertical')
plt.savefig("result_car.jpg")
plt.show()


#  2(ii)

img = cv2.imread("Street.png", 0)
img_pad = np.zeros((512, 512))
p = int((512 - img.shape[0]) / 2)
q = int((512 - img.shape[1]) / 2)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img_pad[i + p][j + q] = img[i][j]

f= np.fft.fft2(img_pad)
fshift = np.fft.fftshift(f)
log_m_spectrum = np.log(np.abs(fshift))

uk = [256, 90]
vk = [90, 256]
uk  = uk - np.ones((2, )) * 256
vk = vk - np.ones((2, )) * 256
print(uk, vk)
x_axis = np.linspace(-256,255,512)
y_axis = np.linspace(-256,255,512)
[v,u] = np.meshgrid(x_axis,y_axis)


img_h = np.zeros(img_pad.shape)
n = 2
d0 = 50
for i in range(img_pad.shape[0]):
    for j in range(img_pad.shape[1]):
        h = 1
        uc = u[i][j]
        vc = v[i][j]
        for k in range(2):
            dk = np.sqrt(np.square(uc - uk[k]) + np.square(vc - vk[k]))
            d_k = np.sqrt(np.square(uc + uk[k]) + np.square(vc + vk[k]))
            if (dk == 0 or d_k == 0):
                h *= 0
            else:
                h *= 1/(1 + math.pow(d0 / dk, 2 * n)) * 1/(1 + math.pow(d0 / d_k, 2 * n))
        img_h[i][j] = h

img_no_shift = np.fft.ifftshift(fshift * img_h)
img_b = np.abs(np.fft.ifft2(img_no_shift))
img_b -= img_b.min()
img_b = img_b[p : img_b.shape[0] - p, q : img_b.shape[1] - q] * 256 / img_b.max()

f = plt.figure(figsize=(14,14))
f_ax1 = f.add_subplot(221)
f_ax2 = f.add_subplot(222)
f_ax3 = f.add_subplot(223)
f_ax4 = f.add_subplot(224)

img1_1 = f_ax1.imshow(img, cmap = 'gray')
f_ax1.title.set_text("unpadded original image")
divider = axes_location(f_ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(img1_1, cax, orientation='vertical')

img1_2 = f_ax2.imshow(log_m_spectrum, cmap = 'gray')
f_ax2.title.set_text("the corresponding 2D DFT log-magnitude")
divider = axes_location(f_ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(img1_2, cax, orientation='vertical')

img2_1 = f_ax3.imshow(img_h, cmap = 'gray')
f_ax3.title.set_text("the butterworth Notch Reject Filter in frequency domain HNR(u, v)")
divider = axes_location(f_ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(img2_1, cax, orientation='vertical')

img2_2 = f_ax4.imshow(img_b, cmap = 'gray')
f_ax4.title.set_text("the final filtered image")
divider = axes_location(f_ax4)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(img2_2, cax, orientation='vertical')
plt.savefig("result_street.jpg")
plt.show()