#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 18:54:10 2023

@author: alexandramikhael
"""
import numpy as np
import imageio
import matplotlib.pyplot as plt



def compute_norm_rgb_histogram(im):
    Red = [];
    Green = [];
    Blue = [];
    R = im[:,:,0]
    G = im[:,:,1]
    B = im[:,:,2]
    for i in range(32):
        Rnb = R[R >= i * 8]
        Rnb = Rnb[Rnb <= (i+1) * 8 -1]
        Red.append(len(Rnb))
        
        Gnb = G[G >= i * 8]
        Gnb = Gnb[Gnb <= (i+1) * 8 -1]
        Green.append(len(Gnb))
        
        Bnb = B[B >= i * 8]
        Bnb = Bnb[Bnb <= (i+1) * 8 -1]
        Blue.append(len(Bnb))
    RGB = Red + Green + Blue
    final_RGB = sum(RGB)
    RGB = [x / final_RGB for x in RGB] 
    
    return RGB

image = imageio.imread('/Users/alexandramikhael/Desktop/MASTERS/Fall 2023/ECE 253 - TRIVEDI/HW1/geisel.jpg',pilmode="RGB")
RGB = compute_norm_rgb_histogram(image)

color = ['red' , 'green', 'blue']
c = [];

for i in range(32):
    c.append(color[0])
    
for i in range(32):
    c.append(color[1])
    
for i in range(32):
    c.append(color[2])
    
plt.bar(range(0, 96), RGB,color=c)
plt.title("Normalized RGB Histogram")



plt.show()
plt.savefig("Problem3_histogram.png")
