#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:45:57 2023

@author: alexandramikhael
"""

import numpy as np

A = [[3,9,5,1],
     [4,25,4,3],
     [63,13,23,9],
    [6,32,77,0],
    [12,8,6,1]];

B = [[0,1,0,1],
     [0,1,1,0],
     [0,0,0,1],
     [1,1,0,1],
     [0,1,0,0]];

"Point-wise multiply A with B and set it to C."

C = np.multiply(A,B);

"Calculate the inner product of the 2nd and 3rd row of C."

ip = np.inner(C[1,:],C[2,:]);

"Find the minimum and maximum values and their corresponding row and column indices in matrix C. "
"If there are multiple min/max values, you must list all their indices."

max_value = np.max(C);
maxpos = np.where(C == max_value);
min_value = np.min(C);
minpos = np.where(C == min_value);
