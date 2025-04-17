#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:06:16 2023

@author: alexandramikhael
"""
import math
import Image

def apply_threshold(self, value):
        "Returns 0 or 255 depending where value is closer"
        return 255 * math.floor(value/128) 

def floyd_steinberg_dither(self, image_file):

        img = Image.open(image_file)
        new_img = img.convert('RGB')
        pixel = new_img.load()

        x_lim, y_lim = new_img.size

        for y in range(1, y_lim):
            for x in range(1, x_lim):
                r_old, g_old, b_old = pixel[x, y]

                r_new = self.apply_threshold(r_old)
                g_new = self.apply_threshold(g_old)
                b_new = self.apply_threshold(b_old)

                pixel[x, y] = r_new, g_new, b_new

                r_error = r_old - r_new
                b_error = b_old - b_new
                g_error = g_old - g_new

                if x < x_lim - 1:
                    red = pixel[x+1, y][0] + round(r_error * 7/16)
                    green = pixel[x+1, y][1] + round(g_error * 7/16)
                    blue = pixel[x+1, y][2] + round(b_error * 7/16)

                    pixel[x+1, y] = (red, green, blue)

                if x > 1 and y < y_lim - 1:
                    red = pixel[x-1, y+1][0] + round(r_error * 3/16)
                    green = pixel[x-1, y+1][1] + round(g_error * 3/16)
                    blue = pixel[x-1, y+1][2] + round(b_error * 3/16)

                    pixel[x-1, y+1] = (red, green, blue)

                if y < y_lim - 1:
                    red = pixel[x, y+1][0] + round(r_error * 5/16)
                    green = pixel[x, y+1][1] + round(g_error * 5/16)
                    blue = pixel[x, y+1][2] + round(b_error * 5/16)

                    pixel[x, y+1] = (red, green, blue)

                if x < x_lim - 1 and y < y_lim - 1:
                    red = pixel[x+1, y+1][0] + round(r_error * 1/16)
                    green = pixel[x+1, y+1][1] + round(g_error * 1/16)
                    blue = pixel[x+1, y+1][2] + round(b_error * 1/16)

                    pixel[x+1, y+1] = (red, green, blue)

