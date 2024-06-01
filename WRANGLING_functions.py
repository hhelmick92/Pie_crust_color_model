# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:21:35 2023

@author: hhelmick
"""

import cv2
import random

import numpy as np
from scipy.optimize import curve_fit

def linear_fit(x, m, b):
    return m*x + b

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def rand_slope(l_values):
    
    rand = random.randrange(1, l_values.shape[0])
    strip = l_values[rand-1:rand,0:l_values.shape[1],0]
    strip2 = l_values[0:l_values.shape[0], rand-1: rand, 0].flatten()

    x_axis = np.arange(0,len(strip[0]))
    r = curve_fit(linear_fit, x_axis, strip[0])
    slope = r[0][0]

    above = []
    mean = strip[0].mean()
    std = strip[0].std()
    ms = mean+std
    for n in strip[0]:
        if n > ms:
            above.append(n)
        
    x_axis2 = np.arange(0,len(strip2))
    r2 = curve_fit(linear_fit, x_axis2, strip2)
    slope2 = r2[0][0]

    above2 = []
    mean2 = strip2.mean()
    std2 = strip2.std()
    ms2 = mean2 + std2
    for n in strip2:
        if n > ms2:
            above2.append(n)

    slope2 = r2[0][0]

    mean_slope = np.mean(slope)
    mean_slope2 = np.mean(slope2)
    mean_above = np.mean(len(above))
    mean_above2 = np.mean(len(above2))

    return [mean_slope, mean_above, mean_slope2, mean_above2]

def get_lab(cropped_in):
        
    lab2 = cv2.cvtColor(cropped_in, cv2.COLOR_BGR2LAB) # convert the BGR image to LAB color space
    
    out = []
    
    l2 = lab2[:, :, 0]
    a2 = lab2[:, :, 1]
    b2 = lab2[:, :, 2]
    
    l3 = l2
    out.append(np.nanmean(l3)/ (255/100)) # standard conversion factor for 8-bit images
    out.append(np.nanmax(l3)/ (255/100)) # standard conversion factor for 8-bit images
    out.append(np.nanmin(l3)/ (255/100)) # standard conversion factor for 8-bit images
    out.append(np.nanstd(l3)/ (255/100)) # standard conversion factor for 8-bit image
    
    a3 = a2
    out.append(np.nanmean(a3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanmax(a3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanmin(a3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanstd(a3) - 128) # standard conversion factor for 8-bit images

    b3 = b2
    out.append(np.nanmean(b3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanmax(b3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanmin(b3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanstd(b3) - 128) # standard conversion factor for 8-bit images

    B1 = cropped_in[:,:,0]
    G1 = cropped_in[:,:,1]
    R1 = cropped_in[:,:,2]
    
    out.append(np.nanmean(B1))
    out.append(np.nanmax(B1))
    out.append(np.nanmin(B1))
    out.append(np.nanstd(B1))

    out.append(np.nanmean(G1))
    out.append(np.nanmax(G1))
    out.append(np.nanmin(G1))
    out.append(np.nanstd(G1))
    
    out.append(np.nanmean(R1))
    out.append(np.nanmax(R1))
    out.append(np.nanmin(R1))
    out.append(np.nanstd(R1))

    mean_slope =[]
    mean_slope2 =[]
    mean_above =[]
    mean_above2 =[]
    
    for i in list(range(0,10)):
        s = rand_slope(lab2)
        mean_slope.append(s[0])
        mean_slope2.append(s[1])
        mean_above.append(s[2])
        mean_above2.append(s[3])
    
    mean_slope = np.mean(mean_slope)
    mean_slope2 = np.mean(mean_slope2)
    mean_above = np.mean(mean_above)
    mean_above2 = np.mean(mean_above2)

    out.append(mean_slope)
    out.append(mean_slope2)
    out.append(mean_above)
    out.append(mean_above2)

    return out

def get_lab_v2(cropped_in):
        
    cropped_in = white_balance(cropped_in)
    lab2 = cv2.cvtColor(cropped_in, cv2.COLOR_BGR2LAB) # convert the BGR image to LAB color space
    
    out = []
    
    l2 = lab2[:, :, 0]
    a2 = lab2[:, :, 1]
    b2 = lab2[:, :, 2]
    
    l3 = l2
    out.append(np.nanmean(l3)/ (255/100)) # standard conversion factor for 8-bit images
    out.append(np.nanmax(l3)/ (255/100)) # standard conversion factor for 8-bit images
    out.append(np.nanmin(l3)/ (255/100)) # standard conversion factor for 8-bit images
    out.append(np.nanstd(l3)/ (255/100)) # standard conversion factor for 8-bit image
    
    a3 = a2
    out.append(np.nanmean(a3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanmax(a3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanmin(a3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanstd(a3) - 128) # standard conversion factor for 8-bit images

    b3 = b2
    out.append(np.nanmean(b3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanmax(b3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanmin(b3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanstd(b3) - 128) # standard conversion factor for 8-bit images

    B1 = cropped_in[:,:,0]
    G1 = cropped_in[:,:,1]
    R1 = cropped_in[:,:,2]
    
    out.append(np.nanmean(B1))
    out.append(np.nanmax(B1))
    out.append(np.nanmin(B1))
    out.append(np.nanstd(B1))

    out.append(np.nanmean(G1))
    out.append(np.nanmax(G1))
    out.append(np.nanmin(G1))
    out.append(np.nanstd(G1))
    
    out.append(np.nanmean(R1))
    out.append(np.nanmax(R1))
    out.append(np.nanmin(R1))
    out.append(np.nanstd(R1))

    mean_slope =[]
    mean_slope2 =[]
    mean_above =[]
    mean_above2 =[]
    
    for i in list(range(0,10)):
        s = rand_slope(lab2)
        mean_slope.append(s[0])
        mean_slope2.append(s[1])
        mean_above.append(s[2])
        mean_above2.append(s[3])
    
    mean_slope = np.mean(mean_slope)
    mean_slope2 = np.mean(mean_slope2)
    mean_above = np.mean(mean_above)
    mean_above2 = np.mean(mean_above2)

    out.append(mean_slope)
    out.append(mean_slope2)
    out.append(mean_above)
    out.append(mean_above2)

    return out
