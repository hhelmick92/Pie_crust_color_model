# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:34:34 2023

@author: hhelmick
"""

import numpy as np
import pandas as pd

import cv2

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.optimize import curve_fit

import cv2
import random

import numpy as np
from scipy.optimize import curve_fit


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

def linear_fit(x, m, b):
    return m*x + b

l_conv = 255/100

def image_features(img_in, h_graph = False, v_graph = False):
    
    a_size = 800
    b_size = 725
    
    im = cv2.resize(img_in, (a_size,b_size))
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB) # DIFFERENT CONVERSION FROM CV2

    features = []

    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    
    l1 =np.where(l == 0, np.nan, l)
    a1 =np.where(a == 0, np.nan, a)
    b1 =np.where(b == 0, np.nan, b)

    B = im[:,:,0]
    G = im[:,:,1]
    R = im[:,:,2]
    
    features.append(np.nanmean(l1) / l_conv)
    features.append(np.nanmean(a1 -128))
    features.append(np.nanmean(b1 -128))
    features.append(np.mean(B))
    features.append(np.mean(G))
    features.append(np.mean(R))


    strip = lab[:,:,0]
    
    y = np.arange(0, l1.shape[0])
    x = np.arange(0, l1.shape[1])
    
    left_slope = []
    left_cept = []
    
    right_slope = []
    right_cept = []
    
    for n in list(range(len(y))):
        strip = lab[n,:,0]
        
        left = strip[0: round((len(strip)/ 2))]
        right = strip[(round(len(strip)/ 2)) :]
        
        fit = curve_fit(linear_fit, (x[0:round((len(strip)/ 2))]), (left/l_conv))
        left_slope.append(fit[0][0])
        left_cept.append(fit[0][1])
        
        fit = curve_fit(linear_fit, (x[round((len(strip)/ 2)) : ]), (right/l_conv))
        right_slope.append(fit[0][0])
        right_cept.append(fit[0][1])
          
    features.append(np.mean(left_slope))
    features.append(np.mean(right_slope))
    
    if h_graph == True:
        lefty = linear_fit(x, left_slope[0], left_cept[0])
        righty = linear_fit(x, right_slope[0], right_cept[0])
        plt.plot((lab[0,:,0]/ l_conv))
        plt.plot(x, lefty)
        plt.plot(x, righty)
        
    strip = lab[:,:,0]

    y = np.arange(0, l1.shape[0])
    x = np.arange(0, l1.shape[1])

    left_slope = []
    left_cept = []

    right_slope = []
    right_cept = []

    for n in list(range(len(y))):
        strip = lab[:,n,0]
        
        left = strip[0: round((len(strip)/ 2))]
        right = strip[(round(len(strip)/ 2)) :]
        
        fit = curve_fit(linear_fit, (y[0:round((len(strip)/ 2))]), (left/l_conv))
        left_slope.append(fit[0][0])
        left_cept.append(fit[0][1])
        
        fit = curve_fit(linear_fit, (y[round((len(strip)/ 2)) : ]), (right/l_conv))
        right_slope.append(fit[0][0])
        right_cept.append(fit[0][1])
      
    if v_graph == True:
        lefty = linear_fit(y, left_slope[0], left_cept[0])
        righty = linear_fit(y, right_slope[0], right_cept[0])
        plt.plot((lab[:,0,0]/ l_conv))
        plt.plot(y, lefty)
        plt.plot(y, righty)

    features.append(np.mean(left_slope))
    features.append(np.mean(right_slope))

    return features
    
def background_grabber(path_in, show = False, thresh = 75):
    og = cv2.imread(path_in)
    a = 800
    b = 725
    og = cv2.resize(og, (a,b))
    
    img = cv2.imread(path_in)
    img1 = cv2.imread(path_in, 0)
    
    img = cv2.resize(img, (a,b))
    img_copy = img
    img_copy2 = img
    img1 = cv2.resize(img, (a,b))
    
    img = np.where(img[:,:,2] < 150, 0, img[:,:,1])
    img_copy[:,:,1] = img
    
    # Convert to graycsale
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1= 10, threshold2= thresh)
    # make the line bigger,
    dilate = cv2.dilate(edges, (10,10), iterations = 10)
    
    # find the edges fo the dilated line
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # fill the line in 
    fill = cv2.fillPoly(dilate, cnts, [255,255,255])
    
    # create a masked image with the polyfill
    mask = np.where(fill < 250, 0, fill)
    
    # find all of the connected components (white blobs in your image).
    # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(mask)
    # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information. 
    # here, we're interested only in the size of the blobs, contained in the last column of stats.
    sizes = stats[:, -1]
    # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
    # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below. 
    sizes = sizes[1:]
    nb_blobs -= 1
    
    # minimum size of particles we want to keep (number of pixels).
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
    #min_size = 20000
    
    # output image with only the kept components
    im_result = np.zeros_like(fill)
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] == sizes.max():
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 255
    
    # find the edges fo the dilated line
    cnts = cv2.findContours(im_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # fill the line in 
    fill = cv2.fillPoly(img1, cnts, [255,255,255])
    # create a masked image with the polyfill
    mask = np.where(fill > 250, 0, fill)
    
    if show == True:
        cv2.imshow('og', og)
        cv2.waitKey(0)
        
        cv2.imshow('red_replace_im', img_copy)
        cv2.waitKey(0)
        
        cv2.imshow('edged', edges)
        cv2.waitKey(0)
        
        cv2.imshow('fill', fill)
        cv2.waitKey(0)
        
        cv2.imshow('fill', mask)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()

    return mask



def scale_data (df):
    scaler = StandardScaler()
    scaler.fit(df)
    x_scaled = scaler.transform(df)
    return x_scaled

standard = pd.read_csv(r'PATH\TO\THIS\FILE\background_features.csv')

df = standard.reset_index()
df2 = df[['ls_mean', 'as_mean', 'bs_mean', 'bog', 'rog', 'gog',
          'left_slope_h', 'right_slope_h', 'left_slope_v', 'right_slope_v', 
          ]]

scale  = scale_data(df2)

pca = PCA(n_components = 2)
components = pca.fit_transform(scale)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

pca_df = pd.DataFrame(components)
pca_df.columns = ['PC1', 'PC2']

df['PC1'] = pca_df['PC1']
df['PC2'] = pca_df['PC2']

mean_pc1 = (df.loc[df['lighting'] == 'dark']).mean()
mean_pc2 = (df.loc[df['lighting'] == 'light']).mean()
light_coord = np.array((mean_pc2['PC1'], mean_pc2['PC2']))
dark_coord = np.array((mean_pc1['PC1'], mean_pc1['PC2']))

def weighting(image_in, graph = False, light_coord = light_coord, dark_coord = dark_coord):
    
    out = []
    
    mid_f = image_features(image_in)

    mid_df = pd.DataFrame(mid_f)
    scale_mid = scale_data(mid_df).reshape(1,-1)

    mid_fit = pca.transform(scale_mid)
    mid_fit_df = pd.DataFrame(mid_fit)
    mid_fit_df.columns = ['PC1', 'PC2']
    
    dis_light_dark = np.linalg.norm(light_coord - dark_coord) # DIATANCE FROM LIGHT TO DARK
    dis_mid_light = np.linalg.norm(mid_fit - light_coord) # DISTANCE FROM LIGHT TO INTERMEDIATE
    dis_mid_dark = np.linalg.norm(mid_fit - dark_coord) # DISTANCE FROM DARK TO INTERMEDIATE

    weight_l = 1 - (dis_mid_light / dis_light_dark)
    weight_d = 1 - (dis_mid_dark / dis_light_dark)

    out.append(weight_l)
    out.append(weight_d)

    mid_point = (mid_fit[0][0], np.mean([dark_coord, light_coord]))
    
    dis_light_dark = np.linalg.norm(light_coord - dark_coord) # DIATANCE FROM LIGHT TO DARK
    dis_mid_light = np.linalg.norm(mid_point - light_coord) # DISTANCE FROM LIGHT TO INTERMEDIATE
    dis_mid_dark = np.linalg.norm(mid_point - dark_coord) # DISTANCE FROM DARK TO INTERMEDIATE
    
    weight_l2 = 1 - (dis_mid_light / dis_light_dark)
    weight_d2 = 1 - (dis_mid_dark / dis_light_dark)

    out.append(weight_l2)
    out.append(weight_d2)
    
    out.append(mid_fit_df['PC1'].iloc[0])    
    out.append(mid_fit_df['PC2'].iloc[0])    

    if graph == True:
        plt.scatter(x = mean_pc1['PC1'], y = mean_pc1['PC2'], color = 'black', label = 'dark mean')
        plt.scatter(x = mean_pc2['PC1'], y = mean_pc2['PC2'], color = 'gold', label = 'light mean')
        plt.scatter(data = mid_fit_df, x = 'PC1', y = 'PC2', color = 'red', label = 'original point')        
        plt.scatter(mid_point[0], mid_point[1],  color = 'blue', label = 'baseline corrected midpoint',)
        
        plt.legend(['dark mean', 'light mean', 'original points', 'corrected midpoints'])

    return out

def crop_to_edges(path_in, print_image = False, edge_thresh = 75):
    
    og = cv2.imread(path_in)
    a = int(og.shape[0]/5)
    b = int(og.shape[1]/4)

    og = cv2.resize(og, (a,b))

    
    img = cv2.imread(path_in)
    img1 = cv2.imread(path_in, 0)

    a = int(img.shape[0]/5)
    b = int(img.shape[1]/4)

    img = cv2.resize(img, (a,b))
    img_copy = img
    img_copy2 = img
    img1 = cv2.resize(img, (a,b))

    img = np.where(img[:,:,2] < 150, 0, img[:,:,1])
    img_copy[:,:,1] = img
    
    # Convert to graycsale
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1= 10, threshold2= edge_thresh)
    # make the line bigger,
    dilate = cv2.dilate(edges, (10,10), iterations = 10)
    
    # find the edges fo the dilated line
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # fill the line in 
    fill = cv2.fillPoly(dilate, cnts, [255,255,255])
    
    # create a masked image with the polyfill
    mask = np.where(fill < 250, 0, fill)
    
    # find all of the connected components (white blobs in your image).
    # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(mask)
    # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information. 
    # here, we're interested only in the size of the blobs, contained in the last column of stats.
    sizes = stats[:, -1]
    # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
    # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below. 
    sizes = sizes[1:]
    #print(sizes)
    nb_blobs -= 1

    # minimum size of particles we want to keep (number of pixels).
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
    #min_size = 20000

    # output image with only the kept components
    im_result = np.zeros_like(fill)
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] == sizes.max():
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 255

    # find the edges fo the dilated line
    cnts = cv2.findContours(im_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # fill the line in 
    fill = cv2.fillPoly(img1, cnts, [255,255,255])
    # create a masked image with the polyfill
    mask = np.where(fill < 250, 0, fill)
    #crop to the filled area from the mask
    crop = np.where(mask < 250, 0, og)    

    if print_image == True:
        cv2.imshow('og', og)
        cv2.waitKey(0)
        
        cv2.imshow('red_replace_im', img_copy)
        cv2.waitKey(0)
    
        cv2.imshow('edged', edges)
        cv2.waitKey(0)
        
        cv2.imshow('fill', fill)
        cv2.waitKey(0)
    
        cv2.imshow('cropped', crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return crop

def get_lab(cropped_in):
        
    a = 800
    b = 725
    img = cv2.resize(cropped_in, (a,b))

    lab2 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # convert the BGR image to LAB color space
        
    out = []
    
    l2 = lab2[:, :, 0]
    a2 = lab2[:, :, 1]
    b2 = lab2[:, :, 2]

    l3 = np.where(l2 ==0 , np.nan, l2)
    out.append(np.nanmean(l3) / (255/100)) # standard conversion factor for 8-bit images
    out.append(np.nanmax(l3) / (255/100)) # standard conversion factor for 8-bit images
    out.append(np.nanmin(l3) / (255/100)) # standard conversion factor for 8-bit images
    out.append(np.nanstd(l3) / (255/100))
    
    a3 = np.where(a2 == 0, np.nan, a2)
    out.append(np.nanmean(a3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanmax(a3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanmin(a3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanstd(a3) - 128)

    b3 = np.where(b2 == 0, np.nan, b2)
    out.append(np.nanmean(b3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanmax(b3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanmin(b3) - 128) # standard conversion factor for 8-bit images
    out.append(np.nanstd(b3) - 128) # standard conversion factor for 8-bit images

    B1 = cropped_in[:,:,0]
    G1 = cropped_in[:,:,1]
    R1 = cropped_in[:,:,2]
    
    B2 = np.where(B1 ==0 , np.nan, B1)
    G2 = np.where(G1 ==0 , np.nan, G1)
    R2 = np.where(R1 ==0 , np.nan, R1)

    out.append(np.nanmean(B2))
    out.append(np.nanmean(G2))
    out.append(np.nanmean(R2))
    
    mean_slope =[]
    mean_slope2 =[]
    mean_above =[]
    mean_above2 =[]
    
    for i in list(range(0,10)):
        s = rand_slope(lab2)
        #[mean_slope, mean_above, mean_slope2, mean_above2]
        mean_slope.append(s[0])
        mean_slope2.append(s[2])
        mean_above.append(s[1])
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
