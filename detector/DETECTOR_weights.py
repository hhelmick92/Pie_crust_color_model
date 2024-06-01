# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 07:51:11 2023

@author: hhelmick
"""

import glob

import cv2

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from DETECTOR_functions import image_features
from DETECTOR_functions import scale_data

import matplotlib.pyplot as plt
import seaborn as sns

# In[]

cols = ['ls_mean', 'as_mean', 'bs_mean', 'bog', 'rog', 'gog', 'left_slope_h', 'right_slope_h', 'left_slope_v', 'right_slope_v']

light_path = r'PATH\TO\LIGHT\IMAGES'

light_files = glob.glob(light_path + '\*.png')

light_f = []
for fn in light_files:
    im = cv2.imread(fn)
    t1 = image_features(im)
    light_f.append(t1)

light_df = pd.DataFrame(light_f)
light_df.columns = cols
light_df['lighting'] = 'light'

dark_path = r'PATH\TO\DARK\IMAGES'

dark_files = glob.glob(dark_path + '\*.png')

dark_f = []
for fn in dark_files:
    im = cv2.imread(fn)
    t1 = image_features(im)
    dark_f.append(t1)

dark_df = pd.DataFrame(dark_f)
dark_df.columns = cols
dark_df['lighting'] = 'dark'

# features fom the feature extractor
df = pd.concat([light_df, dark_df])

#df.to_csv(r'L:\Kokini Lab\Kara Benbow\project_main\code\detector\background_features.csv')

# In[]
df = pd.read_csv(r'PATH\TO\THIS\FILE\background_features.csv')

df = df.reset_index()
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

plt.style.use('seaborn')
sns.scatterplot(data = df, x = 'PC1', y = 'PC2', hue = 'lighting')
#sns.scatterplot(data = df, x = 'PC1', y = 'PC2', hue = 'bs_mean')
plt.scatter(x = mean_pc1['PC1'], y = mean_pc1['PC2'], color = 'black', label = 'average light', marker = 's', s = 150)
plt.scatter(x = mean_pc2['PC1'], y = mean_pc2['PC2'], color = 'goldenrod', label = 'average dark', marker = 'v', s = 150)
plt.legend(bbox_to_anchor=(1,1))

'''
import pickle
name = r'PATH\TO\THIS\FILE\pca_detector.pkl'
with open(name, 'wb') as file:
    pickle.dump(pca, file)
'''

light_coord = np.array((mean_pc2['PC1'], mean_pc2['PC2']))
dark_coord = np.array((mean_pc1['PC1'], mean_pc1['PC2']))

def weighting(path_in, graph = False, light_coord = light_coord, dark_coord = dark_coord):
    
    out = []
    
    mid = cv2.imread(path_in)
    mid_f = image_features(mid)

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

    if graph == True:
        sns.set(style="darkgrid", font_scale=1.5)
        plt.scatter(x = mean_pc1['PC1'], y = mean_pc1['PC2'], color = 'goldenrod', label = 'average light', marker = 's', s = 150)
        plt.scatter(x = mean_pc2['PC1'], y = mean_pc2['PC2'], color = 'black', label = 'average dark', marker = 'v', s = 150)
        plt.scatter(data = mid_fit_df, x = 'PC1', y = 'PC2', color = 'red', label = 'original point')        
        plt.scatter(mid_point[0], mid_point[1],  color = 'blue', label = 'baseline corrected midpoint',)
        
        plt.legend(['dark mean', 'light mean', 'original point', 'corrected midpoint'], loc = 'lower right')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(bbox_to_anchor = (1,1))

    return out

# In[]

mediate_back = r'PATH\TO\MEDIATE\IMAGES'

mediate_files = glob.glob(mediate_back +'\*.png')

mediate = []
for fn in mediate_files[0:1]:
    r = weighting(fn, True)
    mediate.append(r)

med1 = pd.DataFrame(mediate)
med1.columns = ['light_weight1', 'dark_weight1', 'light_weight2', 'dark_weight2']

mean = med1.mean()

