# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 16:44:41 2023

@author: hhelmick
"""

# import the needed libraries and functions
import cv2
import time
import glob
import pandas as pd

from WRANGLING_functions import get_lab

# In[]

###################################################
# WRITE THE PATH, BRAND, LIGHTING CONDITIONS HERE #
###################################################
brand = 'ppg'
lighting = 'light'
path_in = r'L:\Kokini Lab\Kara Benbow\project_main\original_datasets\high_light\ppg_cropped'
path_out = r'L:\Kokini Lab\Kara Benbow\project_main_v2\required_code'
file_name = 'ppg_light_averages'
###################################################
###################################################

# In[]

# start the timer
time_1 = time.time()

# glob the files that we are going to read
files = glob.glob(path_in + '\*.png')

# testing line - can cut this down to make sure the loops run correctly without doing 1000+ images every time
#files = files[0:4]

# get the names of the files. These will be used in the final dataframe output
names = []
for f in files:
    t1 = f.split('\\')
    t2 = t1[-1].replace('.png', '')
    names.append(t2)

# This for loop splits the images into 5ths and stores that in a big list of images. This list is never saved.
out = []
for f in files:
    im = cv2.imread(f)
    a = 800
    b = 725
    
    M = im.shape[0]//6
    N = im.shape[1]//1
    
    tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]

    out.append(tiles)

# this for loop runs the my lab converter and feature extraction function from my_func
# it will five you the average L*, A*, B*, B, G, R, slopes of left to right / top to bottom and points above the standard deviation (each average of 100 random strips)
# this happens for every file in the data. 
tile_names = []
l1 = []
for im, name in zip(out, names):
    t1 = name
    for n, t in enumerate(im[:-1]):
        convert = get_lab(t)
        l1.append(convert)
        t1 += '_tile' + str(n)
        tile_names.append(t1)
        t1 = name
        
# generate a more user friendly data output
df = pd.DataFrame(l1).transpose()
df.index = ['Ls_mean', 'Ls_max', 'Ls_min', 'Ls_std', 
            'As_mean', 'As_max', 'As_min', 'As_std',
            'Bs_mean', 'Bs_max', 'Bs_min', 'Bs_std',
            'Bog_mean', 'Bog_max', 'Bog_min', 'Bog_std',
            'Rog_mean', 'Rog_max', 'Rog_min', 'Rog_std',
            'Gog_mean', 'Gog_max', 'Gog_min', 'Gog_std',
            'random_slope_h','count_above_std1_h','random_slope1_v', 'count_above_std1_v',
            ]

# some data clean up. Need to split columns to merge with data that comes from the hunter meter
df.columns = tile_names
df = df.transpose()
df['splitter'] = df.index

new = df['splitter'].str.split('_', expand = True)
new['name'] = new[0] + '_' + new[1]

df['name'] = new['name']
df['side'] = new[2]
df['tile'] = new[3]
df['brand'] = brand
df['lighting'] = lighting

df['merger'] = df['name'] + df['tile'].str.replace('tile','')

# In[]

# read in the hunte data that is going to be used for merging, clean up a column for merging
df2 = pd.read_csv(r'L:\Kokini Lab\Kara Benbow\project_main\original_datasets\hunter\hunter_data.csv')
new = df2['ID'].str.split('_', expand = True)
new['name'] = new[0] + '_' + new[1]

df2['name'] = new['name']
df2['brand'] = new[2]
df2['rep'] = new[3]
df2['merger'] = df2['name'] + df2['rep'].str.replace('rep','')

# merge the files
merge = pd.merge(df2, df, on = 'merger')

# write out the file to the path 
merge.to_csv(path_out + '\\' + file_name +'.csv')

# In[]

# Check for entries that are misspelled etc. 
test = df.merge(df2, on = 'merger', how = 'left', indicator = True)
test2 = test.loc[test['_merge'] == 'left_only']
unq = test2['merger'].unique()

missing = []
for n in unq:
    if '0' in str(n):
        pass
    else:
        missing.append(n)
        
print('Hey! These entries are in your image data but not the hunter data!')
print(missing)
print('\n')

# In[]
time_2 = time.time() - time_1
time_2 = str(round(time_2,3))
print('It took {} seconds to run {} images'.format(time_2, str(len(df))))









