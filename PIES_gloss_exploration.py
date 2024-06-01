# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:46:21 2023

@author: hhelmick
"""

import glob
import pickle
import time as count_time

import numpy as np
import pandas as pd

import cv2

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

from PIES_functions import weighting
from PIES_functions import get_lab

# In[]

def get_files(path_in):
    
    path = path_in
    files = glob.glob(path + '\*.png')
    
    names = []
    for f in files:
        t1 = f.split('\\')
        names.append(t1[-1])
        
    return [files, names]

t15_back = get_files(r'PATH\TO\BACKGROUNDS')
t15_pie = get_files(r'PATH\TO\CROPPED_IMAGES')

t30_back = get_files(r'PATH\TO\BACKGROUNDS')
t30_pie = get_files(r'PATH\TO\CROPPED_IMAGES')

all_back = t15_back[0] + t30_back[0]
all_pie = t15_pie[0] + t30_pie[0]

all_pie_names = t15_pie[1] + t30_pie[1]

# In[]

def get_pie_data(files_in):
    
    pie_data = []

    for path in files_in:
        cropped = cv2.imread(path)
        cropped_data = get_lab(cropped)
        pie_data.append(cropped_data)
        
    glare_sums = []
    s_out = []

    a = 800
    b = 725

    for fn in files_in:
        im = cv2.imread(fn)
        im = cv2.resize(im, (a,b))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)[1]
        glare_area = np.sum(mask)
        glare_sums.append(glare_area)

        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        
        # I know that this is pulling out h, not s, i'm just lazy and don't want to change other things
        s = hsv[:, :, 0]
        
        s1 = np.where(s < 235 , np.nan, s)
        s_out.append(np.nansum(s1))

    return pie_data, glare_sums, s_out

start_time = count_time.time()

t15_pie_weights = get_pie_data(t15_pie[0])
t30_pie_weights = get_pie_data(t30_pie[0])


all_pie_weights = t15_pie_weights[0] + t30_pie_weights[0]
all_pie_glare = t15_pie_weights[1] + t30_pie_weights[1]
all_pie_s = t15_pie_weights[2] + t30_pie_weights[2]

elap = count_time.time() - start_time

print('pie processing time s/img')
print(elap / len(all_pie))
'''
pie processing time s/img
0.23423869986283152
'''
# In[]

glare = pd.DataFrame(all_pie_glare)
glare.index = all_pie_names
#glare.index = t30_pie[1]

s_sum = pd.DataFrame(all_pie_s)
s_sum.index = all_pie_names
#glare.index = t30_pie[1]


pie = pd.DataFrame(all_pie_weights)
pie.index = all_pie_names
#pie.index = t30_pie[1]


cols = ['Ls_mean', 'Ls_max', 'Ls_min', 'Ls_std', 
     'As_mean', 'As_max', 'As_min', 'As_std',
     'Bs_mean', 'Bs_max', 'Bs_min', 'Bs_std',
     'Bog_mean', 'Gog_mean', 'Rog_mean',
     'mean_slope_h', 'mean_slope_v', 'above_h', 'mean_above_v']

pie.columns = cols

pea = []
ph = []
time = []
weight = []
for l in pie.index:
    t1 = l.split('_')
    pea.append(t1[0].upper().replace('PEA', ''))
    ph.append(t1[1].replace('ph', ''))
    time.append(t1[2].replace('time', ''))
    weight.append(t1[3].replace('weight', ''))

pie['pea'] = pea
pie['ph'] = ph
pie['time'] = time
pie['weight'] = weight

pie['merge_col'] = pie['pea'] + '_' + pie['weight'] + '_' + pie['time'] + '_' + pie['ph']
pie['merge_col'] = pie['merge_col'].apply(lambda x: str(x).replace('0_0_30_0', 'NONE_0_30_0'))
pie['merge_col'] = pie['merge_col'].apply(lambda x: str(x).replace('0_0_15_0', 'NONE_0_15_0'))
pie['merge_col'].unique()

pie['pea_weight_time_ph'] = pie['merge_col']

pie['glare'] = glare[0]
pie['spec'] = s_sum[0]

color = pd.read_csv(r'PATH\TO\THIS\FILE\hunter_data.csv')

pea = []
gly = []
weight = []
rep = []
for i in color['ID'].to_list():
    t1 = i.split('_')
    pea.append(t1[0].strip('pea'))
    gly.append(t1[1].strip('gly'))
    weight.append(t1[2].strip('g'))
    rep.append(t1[3].strip('rep'))

color['pea'] = pea
color['gly'] = gly
color['weight'] = weight
color['rep'] = rep
color['str_time'] = color['bake_time'].apply(lambda x: str(x))
color['str_ph'] = color['ph'].apply(lambda x: str(x))

color['pea_weight_time_ph'] = color['pea'] + '_' + color['weight'] + '_' + color['str_time'] + '_' + color['str_ph']

color = color.sort_values(by = 'ID')

gloss = pd.read_csv(r'PATH\TO\THIS\FILE\gloss.csv')
gloss = gloss.drop(['Unnamed: 13', 'Unnamed: 14'], axis = 1)
gloss = gloss[0:16]
gloss = gloss.drop(['Unnamed: 28', 'Unnamed: 39'], axis = 1)

gmean = gloss.mean()
gstd = gloss.std()

gstat = pd.DataFrame()
gstat['mean'] = gmean
gstat['std'] = gstd

gmelt = gloss.melt()

pea = []
ph = []
time = []
weight = []
for i in gmelt['variable'].to_list():
    t1 = i.split('_')
    pea.append(t1[0].upper().replace('PEA', ''))
    ph.append(t1[1].strip('ph'))
    time.append(t1[2].strip('time'))
    weight.append(t1[3].strip('weight'))

gmelt['pea'] = pea
gmelt['ph'] = ph
gmelt['time'] = time
gmelt['weight'] = weight

gmelt['pea_weight_time_ph'] = gmelt['pea'] + '_' + gmelt['weight'] + '_' + gmelt['time'] + '_' + gmelt['ph']

gmelt['pea_weight_time_ph'] = gmelt['pea_weight_time_ph'].apply(lambda x: str(x).replace('0_0.1_30_0', 'NONE_0_30_0'))
gmelt['pea_weight_time_ph'] = gmelt['pea_weight_time_ph'].apply(lambda x: str(x).replace('0_0.1_15_0', 'NONE_0_15_0'))
gmelt['pea_weight_time_ph'] = gmelt['pea_weight_time_ph'].apply(lambda x: str(x).replace('0_0_30_0', 'NONE_0_30_0'))
gmelt['pea_weight_time_ph'] = gmelt['pea_weight_time_ph'].apply(lambda x: str(x).replace('0_0_15_0', 'NONE_0_15_0'))


cg = color.merge(gmelt, on = 'pea_weight_time_ph')

g = cg.groupby('pea_weight_time_ph')
m = g.mean()
s = g.std()

new_c = []
for c in s.columns:
    new_c.append(c + '_' + 'std')

s.columns = new_c

ms = pd.concat([m, s], axis = 1)

ms['merge'] = m.index

s['L*_std'].describe()
s['a*_std'].describe()
s['b*_std'].describe()


merge = ms.merge(pie, on = 'pea_weight_time_ph')

pea = []
weight = []
time = []
for i in merge['merge'].to_list():
    t1 = i.split('_')
    pea.append(t1[0])
    weight.append(t1[1])
    time.append(t1[2])

merge['pea'] = pea
merge['weight'] = weight
merge['time'] = time

pea = []
weight = []
for i in merge['pea_weight_time_ph'].tolist():
    t1 = i.split('_')
    pea.append(t1[0])
    weight.append(t1[1])
    
merge['pea'] = pea
merge['weight'] = weight

# In[]

plt.figure(0)
sns.lmplot(data = merge, x = 'value', y = 'spec', hue = 'bake_time')

plt.figure(1)
sns.scatterplot(data = merge, x = 'value_std', y = 'spec')

# In[]

data_15 = merge.loc[merge['bake_time'] == 15]
data_15 = data_15.loc[data_15['spec'] < 60000]
sns.lmplot(data = data_15, x = 'value', y = 'spec', hue = 'bake_time')


# In[]

merge.columns

sns.scatterplot(data = merge, x = 'value', y = 'Ls_std')
sns.scatterplot(data = merge, x = 'value', y = 'As_std')
sns.scatterplot(data = merge, x = 'value', y = 'Bs_std')
sns.scatterplot(data = merge, x = 'value', y = 'mean_slope_h')
sns.scatterplot(data = merge, x = 'value', y = 'mean_slope_v')
sns.scatterplot(data = merge, x = 'value', y = 'above_h')
sns.scatterplot(data = merge, x = 'value', y = 'mean_above_v')

sns.scatterplot(data = merge, x = 'value_std', y = 'Ls_std')
sns.scatterplot(data = merge, x = 'value_std', y = 'As_std')
sns.scatterplot(data = merge, x = 'value_std', y = 'Bs_std')
sns.scatterplot(data = merge, x = 'value_std', y = 'mean_slope_h')
sns.scatterplot(data = merge, x = 'value_std', y = 'mean_slope_v')
sns.scatterplot(data = merge, x = 'value_std', y = 'above_h')
sns.scatterplot(data = merge, x = 'value_std', y = 'mean_above_v')


sns.scatterplot(data = merge, x = 'value', y = 'Ls_mean')
sns.scatterplot(data = merge, x = 'value', y = 'As_mean')
sns.scatterplot(data = merge, x = 'value', y = 'Bs_mean')
sns.scatterplot(data = merge, x = 'value', y = 'bake_time')
sns.scatterplot(data = merge, x = 'value', y = 'pea')

sns.boxplot(data = merge, x = 'pea', y = 'value', hue = 'bake_time')
sns.boxplot(data = merge, x = 'pea', y = 'Ls_mean', hue = 'bake_time')
sns.boxplot(data = merge, x = 'pea', y = 'L*', hue = 'bake_time')

sns.scatterplot(data = merge, x = 'value', y = 'glare', hue = 'bake_time')
sns.scatterplot(data = merge, x = 'value_std', y = 'glare')


























