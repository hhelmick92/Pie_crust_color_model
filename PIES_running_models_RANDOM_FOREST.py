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


# [READ IN THE MODESLS THAT WERE SELECTED IN GRID SEARCHING]

name = r'L:\Kokini Lab\Kara Benbow\project_main\code\selected_models\all_out_scaler.pkl'
with open(name, 'rb') as file:
    all_scaler = pickle.load(file)

name = r'L:\Kokini Lab\Kara Benbow\project_main\code\selected_models\light_scaler.pkl'
with open(name, 'rb') as file:
    light_scaler = pickle.load(file)

name = r'L:\Kokini Lab\Kara Benbow\project_main\code\selected_models\dark_scaler.pkl'
with open(name, 'rb') as file:
    dark_scaler = pickle.load(file)

name = r'L:\Kokini Lab\Kara Benbow\project_main\code\selected_models\all_out.pkl'
with open(name, 'rb') as file:
    all_ann = pickle.load(file)

name = r'L:\Kokini Lab\Kara Benbow\project_main\code\selected_models\light_out.pkl'
with open(name, 'rb') as file:
    light_ann = pickle.load(file)

name = r'L:\Kokini Lab\Kara Benbow\project_main\code\selected_models\dark_out.pkl'
with open(name, 'rb') as file:
    dark_ann = pickle.load(file)
        
# In[]

def get_files(path_in):
    
    path = path_in
    files = glob.glob(path + '\*.png')
    
    names = []
    for f in files:
        t1 = f.split('\\')
        names.append(t1[-1])
        
    return [files, names]

t15_back = get_files(r'L:\Kokini Lab\Kara Benbow\project_main_v2\crops\t15\backs')
t15_pie = get_files(r'L:\Kokini Lab\Kara Benbow\project_main_v2\crops\t15\pies')

t30_back = get_files(r'L:\Kokini Lab\Kara Benbow\project_main_v2\crops\t30\backs')
t30_pie = get_files(r'L:\Kokini Lab\Kara Benbow\project_main_v2\crops\t30\pies')

all_back = t15_back[0] + t30_back[0]
all_pie = t15_pie[0] + t30_pie[0]

all_pie_names = t15_pie[1] + t30_pie[1]

# In[]

def back_weights(files_in):
    
    back_weights = []
    
    for path in files_in:
        back = cv2.imread(path)
        weights = weighting(back, False)
        back_weights.append(weights)    

    return back_weights

start_time = count_time.time()

t15_back_weights = back_weights(t15_back[0])
t30_back_weights = back_weights(t30_back[0])

all_back_weights = t15_back_weights + t30_back_weights

elap = count_time.time() - start_time

print('background processing time s/img')
print(elap / len(all_pie))

'''
background processing time s/img
1.7902176521326367
'''

with open(r'L:\Kokini Lab\Kara Benbow\project_main_v2\crops\back_data_list.txt', 'w') as f:
    f.write(str(all_back_weights))

# In[]

def get_pie_data(files_in):
    
    pie_data = []

    for path in files_in:
        cropped = cv2.imread(path)
        cropped_data = get_lab(cropped)
        pie_data.append(cropped_data)

    return pie_data

start_time = count_time.time()

t15_pie_weights = get_pie_data(t15_pie[0])
t30_pie_weights = get_pie_data(t30_pie[0])

all_pie_weights = t15_pie_weights + t30_pie_weights
elap = count_time.time() - start_time

print('pie processing time s/img')
print(elap / len(all_pie))
'''
pie processing time s/img
0.23423869986283152
'''

# In[]

light_out = []
dark_out = []
all_out = []
weighted_out = []
weighted2_out = []

for data, weights in zip(all_pie_weights, all_back_weights):
    
    im_weights = pd.DataFrame(weights)
    im_weights.index = ['light_weight1', 'dark_weight1', 'light_weight2', 'dark_weight2', 'PC1', 'PC2']
    
    cropped_data = data
    
    idx = ['Ls_mean', 'Ls_max', 'Ls_min', 'Ls_std', 
     'As_mean', 'As_max', 'As_min', 'As_std',
     'Bs_mean', 'Bs_max', 'Bs_min', 'Bs_std',
     'Bog_mean', 'Gog_mean', 'Rog_mean',
     'mean_slope_h', 'mean_slope_v', 'above_h', 'mean_above_v']
    
    pie = pd.DataFrame(cropped_data)
    pie.index = idx
    pie = pie.transpose()
    
    pie['abs_slope_h'] = pie['mean_slope_h'].abs()
    pie['abs_slope_v'] = pie['mean_slope_v'].abs()
    
    pie['count_above_std1_h'] = pie['above_h']
    pie['count_above_std1_v'] = pie['mean_above_v']

    '''
    pie['count_above_std1_h'] = pie['above_h'] = 0
    pie['count_above_std1_v'] = pie['mean_above_v'] = 0
    pie['abs_slope_h'] = pie['mean_slope_h'] = 0
    pie['abs_slope_v'] = pie['mean_slope_v'] = 0
    '''
    
    med_in_light = pie[['Ls_mean', 'As_mean', 'Bs_mean', 'abs_slope_h']]
    med_in_dark = pie[['Ls_mean', 'As_mean', 'Bs_mean', 'count_above_std1_v']]
    med_in_all = pie[['Ls_mean', 'As_mean', 'Bs_mean', 'abs_slope_h']]
        
    light_scale = light_scaler.transform(med_in_light)
    dark_scale = dark_scaler.transform(med_in_dark)
    all_scale = all_scaler.transform(med_in_all)
    
    light_pred = light_ann.predict(light_scale)
    dark_pred = dark_ann.predict(dark_scale)
    all_pred = all_ann.predict(all_scale)
    
    light_preds = pd.DataFrame(light_pred)
    light_weighted = light_preds * im_weights.loc['light_weight1'][0]
    
    dark_preds = pd.DataFrame(dark_pred)
    dark_weighted = dark_preds * im_weights.loc['dark_weight1'][0]
    
    weighted_pred = light_weighted + dark_weighted
    
    light_out.append(light_pred[0])
    dark_out.append(dark_pred[0])
    all_out.append(all_pred[0])
    weighted_out.append(weighted_pred.iloc[0])
    
    if im_weights.loc['PC2'][0] < 0:
        weighted2 = [light_pred[0][0], light_pred[0][1], dark_pred[0][2]]
        weighted2_out.append(weighted2)

# In[]

light = pd.DataFrame(light_out)
dark = pd.DataFrame(dark_out)
all_df = pd.DataFrame(all_out)
weighted = pd.DataFrame(weighted_out)
weighted2 = pd.DataFrame(weighted2_out)

weighted.columns = ['L_pred', 'A_pred', 'B_pred']
light.columns = ['L_pred', 'A_pred', 'B_pred']
dark.columns = ['L_pred', 'A_pred', 'B_pred']
all_df.columns = ['L_pred', 'A_pred', 'B_pred']
weighted2.columns = ['L_pred', 'A_pred', 'B_pred']

weighted.index = all_pie_names
light.index = all_pie_names
dark.index = all_pie_names
all_df.index = all_pie_names
weighted2.index = all_pie_names

'''
weighted.index = t15_pie[1]
light.index = t15_pie[1]
dark.index = t15_pie[1]
all_df.index = t15_pie[1]
'''
'''
weighted.index = t30_pie[1]
light.index = t30_pie[1]
dark.index = t30_pie[1]
all_df.index = t30_pie[1]
'''

weighted = light

pea = []
weight = []
time = []
for i in weighted.index:
    t1 = i.split('_')
    pea.append(t1[0].replace('pea', ''))
    weight.append(t1[3].replace('weight', ''))
    time.append(t1[2].replace('time', ''))

weighted['pea'] = pea
weighted['weight'] = weight
weighted['time'] = time
weighted['pea_weight_time'] = weighted['pea'] + '_' + weighted['weight'] + '_' + weighted['time']
weighted['pea_weight_time'] = weighted['pea_weight_time'].replace({'egg_1_15': 'EGG_1_15', 'egg_1_30' : 'EGG_1_30', 
                                                         '0_0_15':'NONE_0_15', '0_0_30':'NONE_0_30'
                                                         })
res2g = weighted.groupby('pea_weight_time')
res2m = res2g.mean()
res2m['merge'] = res2m.index

color = pd.read_csv(r'L:\Kokini Lab\Kara Benbow\project_main_v2\tables\hunter_data.csv')

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
color['pea_weight_time'] = color['pea'] + '_' + color['weight'] + '_' + color['str_time']

color = color.sort_values(by = 'ID')
color['pea_weight_time'].unique()


gloss = pd.read_csv(r'L:\Kokini Lab\Kara Benbow\Characterization\gloss.csv')
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
    pea.append(t1[0].replace('pea', ''))
    ph.append(t1[1].strip('ph'))
    time.append(t1[2].strip('time'))
    weight.append(t1[3].strip('weight'))

cg = pd.concat([color, gmelt], axis = 1)

g = cg.groupby('pea_weight_time')
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

merge = ms.merge(res2m, on = 'merge')

merge['L_error_ann'] = np.sqrt((merge['L*'] - merge['L_pred'])**2)
merge['A_error_ann'] = np.sqrt((merge['a*'] - merge['A_pred'])**2)
merge['B_error_ann'] = np.sqrt((merge['b*'] - merge['B_pred'])**2)
merge['Total_error_ann'] = np.sqrt((merge['L*'] - merge['L_pred'])**2 + (merge['a*'] - merge['A_pred'])**2 + (merge['b*'] - merge['B_pred'])**2)

merge['L_error_ann'].describe()
merge['A_error_ann'].describe()
merge['B_error_ann'].describe()
merge['Total_error_ann'].describe()

print(merge['L_error_ann'].describe())
print(merge['A_error_ann'].describe())
print(merge['B_error_ann'].describe())
print(merge['Total_error_ann'].describe())

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

#merge.to_csv(r'L:\Kokini Lab\Kara Benbow\project_main_v2\used_process_outputs\weighted2_pie_results.csv')
#merge.to_csv(r'L:\Kokini Lab\Kara Benbow\project_main_v2\used_process_outputs\all_df_pie_results.csv')

# In[]

model = 'insert_model_name'
plt.figure(0)
sns.lmplot(data = merge, x = 'L*', y = 'L_pred', hue = 'time')
plt.title(model)
plt.figure(1)
sns.lmplot(data = merge, x = 'a*', y = 'A_pred', hue = 'time')
plt.title(model)
plt.figure(2)
sns.lmplot(data = merge, x = 'b*', y = 'B_pred', hue = 'time')
plt.title(model)

# In[]

#merge = merge.loc[merge['B_pred'] > 15]

plt.style.use('seaborn')

L_lin = linregress(merge['L_pred'], merge['L*'])
A_lin = linregress(merge['A_pred'], merge['a*'])
B_lin = linregress(merge['B_pred'], merge['b*'])

fig, ax = plt.subplots(2,2)
fig.tight_layout(h_pad = 2)

ax[0,0].scatter(merge['L_pred'], merge['L*'])
ax[0,0].plot(merge['L_pred'], L_lin.slope * merge['L_pred'] + L_lin.intercept, color = 'indianred', label = str(round(L_lin.rvalue **2, 3)))
ax[0,0].set_title('L values')
ax[0,0].legend()

ax[0,1].scatter(merge['A_pred'], merge['a*'])
ax[0,1].plot(merge['A_pred'], A_lin.slope * merge['A_pred'] + A_lin.intercept, color = 'indianred', label = str(round(A_lin.rvalue **2, 3)))
ax[0,1].set_title('A values')
ax[0,1].legend()

ax[1,0].scatter(merge['B_pred'], merge['b*'])
ax[1,0].plot(merge['B_pred'], B_lin.slope * merge['B_pred'] + B_lin.intercept, color = 'indianred', label = str(round(B_lin.rvalue **2, 3)))
ax[1,0].set_title('B values')
ax[1,0].legend()

# In[]

weighted2.to_csv(r'L:\Kokini Lab\Kara Benbow\project_main_v2\used_process_outputs\weighted2.csv')
all_df.to_csv(r'L:\Kokini Lab\Kara Benbow\project_main_v2\used_process_outputs\all_df.csv')



































