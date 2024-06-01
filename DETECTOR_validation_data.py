# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 10:25:10 2023

@author: hhelmick
"""

import glob
import pickle

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error 

import cv2
import numpy as np
#from skimage import io, color
import pandas as pd
import random

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

# In[]

name = r'L:\Kokini Lab\Kara Benbow\project_main\code2\selected_models\all_scaler.pkl'
with open(name, 'rb') as file:
    all_scaler = pickle.load(file)

name = r'L:\Kokini Lab\Kara Benbow\project_main\code2\selected_models\light_scaler.pkl'
with open(name, 'rb') as file:
    light_scaler = pickle.load(file)

name = r'L:\Kokini Lab\Kara Benbow\project_main\code2\selected_models\dark_scaler.pkl'
with open(name, 'rb') as file:
    dark_scaler = pickle.load(file)

name = r'L:\Kokini Lab\Kara Benbow\project_main\code2\selected_models\ann_all_data.pkl'
with open(name, 'rb') as file:
    all_ann = pickle.load(file)

name = r'L:\Kokini Lab\Kara Benbow\project_main\code2\selected_models\ann_light_data.pkl'
with open(name, 'rb') as file:
    light_ann = pickle.load(file)

name = r'L:\Kokini Lab\Kara Benbow\project_main\code2\selected_models\ann_dark_data.pkl'
with open(name, 'rb') as file:
    dark_ann = pickle.load(file)

# In[RUN THE MODEL ON THE LIGHT ONLY VALIDATION DATA, BEHR COLOR SWATHS]

light_val = pd.read_csv(r'L:\Kokini Lab\Kara Benbow\project_main\process_outputs\light_validation.csv')
light_val['abs_slope_h'] = light_val['random_slope_h'].abs()
light_val_in = light_val[['Ls_mean', 'As_mean', 'Bs_mean', 'abs_slope_h', 'count_above_std1_h']]

scale = light_scaler.transform(light_val_in)

pred = light_ann.predict(scale)

pred_df = pd.DataFrame(pred)
pred_df.columns = ['L_pred', 'A_pred', 'B_pred']

compare = pred_df
compare['L*'] = light_val['L*']
compare['a*'] = light_val['a*']
compare['b*'] = light_val['b*']
compare['names'] = light_val['name_y']
compare['Ls_mean'] = light_val['Ls_mean']
compare['As_mean'] = light_val['As_mean']
compare['Bs_mean'] = light_val['Bs_mean']

compare['L_error'] = np.abs((compare['L*'] - compare['L_pred'])) / compare['L*'].abs()
compare['A_error'] = np.abs((compare['a*'] - compare['A_pred'])) / compare['a*'].abs()
compare['B_error'] = np.abs((compare['b*'] - compare['B_pred'])) / compare['b*'].abs()
compare['total_error'] = (compare['L_error'] + compare['A_error'] + compare['B_error']) / 3

group = compare.groupby('names')
mean = group.mean()

mean[['L_pred', 'L*']]
mean[['A_pred', 'a*']]
mean[['B_pred', 'b*']]

#sns.scatterplot(data = mean, x = 'L*', y = 'L_pred', hue = 'names')
sns.scatterplot(data = mean, x = 'L*', y = 'L_pred')
line = linregress(mean['L*'],mean['L_pred'])
sns.lineplot(x = mean['L*'], y = mean['L*'] * line.slope + line.intercept, label = 'r2: ' + str(round(line.rvalue **2, 3)))
#plt.legend(bbox_to_anchor = (1,1))
plt.title('Bright Model - Validation Data')

l_rmse = mean_squared_error(mean['L*'], mean['L_pred'], squared = False)
a_rmse = mean_squared_error(mean['a*'], mean['A_pred'], squared = False)
b_rmse = mean_squared_error(mean['b*'], mean['B_pred'], squared = False)

mean_cut = mean.loc[mean['L*'] > 60]
mean_cut[['L_pred', 'L*']]

l_rmse = mean_squared_error(mean_cut['L*'], mean_cut['L_pred'], squared = False)
compare['total_error'].mean()

print('VALIDATION RMSE VALUES')
print(l_rmse)
print(a_rmse)
print(b_rmse)


# In[RUN THE DATA ON THE DARK VALIDATION BEHR COLORS]

dark_val = pd.read_csv(r'L:\Kokini Lab\Kara Benbow\project_main\process_outputs\dark_validation.csv')
dark_val['abs_slope_h'] = dark_val['random_slope_h'].abs()
dark_val['abs_slope_v'] = dark_val['random_slope1_v'].abs()

dark_val_in = dark_val[['Bog_mean', 'Gog_mean', 'Rog_mean', 'abs_slope_h', 'abs_slope_v']]

scale = dark_scaler.transform(dark_val_in)

pred = dark_ann.predict(scale)

pred_df = pd.DataFrame(pred)
pred_df.columns = ['L_pred', 'A_pred', 'B_pred']

compare = pred_df
compare['L*'] = dark_val['L*']
compare['a*'] = dark_val['a*']
compare['b*'] = dark_val['b*']
compare['names'] = dark_val['name_y']
compare['Ls_mean'] = dark_val['Ls_mean']
compare['As_mean'] = dark_val['As_mean']
compare['Bs_mean'] = dark_val['Bs_mean']

compare['L_error'] = np.abs((compare['L*'] - compare['L_pred'])) / compare['L*'].abs()
compare['A_error'] = np.abs((compare['a*'] - compare['A_pred'])) / compare['a*'].abs()
compare['B_error'] = np.abs((compare['b*'] - compare['B_pred'])) / compare['b*'].abs()
compare['total_error'] = (compare['L_error'] + compare['A_error'] + compare['B_error']) / 3


group = compare.groupby('names')
mean = group.mean()

mean[['L_pred', 'L*']]
mean[['A_pred', 'a*']]
mean[['B_pred', 'b*']]

#sns.scatterplot(data = mean, x = 'L*', y = 'L_pred', hue = 'names')
sns.scatterplot(data = mean, x = 'L*', y = 'L_pred')
line = linregress(mean['L*'],mean['L_pred'])
sns.lineplot(x = mean['L*'], y = mean['L*'] * line.slope + line.intercept, label = 'r2: ' + str(round(line.rvalue **2, 3)))
#plt.legend(bbox_to_anchor = (1,1))
plt.title('Dark Model- Validation Data')

l_rmse = mean_squared_error(mean['L*'], mean['L_pred'], squared = False)
a_rmse = mean_squared_error(mean['a*'], mean['A_pred'], squared = False)
b_rmse = mean_squared_error(mean['b*'], mean['B_pred'], squared = False)

mean_cut = mean.loc[mean['L*'] < 60]
mean_cut[['L_pred', 'L*']]

l_rmse = mean_squared_error(mean_cut['L*'], mean_cut['L_pred'], squared = False)
compare['total_error'].mean()

print('VALIDATION RMSE VALUES')
print(l_rmse)
print(a_rmse)
print(b_rmse)

# In[RUN THE MODEL ON THE ALL DATA MODEL - IN THEORY REPRESNTATIVE OF BOTH SETS OF COLORS]

all_val = pd.concat([light_val, dark_val])
all_val = all_val.reset_index()
all_val_in = all_val[['Ls_mean', 'As_mean', 'Bs_mean', 'count_above_std1_v']]

scale = all_scaler.transform(all_val_in)

pred = all_ann.predict(scale)
pred_df = pd.DataFrame(pred)
pred_df.columns = ['L_pred', 'A_pred', 'B_pred']

compare = pred_df
compare['L*'] = all_val['L*']
compare['a*'] = all_val['a*']
compare['b*'] = all_val['b*']
compare['Ls_mean'] = all_val['Ls_mean']
compare['As_mean'] = all_val['As_mean']
compare['Bs_mean'] = all_val['Bs_mean']
compare['names'] = all_val['name_y']
compare['lighting'] = all_val['lighting']

compare['L_error'] = np.abs((compare['L*'] - compare['L_pred'])) / compare['L*'].abs()
compare['A_error'] = np.abs((compare['a*'] - compare['A_pred'])) / compare['a*'].abs()
compare['B_error'] = np.abs((compare['b*'] - compare['B_pred'])) / compare['b*'].abs()
compare['total_error'] = (compare['L_error'] + compare['A_error'] + compare['B_error']) / 3

group = compare.groupby(['names', 'lighting'])
mean = group.mean()

mean[['L_pred', 'L*']]
mean[['A_pred', 'a*']]
mean[['B_pred', 'b*']]

plt.style.use('seaborn')
sns.scatterplot(data = mean, x = 'L*', y = 'L_pred', hue = 'lighting')
plt.xlabel('Colorimeter L* Value')
plt.ylabel('Predicted L Value')
plt.legend(bbox_to_anchor = (1,1))

l_rmse = mean_squared_error(mean['L*'], mean['L_pred'], squared = False)
a_rmse = mean_squared_error(mean['a*'], mean['A_pred'], squared = False)
b_rmse = mean_squared_error(mean['b*'], mean['B_pred'], squared = False)

mean_cut = mean.loc[mean['L*'] < 60]
mean_cut[['L_pred', 'L*']]

l_rmse = mean_squared_error(mean_cut['L*'], mean_cut['L_pred'], squared = False)
compare['total_error'].mean()

print('VALIDATION RMSE VALUES')
print(l_rmse)
print(a_rmse)
print(b_rmse)

# In[RUN IT ON THE INTERMEIDATE WEIGTING]

med1 = pd.read_csv(r'L:\Kokini Lab\Kara Benbow\project_main\process_outputs\mediate1_validation.csv')
med1['abs_slope_h'] = med1['random_slope_h'].abs()
med1['abs_slope_v'] = med1['random_slope1_v'].abs()

weights = pd.read_csv(r'L:\Kokini Lab\Kara Benbow\project_main\process_outputs\mediate1_background_weights.csv')
weights = weights.drop(['Unnamed: 0'], axis = 1)

light_weight = weights['light_weight1']
dark_weight = weights['dark_weight1']

med_in_light = med1[['Ls_mean', 'As_mean', 'Bs_mean', 'abs_slope_h', 'count_above_std1_h']]
med_in_dark = med1[['Bog_mean', 'Gog_mean', 'Rog_mean', 'abs_slope_h', 'abs_slope_v']]
med_in_all = med1[['Ls_mean', 'As_mean', 'Bs_mean', 'count_above_std1_v']]

light_scale = light_scaler.transform(med_in_light)
dark_scale = dark_scaler.transform(med_in_dark)
all_scale = all_scaler.transform(med_in_all)

light_pred = light_ann.predict(light_scale)
dark_pred = dark_ann.predict(dark_scale)
all_pred = all_ann.predict(all_scale)

# ALL THE PREDS DF
all_preds = pd.DataFrame(all_pred)
all_preds['model'] = 'all'
all_preds['names'] = med1['name_y']

all_group = all_preds.groupby('names')
all_mean = all_group.mean()

# THE LIGHT PREDS DF AND WEIGHTING
light_preds = pd.DataFrame(light_pred)
light_preds['model'] = 'light'
light_preds['names'] = med1['name_y']

light_group = light_preds.groupby('names')
light_mean = light_group.mean()

light_weighted = light_mean * light_weight

# THE DARK PREDS DF AND WEIGHTING
dark_preds = pd.DataFrame(dark_pred)
dark_preds['model'] = 'dark'
dark_preds['names'] = med1['name_y']

dark_group = dark_preds.groupby('names')
dark_mean = dark_group.mean()

dark_weighted = dark_mean * dark_weight

weighted_pred = light_weighted + dark_weighted
weighted_pred.columns = ['L_pred','A_pred', 'B_pred']

og_group = med1.groupby(by = 'name_y')
og_mean = og_group.mean()

weighted_pred['L*'] = og_mean['L*']
weighted_pred['a*'] = og_mean['a*']
weighted_pred['b*'] = og_mean['b*']

weighted_pred['L_pred_all'] = all_mean[0]
weighted_pred['A_pred_all'] = all_mean[0]
weighted_pred['B_pred_all'] = all_mean[0]

weighted_pred['L_error'] = np.abs(weighted_pred['L_pred'] - weighted_pred['L*']) / weighted_pred['L*']
weighted_pred['A_error'] = np.abs(weighted_pred['A_pred'] - weighted_pred['a*']) / weighted_pred['a*']
weighted_pred['B_error'] = np.abs(weighted_pred['B_pred'] - weighted_pred['b*']) / weighted_pred['b*']

weighted_pred['tota error'] = np.sqrt((weighted_pred['L_pred'] - weighted_pred['L*'] ) **2 + 
                                      (weighted_pred['A_pred'] - weighted_pred['a*'] ) **2 + 
                                      (weighted_pred['B_pred'] - weighted_pred['b*'] ) **2)


#weighted_pred = weighted_pred.loc[(weighted_pred['L*'] > 30) &  (weighted_pred['L*'] < 70)]

l_rmse = mean_squared_error(weighted_pred['L*'], weighted_pred['L_pred'], squared = False)
a_rmse = mean_squared_error(weighted_pred['a*'], weighted_pred['A_pred'], squared = False)
b_rmse = mean_squared_error(weighted_pred['b*'], weighted_pred['B_pred'], squared = False)

l_rmse_all = mean_squared_error(weighted_pred['L*'], weighted_pred['L_pred_all'], squared = False)
a_rmse_all = mean_squared_error(weighted_pred['a*'], weighted_pred['A_pred_all'], squared = False)
b_rmse_all = mean_squared_error(weighted_pred['b*'], weighted_pred['B_pred_all'], squared = False)

         
''' 
plt.figure(1)
line = linregress(weighted_pred['a*'],weighted_pred['A_pred'])
sns.scatterplot(data = weighted_pred, x = 'a*', y = 'A_pred')
sns.lineplot(x = weighted_pred['a*'], y = weighted_pred['a*'] * line.slope + line.intercept, label = 'r2: ' + str(round(line.rvalue **2, 3)) + '\n ' + str(round(l_rmse, 3)))
plt.legend(bbox_to_anchor = (1,1))

plt.figure(2)
line = linregress(weighted_pred['b*'],weighted_pred['B_pred'])
sns.scatterplot(data = weighted_pred, x = 'b*', y = 'B_pred')
sns.lineplot(x = weighted_pred['b*'], y = weighted_pred['b*'] * line.slope + line.intercept, label = 'r2: ' + str(round(line.rvalue **2, 3)) + '\n ' + str(round(l_rmse, 3)))
plt.legend(bbox_to_anchor = (1,1))
'''

print('VALIDATION RMSE VALUES- WEIGHTED AVERAGE MODEL')
print(l_rmse)
print(a_rmse)
print(b_rmse)

print('VALIDATION RMSE VALUES- ORIGINAL ALL DATA MODEL')
print(l_rmse_all)
print(a_rmse_all)
print(b_rmse_all)

weighted_pred['tota error'].mean()

# In[]

plt.figure(0)
line = linregress(weighted_pred['L*'],weighted_pred['L_pred'])
sns.scatterplot(data = weighted_pred, x = 'L*', y = 'L_pred')
sns.lineplot(x = weighted_pred['L*'], y = weighted_pred['L*'] * line.slope + line.intercept, 
             label = 'Weighted Model: \n' + 'r2: ' + str(round(line.rvalue **2, 3)))
plt.xlabel('Colorimeter L* Value')
plt.ylabel('Predicted L Value')

plt.figure(0)
line = linregress(weighted_pred['L*'],weighted_pred['L_pred_all'])
sns.scatterplot(data = weighted_pred, x = 'L*', y = 'L_pred_all')
sns.lineplot(x = weighted_pred['L*'], y = weighted_pred['L*'] * line.slope + line.intercept, 
             label = 'All Data Model: \n' +'r2: ' + str(round(line.rvalue **2, 3)))
plt.xlabel('Colorimeter L* Value')
plt.ylabel('Predicted L Value')

plt.legend()

plt.savefig(r'L:\Kokini Lab\Kara Benbow\project_main_v2\figures\validation_l.png', dpi = 400, bbox_inches = 'tight')

# In[]

plt.figure(0)
line = linregress(weighted_pred['a*'],weighted_pred['A_pred'])
sns.scatterplot(data = weighted_pred, x = 'a*', y = 'A_pred')
sns.lineplot(x = weighted_pred['a*'], y = weighted_pred['a*'] * line.slope + line.intercept, 
             label = 'Weighted Model: \n' + 'r2: ' + str(round(line.rvalue **2, 3)))
plt.legend(bbox_to_anchor = (1,1))
plt.xlabel('Colorimeter a* Value')
plt.ylabel('Predicted A Value')

plt.figure(0)
line = linregress(weighted_pred['a*'],weighted_pred['A_pred_all'])
sns.scatterplot(data = weighted_pred, x = 'a*', y = 'A_pred_all')
sns.lineplot(x = weighted_pred['a*'], y = weighted_pred['a*'] * line.slope + line.intercept, 
             label = 'All Data Model: \n' +'r2: ' + str(round(line.rvalue **2, 3)))
plt.legend(bbox_to_anchor = (1,1))
plt.xlabel('Colorimeter a* Value')
plt.ylabel('Predicted A Value')

plt.legend()

plt.savefig(r'L:\Kokini Lab\Kara Benbow\project_main_v2\figures\validation_a.png', dpi = 400, bbox_inches = 'tight')


# In[]

plt.figure(0)
line = linregress(weighted_pred['b*'],weighted_pred['B_pred'])
sns.scatterplot(data = weighted_pred, x = 'b*', y = 'B_pred')
sns.lineplot(x = weighted_pred['b*'], y = weighted_pred['b*'] * line.slope + line.intercept, 
             label = 'Weighted Model: \n' + 'r2: ' + str(round(line.rvalue **2, 3)))
plt.xlabel('Colorimeter b* Value')
plt.ylabel('Predicted A Value')

plt.figure(0)
line = linregress(weighted_pred['b*'],weighted_pred['B_pred_all'])
sns.scatterplot(data = weighted_pred, x = 'b*', y = 'B_pred_all')
sns.lineplot(x = weighted_pred['b*'], y = weighted_pred['b*'] * line.slope + line.intercept, 
             label = 'All Data Model: \n' +'r2: ' + str(round(line.rvalue **2, 3)))
plt.xlabel('Colorimeter b* Value')
plt.ylabel('Predicted A Value')

plt.legend()

plt.savefig(r'L:\Kokini Lab\Kara Benbow\project_main_v2\figures\validation_b.png', dpi = 400, bbox_inches = 'tight')


# In[]

plt.figure(0)
line = linregress(weighted_pred['L*'],weighted_pred['L_pred'])
sns.scatterplot(data = weighted_pred, x = 'L*', y = 'L_pred')
sns.lineplot(x = weighted_pred['L*'], y = weighted_pred['L*'] * line.slope + line.intercept, label = 'r2: ' + str(round(line.rvalue **2, 3)))
#plt.legend(bbox_to_anchor = (1,1))
plt.xlabel('Colorimeter L* Value')
plt.ylabel('Predicted L Value')

plt.figure(1)
line = linregress(weighted_pred['L*'],weighted_pred['L_pred_all'])
sns.scatterplot(data = weighted_pred, x = 'L*', y = 'L_pred_all')
sns.lineplot(x = weighted_pred['L*'], y = weighted_pred['L*'] * line.slope + line.intercept, label = 'r2: ' + str(round(line.rvalue **2, 3)))
#plt.legend(bbox_to_anchor = (1,1))
plt.xlabel('Colorimeter L* Value')
plt.ylabel('Predicted L Value')







