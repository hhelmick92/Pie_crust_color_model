# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:02:08 2023

@author: hhelmick
"""

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from scipy.stats import linregress

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import time

from grid_funcs import ann_searching_val
from grid_funcs import grid_check
from grid_funcs import splitter
from grid_funcs import forest_searching_val

# In[]

all_train = pd.read_csv(r'L:\Kokini Lab\Kara Benbow\project_main\data2\train_data.csv')
all_val = pd.read_csv(r'L:\Kokini Lab\Kara Benbow\project_main\data2\val.csv')

light_train = all_train.loc[all_train['lighting'] == 'high']
dark_train = all_train.loc[all_train['lighting'] == 'low']

light_val = all_val.loc[all_val['lighting'] == 'high']
dark_val = all_val.loc[all_val['lighting'] == 'low']

# In[]

combos = [['Bog_mean', 'Gog_mean', 'Rog_mean'],
          ['Bog_mean', 'Gog_mean', 'Rog_mean', 'abs_slope_h'],
          ['Bog_mean', 'Gog_mean', 'Rog_mean', 'count_above_std1_h'],
          ['Bog_mean', 'Gog_mean', 'Rog_mean', 'abs_slope_h', 'count_above_std1_h'],
          ['Bog_mean', 'Gog_mean', 'Rog_mean', 'abs_slope_v'],
          ['Bog_mean', 'Gog_mean', 'Rog_mean', 'count_above_std1_v'],
          ['Bog_mean', 'Gog_mean', 'Rog_mean', 'abs_slope_h', 'abs_slope_v'],
          ['Ls_mean', 'As_mean', 'Bs_mean'],
          ['Ls_mean', 'As_mean', 'Bs_mean', 'abs_slope_h'],
          ['Ls_mean', 'As_mean', 'Bs_mean', 'count_above_std1_h'],
          ['Ls_mean', 'As_mean', 'Bs_mean', 'abs_slope_h', 'count_above_std1_h'],
          ['Ls_mean', 'As_mean', 'Bs_mean', 'Rog_mean', 'abs_slope_v'],
          ['Ls_mean', 'As_mean', 'Bs_mean', 'count_above_std1_v'],
          ['Ls_mean', 'As_mean', 'Bs_mean', 'abs_slope_h', 'abs_slope_v']
          ]

# In[]

start = time.time()

all_splits = []
light_splits = []
dark_splits = []

for c in combos:
    all_splits.append(splitter(all_train, c))
    light_splits.append(splitter(light_train, c))
    dark_splits.append(splitter(dark_train, c))

# In[]

all_grid = []
light_grid = []
dark_grid = []

for x in light_splits:   
    light_grid.append(forest_searching_val(x[0], x[1], x[2], x[3], x[4], x[5], light_val))
print('light forest is done')

for x in dark_splits:   
    dark_grid.append(forest_searching_val(x[0], x[1], x[2], x[3], x[4], x[5], dark_val))
print('dark forest is done')

for x in all_splits:
    all_grid.append(forest_searching_val(x[0], x[1], x[2], x[3], x[4], x[5], all_val))
print('all forest is done')

# In[]

all_grid_ann = []
light_grid_ann = []
dark_grid_ann = []

# In[]

for x in light_splits:   
    light_grid_ann.append(ann_searching_val(x[0], x[1], x[2], x[3], x[4], x[5], light_val))
print('light ANN is done')

# In[]

for x in dark_splits:   
    dark_grid_ann.append(ann_searching_val(x[0], x[1], x[2], x[3], x[4], x[5], dark_val))
print('dark ANN is done')

# In[]

for x in all_splits:
    all_grid_ann.append(ann_searching_val(x[0], x[1], x[2], x[3], x[4], x[5], all_val))
print('all ANN is done')

# In[]

cols = ['rmse_L_model', 'rmse_A_model', 'rmse_B_model', 'total_error_model', 
        'rmse_L_val', 'rmse_A_val', 'rmse_B_val', 'total_error_val', 'best_params']

numbers = [l for l in range(0, len(combos))]

idx = []
for n in numbers:
    idx.append('model_' + str(n))

all_results = pd.DataFrame(all_grid)
all_results.columns = cols
all_results.index = idx
all_results['DATA'] = 'ALL DATA'
all_results['TYPE'] = 'RF'

light_results = pd.DataFrame(light_grid)
light_results.columns = cols
light_results.index = idx
light_results['DATA'] = 'LIGHT DATA'
light_results['TYPE'] = 'RF'

dark_results = pd.DataFrame(dark_grid)
dark_results.columns = cols
dark_results.index = idx
dark_results['DATA'] = 'DARK DATA'
dark_results['TYPE'] = 'RF'

all_results_ann = pd.DataFrame(all_grid_ann)
all_results_ann.columns = cols
all_results_ann.index = idx
all_results_ann['DATA'] = 'ALL DATA'
all_results_ann['TYPE'] = 'ANN'

#all_results_ann.to_csv(r'L:\Kokini Lab\Kara Benbow\project_main\model_results2\ann_all_grid.csv')

light_results_ann  = pd.DataFrame(light_grid_ann)
light_results_ann.columns = cols
light_results_ann.index = idx
light_results_ann['DATA'] = 'LIGHT DATA'
light_results_ann['TYPE'] = 'ANN'

#light_results_ann.to_csv(r'L:\Kokini Lab\Kara Benbow\project_main\model_results2\ann_light_grid.csv')

dark_results_ann  = pd.DataFrame(dark_grid_ann)
dark_results_ann.columns = cols
dark_results_ann.index = idx
dark_results_ann['DATA'] = 'DARK DATA'
dark_results_ann['TYPE'] = 'ANN'

#dark_results_ann.to_csv(r'L:\Kokini Lab\Kara Benbow\project_main\model_results2\ann_dark_grid.csv')


out = pd.concat([all_results, light_results, dark_results,
                 all_results_ann, light_results_ann, dark_results_ann
                 ])
'''
out = pd.concat([all_results, light_results, dark_results,
                 ])
'''

out.to_csv(r'L:\Kokini Lab\Kara Benbow\project_main\model_results2\rf_grid_search.csv')

# In[]

elapse = time.time() - start

print('GRID SEARCHING RUN TIME')
print('SECONDS')
print(elapse)
print('MINUTES')
print(elapse / 60)
















