# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:17:47 2023

@author: hhelmick
"""


import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from scipy.stats import linregress

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import pickle

import time


def linear_fit(x, m, b):
    return m*x + b

def scale_data (df):
    scaler = StandardScaler()
    scaler.fit(df)
    x_scaled = scaler.transform(df)
    return x_scaled

# In[]

all_train = pd.read_csv(r'L:\Kokini Lab\Kara Benbow\project_main\data2\train_data.csv')
all_val = pd.read_csv(r'L:\Kokini Lab\Kara Benbow\project_main\data2\val.csv')

light_train = all_train.loc[all_train['lighting'] == 'high']
dark_train = all_train.loc[all_train['lighting'] == 'low']

light_val = all_val.loc[all_val['lighting'] == 'high']
dark_val = all_val.loc[all_val['lighting'] == 'low']

# In[]

df = all_train

x = df[['Ls_mean', 'As_mean', 'Bs_mean', 'count_above_std1_v']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

scaler = StandardScaler()
fit = scaler.fit(x)
x_scale = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (20, 20, 20), 'learning_rate': 'constant', 'solver': 'adam'}

regr = MLPRegressor(max_iter = 5000,
                    activation = grid['activation'],
                    alpha = grid['alpha'],
                    hidden_layer_sizes = (grid['hidden_layer_sizes']),
                    learning_rate = grid['learning_rate'],
                    solver = grid['solver'],
                    random_state = 26
                    )

regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
preds = pd.DataFrame(y_pred)
preds.columns = ['L_pred', 'A_pred', 'B_pred']

rmse_L = np.sqrt(mean_squared_error(y_test['L*'], preds['L_pred']))
rmse_A = np.sqrt(mean_squared_error(y_test['a*'], preds['A_pred']))
rmse_B = np.sqrt(mean_squared_error(y_test['b*'], preds['B_pred']))

r_L = y_test2['L*'].corr(preds['L_pred'])
r_A = y_test2['a*'].corr(preds['A_pred'])
r_B = y_test2['b*'].corr(preds['B_pred'])

'''
name = r'L:\Kokini Lab\Kara Benbow\project_main\code2\selected_models\ann_all_data.pkl'
with open(name, 'wb') as file:
    pickle.dump(regr, file)

name = r'L:\Kokini Lab\Kara Benbow\project_main\code2\selected_models\all_scaler.pkl'
with open(name, 'wb') as file:
    pickle.dump(fit, file)
'''

# In[]

L_lin = linregress(preds['L_pred'], y_test['L*'])
A_lin = linregress(preds['A_pred'], y_test['a*'])
B_lin = linregress(preds['B_pred'], y_test['b*'])

train_melt = preds.melt()
test_melt = y_test.melt()
all_lin = linregress(train_melt['value'], test_melt['value'])

fig, ax = plt.subplots(2,2)
fig.tight_layout(h_pad = 2)

ax[0,0].scatter(preds['L_pred'], y_test['L*'])
ax[0,0].plot(preds['L_pred'], L_lin.slope * preds['L_pred'] + L_lin.intercept, color = 'red', label = str(round(L_lin.rvalue **2, 3)))
ax[0,0].set_title('L values')
ax[0,0].set_ylabel('Experimental Values')
ax[0,0].legend()

ax[0,1].scatter(preds['A_pred'], y_test['a*'])
ax[0,1].plot(preds['A_pred'], A_lin.slope * preds['A_pred'] + A_lin.intercept, color = 'red', label = str(round(A_lin.rvalue **2, 3)))
ax[0,1].set_title('A values')
ax[0,1].legend()

ax[1,0].scatter(preds['B_pred'], y_test['b*'])
ax[1,0].plot(preds['B_pred'], B_lin.slope * preds['B_pred'] + B_lin.intercept, color = 'red', label = str(round(B_lin.rvalue **2, 3)))
ax[1,0].set_title('B values')
ax[1,0].set_xlabel('Predicted Values')
ax[1,0].set_ylabel('Experimental Values')
ax[1,0].legend()

ax[1,1].scatter(preds, y_test)
ax[1,1].plot(train_melt['value'], all_lin.slope * train_melt['value'] + all_lin.intercept, color = 'red', label = str(round(all_lin.rvalue **2, 3)))
ax[1,1].set_title('All values')
ax[1,1].set_xlabel('Predicted Values')
ax[1,1].legend()

plt.savefig(r'L:\Kokini Lab\Kara Benbow\project_main_v2\figures\all_data.png', dpi = 400)

# In[]

df = light_train

x = df[['Ls_mean', 'As_mean', 'Bs_mean', 'abs_slope_h', 'count_above_std1_h']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

scaler = StandardScaler()
fit = scaler.fit(x)
x_scale = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (20, 20, 20), 'learning_rate': 'adaptive', 'solver': 'sgd'}

regr = MLPRegressor(max_iter = 5000,
                    activation = grid['activation'],
                    alpha = grid['alpha'],
                    hidden_layer_sizes = (grid['hidden_layer_sizes']),
                    learning_rate = grid['learning_rate'],
                    solver = grid['solver'],
                    random_state = 26
                    )

regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
preds = pd.DataFrame(y_pred)
preds.columns = ['L_pred', 'A_pred', 'B_pred']

rmse_L = np.sqrt(mean_squared_error(y_test['L*'], preds['L_pred']))
rmse_A = np.sqrt(mean_squared_error(y_test['a*'], preds['A_pred']))
rmse_B = np.sqrt(mean_squared_error(y_test['b*'], preds['B_pred']))

r_L = y_test2['L*'].corr(preds['L_pred'])
r_A = y_test2['a*'].corr(preds['A_pred'])
r_B = y_test2['b*'].corr(preds['B_pred'])

'''
name = r'L:\Kokini Lab\Kara Benbow\project_main\code2\selected_models\ann_light_data.pkl'
with open(name, 'wb') as file:
    pickle.dump(regr, file)

name = r'L:\Kokini Lab\Kara Benbow\project_main\code2\selected_models\light_scaler.pkl'
with open(name, 'wb') as file:
    pickle.dump(fit, file)
'''

L_lin = linregress(preds['L_pred'], y_test['L*'])
A_lin = linregress(preds['A_pred'], y_test['a*'])
B_lin = linregress(preds['B_pred'], y_test['b*'])

train_melt = preds.melt()
test_melt = y_test.melt()
all_lin = linregress(train_melt['value'], test_melt['value'])

fig, ax = plt.subplots(2,2)
fig.tight_layout(h_pad = 2)

ax[0,0].scatter(preds['L_pred'], y_test['L*'])
ax[0,0].plot(preds['L_pred'], L_lin.slope * preds['L_pred'] + L_lin.intercept, color = 'red', label = str(round(L_lin.rvalue **2, 3)))
ax[0,0].set_title('L values')
ax[0,0].set_ylabel('Experimental Values')
ax[0,0].legend()

ax[0,1].scatter(preds['A_pred'], y_test['a*'])
ax[0,1].plot(preds['A_pred'], A_lin.slope * preds['A_pred'] + A_lin.intercept, color = 'red', label = str(round(A_lin.rvalue **2, 3)))
ax[0,1].set_title('A values')
ax[0,1].legend()

ax[1,0].scatter(preds['B_pred'], y_test['b*'])
ax[1,0].plot(preds['B_pred'], B_lin.slope * preds['B_pred'] + B_lin.intercept, color = 'red', label = str(round(B_lin.rvalue **2, 3)))
ax[1,0].set_title('B values')
ax[1,0].set_xlabel('Predicted Values')
ax[1,0].set_ylabel('Experimental Values')
ax[1,0].legend()

ax[1,1].scatter(preds, y_test)
ax[1,1].plot(train_melt['value'], all_lin.slope * train_melt['value'] + all_lin.intercept, color = 'red', label = str(round(all_lin.rvalue **2, 3)))
ax[1,1].set_title('All values')
ax[1,1].set_xlabel('Predicted Values')
ax[1,1].legend()

plt.savefig(r'L:\Kokini Lab\Kara Benbow\project_main_v2\figures\light_data.png', dpi = 400)

# In[]

df = dark_train

x = df[['Bog_mean', 'Gog_mean', 'Rog_mean', 'abs_slope_h', 'abs_slope_v']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

scaler = StandardScaler()
fit = scaler.fit(x)
x_scale = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (20, 20), 'learning_rate': 'adaptive', 'solver': 'sgd'}

regr = MLPRegressor(max_iter = 5000,
                    activation = grid['activation'],
                    alpha = grid['alpha'],
                    hidden_layer_sizes = (grid['hidden_layer_sizes']),
                    learning_rate = grid['learning_rate'],
                    solver = grid['solver'],
                    random_state = 26
                    )

regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
preds = pd.DataFrame(y_pred)
preds.columns = ['L_pred', 'A_pred', 'B_pred']

rmse_L = np.sqrt(mean_squared_error(y_test['L*'], preds['L_pred']))
rmse_A = np.sqrt(mean_squared_error(y_test['a*'], preds['A_pred']))
rmse_B = np.sqrt(mean_squared_error(y_test['b*'], preds['B_pred']))

r_L = y_test2['L*'].corr(preds['L_pred'])
r_A = y_test2['a*'].corr(preds['A_pred'])
r_B = y_test2['b*'].corr(preds['B_pred'])

'''
name = r'L:\Kokini Lab\Kara Benbow\project_main\code2\selected_models\ann_dark_data.pkl'
with open(name, 'wb') as file:
    pickle.dump(regr, file)

name = r'L:\Kokini Lab\Kara Benbow\project_main\code2\selected_models\dark_scaler.pkl'
with open(name, 'wb') as file:
    pickle.dump(fit, file)
'''
L_lin = linregress(preds['L_pred'], y_test['L*'])
A_lin = linregress(preds['A_pred'], y_test['a*'])
B_lin = linregress(preds['B_pred'], y_test['b*'])

train_melt = preds.melt()
test_melt = y_test.melt()
all_lin = linregress(train_melt['value'], test_melt['value'])

fig, ax = plt.subplots(2,2)
fig.tight_layout(h_pad = 2)

ax[0,0].scatter(preds['L_pred'], y_test['L*'])
ax[0,0].plot(preds['L_pred'], L_lin.slope * preds['L_pred'] + L_lin.intercept, color = 'red', label = str(round(L_lin.rvalue **2, 3)))
ax[0,0].set_title('L values')
ax[0,0].set_ylabel('Experimental Values')
ax[0,0].legend()

ax[0,1].scatter(preds['A_pred'], y_test['a*'])
ax[0,1].plot(preds['A_pred'], A_lin.slope * preds['A_pred'] + A_lin.intercept, color = 'red', label = str(round(A_lin.rvalue **2, 3)))
ax[0,1].set_title('A values')
ax[0,1].legend()

ax[1,0].scatter(preds['B_pred'], y_test['b*'])
ax[1,0].plot(preds['B_pred'], B_lin.slope * preds['B_pred'] + B_lin.intercept, color = 'red', label = str(round(B_lin.rvalue **2, 3)))
ax[1,0].set_title('B values')
ax[1,0].set_xlabel('Predicted Values')
ax[1,0].set_ylabel('Experimental Values')
ax[1,0].legend()

ax[1,1].scatter(preds, y_test)
ax[1,1].plot(train_melt['value'], all_lin.slope * train_melt['value'] + all_lin.intercept, color = 'red', label = str(round(all_lin.rvalue **2, 3)))
ax[1,1].set_title('All values')
ax[1,1].set_xlabel('Predicted Values')
ax[1,1].legend()

plt.savefig(r'L:\Kokini Lab\Kara Benbow\project_main_v2\figures\dark_data.png', dpi = 400)















