# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:02:08 2023

@author: hhelmick
"""

import numpy as np
import pandas as pd


from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from scipy.stats import linregress

import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns

import time

# In[]

def linear_fit(x, m, b):
    return m*x + b

def scale_data (df):
    scaler = StandardScaler()
    scaler.fit(df)
    x_scaled = scaler.transform(df)
    return x_scaled

# FAIRLY AVERAGE GRID CHECK
def grid_check(x_trainer, y_trainer):
    clf = MLPRegressor(max_iter = 5000)
    
    check_parameters = {
        'hidden_layer_sizes' : [
            (5,5,5), (10,10,10), (20,20,20),
            (5,5), (10,10), (20,20)],
        'activation' : ['tanh', 'relu'],
        'solver' : ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate':['constant', 'adaptive']
    }

    # conduct a grid serach based on th eboave classigiers. Cv is a split function to further into
    gridsearchcv = GridSearchCV(clf, check_parameters, n_jobs = -1, cv =2)
    gridsearchcv.fit(x_trainer, y_trainer)
    
    #print('Best parameters found:\n', gridsearchcv.best_params_)
    
    return gridsearchcv.best_params_

# In[]
t1 = time.time()

df1 = pd.read_csv(r'PATH\TO\THIS\FILE\ppg_highlight_averages.csv')
df2 = pd.read_csv(r'PATH\TO\THIS\FILE\sw_highlight_averages.csv')

df3 = pd.read_csv(r'PATH\TO\THIS\FILE\ppg_lowlight_averages.csv')
df4 = pd.read_csv(r'PATH\TO\THIS\FILE\sw_lowlight_averages.csv')

df = pd.concat([df3, df4])

df['abs_slope_h'] = df['random_slope_h'].abs()
df['abs_slope_v'] = df['random_slope1_v'].abs()

# In[]

models = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6', 'model7', 'model8', 'model9', 'model10',
          'model11', 'model12', 'model13', 'model14'
          ]
results_RMSE_L = []
results_RMSE_A = []
results_RMSE_B = []

results_R_L = []
results_R_A = []
results_R_B = []

params = []

# In[MODEL 1]

'''
###########################################################################################################################################

IN THIS SECTION I BASICALLY FOLLOW THIS PAPER
https://www.sciencedirect.com/science/article/pii/S0963996906000470

HIGHLIGHTS:
    THEY TAKE THE AVERAGE R, G, AND B VALUES FROM DIFFERENT COLORS
    THEY USE THIS TO PREDICT THE L, A, AND B VALUES FROM HUNTER METER
###########################################################################################################################################

'''

x = df[['Bog_mean', 'Gog_mean', 'Rog_mean']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

x_scale = scale_data(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = grid_check(x_train, y_train)
print('Model 1')
print(grid)

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

results_R_L.append(r_L)
results_R_A.append(r_A)
results_R_B.append(r_B)

results_RMSE_L.append(rmse_L)
results_RMSE_A.append(rmse_A)
results_RMSE_B.append(rmse_B)

params.append(grid)

# In[MODEL 2]

'''
###########################################################################################################################################
IN THIS SECTION I ADD IN THE ABSOLUTE VALUE OF THE SLOPES THAT IS CALCULATED IN THE HORIZANTAL DIRECTION
###########################################################################################################################################
'''

x = df[['Bog_mean', 'Gog_mean', 'Rog_mean', 'abs_slope_h']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

x_scale = scale_data(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = grid_check(x_train, y_train)
print('Model 2')
print(grid)

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

results_R_L.append(r_L)
results_R_A.append(r_A)
results_R_B.append(r_B)

results_RMSE_L.append(rmse_L)
results_RMSE_A.append(rmse_A)
results_RMSE_B.append(rmse_B)

params.append(grid)

# In[MODEL 3]

'''
###########################################################################################################################################
IN THIS SECTION I ADD IN THE COUNT OF VALUES GREATER THAN A STD ABOVE THE MEAN OF THE SLOPING LINE
###########################################################################################################################################
'''

x = df[['Bog_mean', 'Gog_mean', 'Rog_mean', 'count_above_std1_h']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

x_scale = scale_data(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = grid_check(x_train, y_train)
print('Model 3')
print(grid)

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

results_R_L.append(r_L)
results_R_A.append(r_A)
results_R_B.append(r_B)

results_RMSE_L.append(rmse_L)
results_RMSE_A.append(rmse_A)
results_RMSE_B.append(rmse_B)

params.append(grid)

# In[MODEL 4]

'''
###########################################################################################################################################
IN THIS SECTION I ADD IN THE COUNT OF VALUES GREATER THAN A STD ABOVE THE MEAN OF THE SLOPING LINE
###########################################################################################################################################
'''

x = df[['Bog_mean', 'Gog_mean', 'Rog_mean', 'abs_slope_h', 'count_above_std1_h']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

x_scale = scale_data(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = grid_check(x_train, y_train)
print('Model 4')
print(grid)

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

results_R_L.append(r_L)
results_R_A.append(r_A)
results_R_B.append(r_B)

results_RMSE_L.append(rmse_L)
results_RMSE_A.append(rmse_A)
results_RMSE_B.append(rmse_B)

params.append(grid)

# In[MODEL 5]

'''
###########################################################################################################################################
IN THIS SECTION I ADD IN THE COUNT OF VALUES GREATER THAN A STD ABOVE THE MEAN OF THE SLOPING LINE
###########################################################################################################################################
'''

x = df[['Bog_mean', 'Gog_mean', 'Rog_mean', 'abs_slope_v']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

x_scale = scale_data(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = grid_check(x_train, y_train)
print('Model 5')
print(grid)

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

results_R_L.append(r_L)
results_R_A.append(r_A)
results_R_B.append(r_B)

results_RMSE_L.append(rmse_L)
results_RMSE_A.append(rmse_A)
results_RMSE_B.append(rmse_B)

params.append(grid)

# In[MODEL 6]

'''
###########################################################################################################################################
IN THIS SECTION I ADD IN THE COUNT OF VALUES GREATER THAN A STD ABOVE THE MEAN OF THE SLOPING LINE
###########################################################################################################################################
'''

x = df[['Bog_mean', 'Gog_mean', 'Rog_mean', 'count_above_std1_v']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

x_scale = scale_data(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = grid_check(x_train, y_train)
print('Model 6')
print(grid)

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

results_R_L.append(r_L)
results_R_A.append(r_A)
results_R_B.append(r_B)

results_RMSE_L.append(rmse_L)
results_RMSE_A.append(rmse_A)
results_RMSE_B.append(rmse_B)

params.append(grid)

# In[MODEL 7]

'''
###########################################################################################################################################
###########################################################################################################################################
'''

x = df[['Bog_mean', 'Gog_mean', 'Rog_mean', 'abs_slope_h', 'abs_slope_v']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

x_scale = scale_data(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = grid_check(x_train, y_train)
print('Model 7')
print(grid)

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

results_R_L.append(r_L)
results_R_A.append(r_A)
results_R_B.append(r_B)

results_RMSE_L.append(rmse_L)
results_RMSE_A.append(rmse_A)
results_RMSE_B.append(rmse_B)

params.append(grid)

# In[MODEL 8]

'''
###########################################################################################################################################

IN THIS SECTION I BASICALLY FOLLOW THIS PAPER
https://www.sciencedirect.com/science/article/pii/S0963996906000470

HIGHLIGHTS:
    THEY TAKE THE AVERAGE R, G, AND B VALUES FROM DIFFERENT COLORS
    THEY USE THIS TO PREDICT THE L, A, AND B VALUES FROM HUNTER METER
###########################################################################################################################################

'''

x = df[['Ls_mean', 'As_mean', 'Bs_mean']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

x_scale = scale_data(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = grid_check(x_train, y_train)
print('Model 8')
print(grid)

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

results_R_L.append(r_L)
results_R_A.append(r_A)
results_R_B.append(r_B)

results_RMSE_L.append(rmse_L)
results_RMSE_A.append(rmse_A)
results_RMSE_B.append(rmse_B)

params.append(grid)

# In[MODEL 9]

'''
###########################################################################################################################################
IN THIS SECTION I ADD IN THE ABSOLUTE VALUE OF THE SLOPES THAT IS CALCULATED IN THE HORIZANTAL DIRECTION
###########################################################################################################################################
'''

x = df[['Ls_mean', 'As_mean', 'Bs_mean', 'abs_slope_h']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

x_scale = scale_data(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = grid_check(x_train, y_train)
print('Model 9')
print(grid)

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

results_R_L.append(r_L)
results_R_A.append(r_A)
results_R_B.append(r_B)

results_RMSE_L.append(rmse_L)
results_RMSE_A.append(rmse_A)
results_RMSE_B.append(rmse_B)

params.append(grid)

# In[MODEL 10]

'''
###########################################################################################################################################
IN THIS SECTION I ADD IN THE COUNT OF VALUES GREATER THAN A STD ABOVE THE MEAN OF THE SLOPING LINE
###########################################################################################################################################
'''

x = df[['Ls_mean', 'As_mean', 'Bs_mean', 'count_above_std1_h']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

x_scale = scale_data(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = grid_check(x_train, y_train)
print('Model 10')
print(grid)

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

results_R_L.append(r_L)
results_R_A.append(r_A)
results_R_B.append(r_B)

results_RMSE_L.append(rmse_L)
results_RMSE_A.append(rmse_A)
results_RMSE_B.append(rmse_B)

params.append(grid)

# In[MODEL 11]

'''
###########################################################################################################################################
IN THIS SECTION I ADD IN THE COUNT OF VALUES GREATER THAN A STD ABOVE THE MEAN OF THE SLOPING LINE
###########################################################################################################################################
'''

x = df[['Ls_mean', 'As_mean', 'Bs_mean', 'abs_slope_h', 'count_above_std1_h']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

x_scale = scale_data(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = grid_check(x_train, y_train)
print('Model 11')
print(grid)

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

results_R_L.append(r_L)
results_R_A.append(r_A)
results_R_B.append(r_B)

results_RMSE_L.append(rmse_L)
results_RMSE_A.append(rmse_A)
results_RMSE_B.append(rmse_B)

params.append(grid)

# In[MODEL 12]

'''
###########################################################################################################################################
IN THIS SECTION I ADD IN THE COUNT OF VALUES GREATER THAN A STD ABOVE THE MEAN OF THE SLOPING LINE
###########################################################################################################################################
'''

x = df[['Ls_mean', 'As_mean', 'Bs_mean', 'Rog_mean', 'abs_slope_v']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

x_scale = scale_data(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = grid_check(x_train, y_train)
print('Model 12')
print(grid)

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

results_R_L.append(r_L)
results_R_A.append(r_A)
results_R_B.append(r_B)

results_RMSE_L.append(rmse_L)
results_RMSE_A.append(rmse_A)
results_RMSE_B.append(rmse_B)

params.append(grid)

# In[MODEL 13]

'''
###########################################################################################################################################
IN THIS SECTION I ADD IN THE COUNT OF VALUES GREATER THAN A STD ABOVE THE MEAN OF THE SLOPING LINE
###########################################################################################################################################
'''

x = df[['Ls_mean', 'As_mean', 'Bs_mean', 'count_above_std1_v']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

x_scale = scale_data(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = grid_check(x_train, y_train)
print('Model 13')
print(grid)

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

results_R_L.append(r_L)
results_R_A.append(r_A)
results_R_B.append(r_B)

results_RMSE_L.append(rmse_L)
results_RMSE_A.append(rmse_A)
results_RMSE_B.append(rmse_B)

params.append(grid)

# In[MODEL 14]

'''
###########################################################################################################################################
###########################################################################################################################################
'''

x = df[['Ls_mean', 'As_mean', 'Bs_mean', 'abs_slope_h', 'abs_slope_v']]
y = df[['L*', 'a*', 'b*']]
strat = df[['brand_x', 'lighting']]

x_scale = scale_data(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 26, stratify = strat)
y_test2 = y_test.reset_index()

grid = grid_check(x_train, y_train)
print('Model 14')
print(grid)

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

results_R_L.append(r_L)
results_R_A.append(r_A)
results_R_B.append(r_B)

results_RMSE_L.append(rmse_L)
results_RMSE_A.append(rmse_A)
results_RMSE_B.append(rmse_B)

params.append(grid)


# In[]

result_df = pd.DataFrame()
result_df['RMSE_L'] = results_RMSE_L
result_df['RMSE_A'] = results_RMSE_A
result_df['RMSE_B'] = results_RMSE_B
result_df['R_L'] = results_R_L
result_df['R_A'] = results_R_A
result_df['R_B'] = results_R_B
result_df['params'] = params
result_df.index = models

t2 = time.time() - t1

print('all these modesl took {} seconds to run'.format(str(round(t2, 3))))

result_df.to_csv(r'PATH\TO\THIS\FILE\grid_check_dark_only_ann.csv')

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
ax[0,0].legend()

ax[0,1].scatter(preds['A_pred'], y_test['a*'])
ax[0,1].plot(preds['A_pred'], A_lin.slope * preds['A_pred'] + A_lin.intercept, color = 'red', label = str(round(A_lin.rvalue **2, 3)))
ax[0,1].set_title('A values')
ax[0,1].legend()

ax[1,0].scatter(preds['B_pred'], y_test['b*'])
ax[1,0].plot(preds['B_pred'], B_lin.slope * preds['B_pred'] + B_lin.intercept, color = 'red', label = str(round(B_lin.rvalue **2, 3)))
ax[1,0].set_title('B values')
ax[1,0].legend()

ax[1,1].scatter(preds, y_test)
ax[1,1].plot(train_melt['value'], all_lin.slope * train_melt['value'] + all_lin.intercept, color = 'red', label = str(round(all_lin.rvalue **2, 3)))
ax[1,1].set_title('All values')
ax[1,1].legend()

















