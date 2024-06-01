# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 12:54:25 2023

@author: hhelmick
"""

import pandas as pd
# In[]

df1 = pd.read_csv(r'PATH\TO\THIS\FILE\ppg_highlight_averages.csv')
df2 = pd.read_csv(r'PATH\TO\THIS\FILE\sw_highlight_averages.csv')

df3 = pd.read_csv(r'PATH\TO\THIS\FILE\ppg_lowlight_averages.csv')
df4 = pd.read_csv(r'PATH\TO\THIS\FILE\sw_lowlight_averages.csv')

df = pd.concat([df1, df2, df3, df4])

df['abs_slope_h'] = df['random_slope_h'].abs()
df['abs_slope_v'] = df['random_slope1_v'].abs()

df = df.reset_index()
df = df.drop(['index', 'Unnamed: 0'], axis = 1)

light = df.loc[df['lighting'] == 'high']
dark = df.loc[df['lighting'] == 'low']

# In[]

colors = []

for color in df['color'].unique():
    cdf = df.loc[df['color'] == color]
    samp = cdf['ID'].sample(n = 1, random_state = 26)
    colors.append(samp)

samples = []
for c in colors:
    t1 = str(c.item())
    t2 = t1.split('_')
    t3 = t2[0] + '_' + t2[1] + '_' + t2[2]
    samples.append(t3)
