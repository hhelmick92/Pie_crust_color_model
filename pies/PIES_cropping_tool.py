
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:59:00 2023

@author: hhelmick
"""

import glob
import os

import cv2

from PIES_functions import background_grabber
from PIES_functions import crop_to_edges

import time as count_time

# In[]

folders = list()
for root, dirs, files in os.walk(r"PATH\TO\THE\IMAGES\TO\PROCESS", topdown=False):
    for name in dirs:
        folders.append(os.path.join(root, name))

names = []
for f in folders:
    t1 = f.split('\\')
    names.append(t1[-1])

imgs = []

for f in folders:
    t1 = glob.glob(f + '\*.jpg')
    imgs.append(t1)


def grab_time(list_in, desired_time):
    
    t1 = []
    for ims in list_in:
        for i in ims:
            if desired_time in i:
                t1.append(i)
    
    t_names = []
    for i in t1:
        t2 = i.split('\\')
        t_names.append(t2[-2] + '_' + t2[-1].lower().replace('.jpg', '.png'))

    
    return [t1, t_names]

t15 = grab_time(imgs, 'time15')    
t30 = grab_time(imgs, 'time30')    

len(t15[0])
len(t30[0])

# In[]

test = crop_to_edges(t15[0][1], print_image = True, edge_thresh = 75)

# In[]

def write_out_backs_pies(imgs_to_grab, put_backs_here, put_pies_here, treshold_in = 75):
    
    if type(imgs_to_grab) == str:
        files = glob.glob(imgs_to_grab  + '\*.png')
    
        names = []
        for f in files:
            t1 = f.split('\\')
            names.append(t1[-1])
    
    elif type(imgs_to_grab) == list:
        files = imgs_to_grab
        
        names = []
        for f in files:
            t2 = f.split('\\')
            names.append(t2[-2] + '_' + t2[-1].lower().replace('.jpg', '.png'))
        
    for fn, n in zip(files, names):
        im = background_grabber(fn, show = False, thresh = treshold_in)
        im2 = crop_to_edges(fn, print_image = False, edge_thresh = treshold_in)
        
        cv2.imwrite(put_backs_here + '\{}'.format(n), im)
        cv2.imwrite(put_pies_here + '\{}'.format(n), im2)

# In[CROP OUT THE 15 MINUTES BAKE DATA USING A THRESHOLD OF 75]

start_time = count_time.time()
# WRITE OUT THE CROPPED IMAGES FOR EASIER FUTURE ANALYSIS. CROPPING IS DONE PER TIME BASED ON THRESHOLD VALUES
write_out_backs_pies(
    imgs_to_grab = t15[0],
    put_backs_here = r'PATH\TO\WHERE\TO\SAVE\DATA\BACKGROUND',
    put_pies_here = r'PATH\TO\WHERE\TO\SAVE\DATA\CROPPED_IMAGES',
    treshold_in = 75
    )

elap = count_time.time() - start_time
# for full data set, elap = 5.166666667

# In[]

done = glob.glob(r'PATH\TO\CROPPED\IMAGES\*.png')

done_names = []
for fn in done:
    t1 = (fn.split('\\')[-1][-8:])
    #failed_names.append(t1)
    done_names.append(t1.lower().replace('.png', '.jpg'))

done2 = []
for f in t30[0]:
    f1 = f.split('\\')
    f2 = (f1[-1]).lower()
    for n in done_names:
        if n in f2:
            done2.append(f)

not_done = list(set(t30[0]) - set(done2))
           
len(done)
len(not_done)
len(t30[0])
len(not_done) + len(done) == len(t30[0])

# In[]

start_time = count_time.time()
write_out_backs_pies(
    imgs_to_grab = not_done,
    put_backs_here = r'PATH\TO\WHERE\TO\SAVE\DATA\BACKGROUND',
    put_pies_here = r'PATH\TO\WHERE\TO\SAVE\DATA\CROPPED_IMAGES',
    treshold_in = 200
    )
elap = count_time.time() - start_time
print('processing time, s / img')
print((elap / len(not_done)))






