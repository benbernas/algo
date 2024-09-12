# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:13:35 2024

@author: Benoît Bernas
"""
import sys
import os

import scipy.io
import h5py


import matplotlib.pyplot as plt
import numpy as np
#%% define path directory and call functions from 'my module'

dir_path = r'C:\Benoit\deep_learning'
os.chdir(dir_path)
simu_path = dir_path + r'\field_simu\2024-08-23_10blocks_5scats_no_noise'
save_fig = 1

#path database
# D:\Benoit\machine_learning\python\deep_learning\algo\generate_database\database
load_path = dir_path + r'\algo\generate_database\database\database_IQ_mb_noised_sizeImg_91_128_mb_5.h5'

#path containing functions 
function_path = dir_path + r'\algo\function'
# Add the directory to sys.path
sys.path.append(function_path)
from my_module import check_data
#%% load data


# result_path = dir_path  + folder_path
#%% 
#%% load database
    
os.makedirs(os.path.dirname(load_path), exist_ok=True)
BfStruct = scipy.io.loadmat(os.path.join(simu_path, 'BfStruct.mat'))['BfStruct']

# BfStruct = BfStruct['BfStruct']
xExtent = float(BfStruct['x0']), float(BfStruct['x1']); # equivaut to [- (number of probe elements)/2 , +(number of probe elements)/2] * pitch
zExtent = float(BfStruct['z0']), float(BfStruct['z1']);
dz = float(BfStruct['dz']);
dx = float(BfStruct['dx']);
nframes = float(BfStruct['nframes']);
    
with h5py.File(load_path, 'r') as file:
    
    norm_IQ_mb_noised = file["norm_IQ_mb_noised"][:,:,:]

    # special research into all strings files to know if there is a sub-img or not in the loaded database
    list_key = list(file.keys())[:]
    search_text = 'sub_img_norm_IQ_mb_shuffled'
    found_string = None
    for key in list_key:
        if search_text in key:
            found_string = key
            subimages = file["sub_img_norm_IQ_mb_shuffled"][:,:,:]
            break

        else:
            subimages = 'None'
            # norm_IQ_mb_shuffled = file["norm_IQ_mb_shuffled"][:,:,:]
            
    coord_mb_shuffled = file["coord_mb_shuffled"][:,:]
    IQ_train = file["IQ_train"][:,:,:]
    coord_train = file["coord_train"][:,:]
    # xTrain = file["xTrain"][:,:,:]
    # yTrain = file["yTrain"][:,:]
    xTest = file["xTest"][:,:,:]
    yTest = file["yTest"][:,:]
    # xVal = file["xVal"][:,:,:]
    # yVal = file["yVal"][:,:]

        
#%% check gaussian noise
# plt.figure()      
# for ii in range(50):
#     check_data()
    # plt.imshow(norm_IQ_mb_noised_trans[ii,:,:], extent=[xExtent[0]-dx/2, xExtent[1]+dx/2, zExtent[1]+dz/2, zExtent[0]-dz/2])
#%% norm img
check_IQ_mb_noised = np.transpose(norm_IQ_mb_noised, axes=(2, 0, 1))
# check_coord_mb_shuffled = np.transpose(coord_mb_shuffled, axes=(1, 2, 1))
check_data(check_IQ_mb_noised[1,:,:], coord_mb_shuffled[1,:,:], [zExtent, xExtent], [dz, dx])

#%% Check IQ Train
for i in range(100):
    check_data(IQ_train[i,:,:], coord_train[i,:,:], [zExtent, xExtent], [dz, dx])
    
#%% check xTest,yTest

check_data(xTest[i,:,:], yTest[i,:,:], [zExtent, xExtent], [dz, dx])