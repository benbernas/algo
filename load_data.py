# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:31:24 2024

@author: berna
"""
import pickle
import tensorflow as tf
from joblib import dump, load
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
load_data = 1
load_fig = 1

dir_path =  r'C:\Benoit\deep_learning\results'
folder_path = r'\test_simu_1bulle\cnn_prediction\IQ_mb_no_noise\adding_gaussian_noise_5_per_correction\size_conv_img\sizeConv2D_6_6\cross_val_nKFold_5\epoch_5000'
result_path = dir_path  + folder_path
#%% load figs
plt.close("all")
if load_fig == 1: 
    load_folder = result_path
    load_subfolder = r'\fig'
    file = open(load_folder + load_subfolder + r'\loss.mpl', 'rb')
    figure = pickle.load(file)
    figure.show()
    
    file = open(load_folder + load_subfolder + r'\accuracy.mpl', 'rb')
    figure = pickle.load(file)
    figure.show()
    
    file = open(load_folder + load_subfolder + r'\distribution_xcoord.mpl', 'rb')
    figure = pickle.load(file)
    figure.show()
    
    file = open(load_folder + load_subfolder + r'\distribution_zcoord.mpl', 'rb')
    figure = pickle.load(file)
    figure.show()
    
    file = open(load_folder + load_subfolder + r'\distance_xcoord_true_pred.mpl', 'rb')
    figure = pickle.load(file)
    figure.show()
    
    file = open(load_folder + load_subfolder + r'\distance_zcoord_true_pred.mpl', 'rb')
    figure = pickle.load(file)
    figure.show()
    
    file = open(load_folder + load_subfolder  + r'\abs_distance_xzcoord_true_pred.mpl', 'rb')
    figure = pickle.load(file)
    figure.show()
elif load_fig == 0:
    print('no need to load figures')
    
#%% load data

if load_data == 1:
    
    load_model = tf.keras.models.load_model(result_path + r'\model.keras')
    load_model.summary()
    param_model = load(result_path + r'\model_param.joblib')
    gpuState = param_model['runOnGPU']
    nKFold = param_model['nKFold']
    eval_result = load(result_path + r'\training_and_evaluation_results.joblib')
    elaps_time = eval_result['elapsed_time']
elif load_data == 0:
    print('no reload data')
# eval_result['cross_val_name']
# print(test_a)
# test_b = datatest['b']