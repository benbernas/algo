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


import sys
import os

import scipy.io
import h5py



load_data = 1
load_fig = 0
check_filter = 0 
dir_path =  r'D:\Benoit\machine_learning\python\deep_learning\results'
folder_path = r'\test_simu_1bulle\cnn_prediction\IQ_mb_no_noise\adding_gaussian_noise_5_per_correction\nblocks_20\sizeConv2D_7_7\cross_val_KFold_5\epoch_5000'
result_path = dir_path  + folder_path
#%% load figs
plt.close("all")
if load_fig == 0: 
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
    
    load_train = load(result_path + r'\training_dataset.joblib')
elif load_data == 0:
    print('no reload data')
# eval_result['cross_val_name']
# print(test_a)
# test_b = datatest['b']

#%% check conv layer
if check_filter == 1:
    
    conv_layer = load_model.layers[0]  # Changez l'indice si nécessaire pour choisir une autre couche Conv2D
    filters, biases = conv_layer.get_weights()
    
    # Les filtres ont la forme (height, width, input_channels, num_filters)
    print(f"Shape of the filters: {filters.shape}")  # Exemple : (3, 3, 1, 16) pour 16 filtres 3x3 avec 1 canal d'entrée
    
    # # Normaliser les filtres pour les rendre visibles (mettre dans l'intervalle 0-1)
    # filters_min, filters_max = filters.min(), filters.max()
    # normalized_filters = (filters - filters_min) / (filters_max - filters_min)
    
    
    # Nombre de filtres dans la couche
    num_filters = filters.shape[3]  # Exemple : 16 filtres
    
    # Affichage des filtres
    n_rows = 4  # Nombre de lignes pour l'affichage
    n_columns = num_filters // n_rows  # Nombre de colonnes nécessaires
    
    plt.figure(figsize=(10, 10))
    
    for i in range(num_filters):
        ax = plt.subplot(n_rows, n_columns, i + 1)
        
        # Filtre i
        filter_i = filters[:, :, :, i]  # Filtre i
        
        if filter_i.shape[-1] == 1:  # Si le filtre est en 2D (canal unique), on affiche en gris
            plt.imshow(filter_i[:, :, 0])
        else:  # Si le filtre a plusieurs canaux, on les combine en image RGB
            plt.imshow(filter_i[:,:,0])
        
        ax.set_title(f"Filter {i+1}", fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
elif check_filter == 0:
    ('no check filter')


