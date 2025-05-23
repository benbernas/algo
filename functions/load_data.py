# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:55:23 2025

@author: Benoît Bernas

Load data function 

Avaible function: 
    - load_ultrasound_data
    - load_and_plot_histories_joblib

"""



import os
import scipy.io
import h5py
import numpy as np

def load_ultrasound_data(simu_path, load_path, load_optional=None):
    """
    Loads ultrasound imaging data and metadata from specified paths.
    
    Parameters:
    - simu_path (str): Path to 'BfStruct.mat'.
    - load_path (str): Path to the .h5 dataset file.
    - load_optional (list of str, optional): List of optional datasets to load, e.g., 
        ['xTrain', 'yTrain', 'sub_img_norm_IQ_mb_shuffled'].

    Returns:
    - data_dict (dict): Contains all loaded data and metadata.
    """
    
    os.makedirs(os.path.dirname(load_path), exist_ok=True)
    
    # Load BfStruct metadata
    BfStruct = scipy.io.loadmat(os.path.join(simu_path, 'BfStruct.mat'))['BfStruct']
    xExtent = float(BfStruct['x0']), float(BfStruct['x1'])
    zExtent = float(BfStruct['z0']), float(BfStruct['z1'])
    dx = float(BfStruct['dx'])
    dz = float(BfStruct['dz'])
    nframes = int(BfStruct['nframes'])

    data = {
        'xExtent': xExtent,
        'zExtent': zExtent,
        'dx': dx,
        'dz': dz,
        'nframes': nframes
    }

    # Load required and optional datasets from HDF5
    with h5py.File(load_path, 'r') as file:
        # Required datasets
        data['norm_IQ_mb_noised'] = file["norm_IQ_mb_noised"][:, :, :]
        data['coord_mb_shuffled'] = file["coord_mb_shuffled"][:, :]
        data['IQ_train'] = file["IQ_train"][:, :, :]
        data['coord_train'] = file["coord_train"][:, :]
        data['xTest'] = file["xTest"][:, :, :]
        data['yTest'] = file["yTest"][:, :]

        # Optional datasets
        if load_optional:
            for dataset_name in load_optional:
                if dataset_name in file:
                    data[dataset_name] = file[dataset_name][:]
                else:
                    data[dataset_name] = None  # not found

        # Special case: search for subimage key by substring
        if load_optional and any('sub_img_norm_IQ_mb_shuffled' in name for name in load_optional):
            for key in file.keys():
                if 'sub_img_norm_IQ_mb_shuffled' in key:
                    data['sub_img_norm_IQ_mb_shuffled'] = file[key][:, :, :]
                    break
            else:
                data['sub_img_norm_IQ_mb_shuffled'] = None

    return data


import joblib
import matplotlib.pyplot as plt

def load_and_plot_histories_joblib(filepath, max_epoch=None):
    """
    Charge et affiche les courbes de loss à partir d’un fichier .joblib contenant les historiques.

    Args:
        filepath (str): Chemin vers le fichier .joblib (ex: "model_param.joblib")
        max_epoch (int): Nombre d’epochs max à afficher (optionnel)
    """
    histories = joblib.load(filepath)

    n_folds = len(histories)
    if n_folds == 5:
        n_rows, n_cols = 2, 3
    elif n_folds == 2:
        n_rows, n_cols = 1, 2
    else:
        n_rows, n_cols = (n_folds + 2) // 3, 3

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 6))
    ax = ax.flatten()

    for i in range(n_folds):
        train_loss = histories[i].get('loss', [])
        val_loss = histories[i].get('val_loss', [])
        epochs = range(len(train_loss))
        if max_epoch:
            epochs = range(min(max_epoch, len(train_loss)))

        ax[i].plot(epochs, train_loss[:len(epochs)], label='Train Loss')
        ax[i].plot(epochs, val_loss[:len(epochs)], label='Val Loss')
        ax[i].set_title(f'Fold {i+1}')
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel('Loss')
        ax[i].legend()
        ax[i].grid(True)

    for j in range(n_folds, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()
    plt.show()
