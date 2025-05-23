# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:13:50 2024

@author: Benoît Bernas 
'my_module' gathers several functions useful and used for deep learning algos

- extract_subimage_from_scatterer
--> function to extract a sub-image from an original size of a frames (img)

main ones: 
- check_environment()
    
- add_gaussian_noise(frames, 
                     noise_level)

- normalize_IQ_mb(frames)


- extract_subimage_from_scatterer(
    image, 
    coord_px, 
    new_size_px, 
    offset=None)


optional ones: 
- calculate_new_extent(start_index, 
                       end_index, 
                       extent, 
                       resolution)

- convert_mm_to_pixel(coord_mm, 
                     nMB, 
                     extent, 
                     resolution)

- calculate_distances_from_coords(coords)

- generate_distance_labels(coord_labels)

"""
import matplotlib.pyplot as plt
import numpy as np
import random

import tensorflow as tf

import sys
import os


def check_environment():
    # Check Python version
    python_version = sys.version_info
    is_python_310 = python_version.major == 3 and python_version.minor == 10

    # Check if the environment is 'myenv'
    # This assumes that 'myenv' is part of the PATH environment variable
    is_myenv = 'myenv' in sys.prefix

    # Check if GPU is available
    num_gpus = len(tf.config.list_physical_devices('GPU'))

    if is_python_310 and is_myenv:
        if num_gpus > 0:
            python = sys.version[:7]
            tensorFlow = tf.__version__
            cuda = 11.2
            cuDNN = 8.1
            run_GPU = 1
            print(f"Running on Python {python} in 'myenv' Anaconda environment with GPU support.")
            print(f"TensorFlow version: {tensorFlow}")
            print(f"CUDA version: {cuda}")
            print(f"cuDNN version: {cuDNN}")
            print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
            

        else:
            python = 3.10
            tensorFlow = tf.__version__
            cuda = 'not running on GPU'
            cuDNN = cuda
            run_GPU = 0
            print(f"Running on Python {python} in 'myenv' Anaconda environment without GPU support.")
            print(f"TensorFlow version: {tensorFlow}")
            print(f"CUDA version: {cuda}")
            print(f"cuDNN version: {cuDNN}")
            print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
            

            
    else:
        python = sys.version[:7]
        tensorFlow = tf.__version__
        cuda = 'not running on GPU'
        cuDNN = cuda
        run_GPU = 0
        print(f"Running on Python {python}, not in 'myenv' Anaconda environment. Not using the GPU.")
        print(f"TensorFlow version: {tensorFlow}")
        print(f"CUDA version: {cuda}")
        print(f"cuDNN version: {cuDNN}")
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        
    
        
    return run_GPU, python, tensorFlow, cuda, cuDNN 

def add_gaussian_noise(frames, noise_level):
    # Créer un tableau vide pour les images bruitées, avec le même type que les frames
    #noise level --> pourcentage du max d'intensité de la frame --> exemple 5% --> mettre 0.05 
    
    # noise_level in percentage; example noise_level = 0.05  # 
    noised_img = np.empty(frames.shape, dtype=frames.dtype)

    # Génération du bruit pour chaque image
    for i in range(frames.shape[2]):
        # Trouver le maximum absolu de l'image actuelle
        max_value = np.max(np.abs(frames[:, :, i])) * np.random.uniform(0, noise_level)
        
        # Générer le bruit gaussien complexe
        noise_real = np.random.normal(0, 1, frames[:, :, i].shape)
        noise_imag = np.random.normal(0, 1, frames[:, :, i].shape)
        complex_noise = noise_real + 1j * noise_imag  # Bruit gaussien complexe non limité
        
        
        # Normaliser le bruit pour qu'il ne dépasse pas 5% du maximum de l'image
        noise = complex_noise / np.max(np.abs(complex_noise)) * (max_value)
        
        # Ajouter le bruit à l'image brute
        noised_img[:, :, i] = frames[:, :, i] + noise
        

    return noised_img


def normalize_IQ_mb(frames):
    # norm_IQ = np.empty(frames.shape)
    # for i in range(frames.shape[2]):
    im_IQ = abs(frames[:,:,:])
    max_IQ = np.max(im_IQ)
    # norm_IQ[:,:,i] =  im_IQ / max_IQ
    norm_IQ =  im_IQ / max_IQ
    return norm_IQ



def extract_subimage_from_scatterer(image, coord_px, new_size_px, offset=None):
    """
    Extraire une sous-image centrée autour d'une coordonnée pixel tout en retournant les indices.
    
    :param image: Image originale (numpy array).
    :param center_px: Coordonnée centrale en pixels (row, col).
    :param new_size_px: Taille de la sous-image en pixels (height, width).
    :param offset: Décalage pour éviter de centrer le scatterer (tuple: (offset_z, offset_x)).
    :return: Sous-image de taille (new_size_px, new_size_px) et indices de la sous-image.
    """
    center_z, center_x = coord_px
    half_height = new_size_px[0] // 2
    half_width = new_size_px[1] // 2
    
    
    
    # Décalage aléatoire ou spécifié pour éviter de centrer le scatterer
    if offset is None:
        offset_z = random.randint(-half_height // 2, half_height // 2)
        offset_x = random.randint(-half_width // 2, half_width // 2)
    else:
        offset_z, offset_x = offset
    
    # Calcul des limites de la sous-image avec le décalage appliqué
    start_z = max(0, center_z - half_height + offset_z)
    end_z = min(image.shape[0], start_z + new_size_px[0])
    start_x = max(0, center_x - half_width + offset_x)
    end_x = min(image.shape[1], start_x + new_size_px[1])
    
    # Ajustement si les dimensions dépassent
    if end_z - start_z < new_size_px[0]:
        start_z = end_z - new_size_px[0]
    if end_x - start_x < new_size_px[1]:
        start_x = end_x - new_size_px[1]
    
    # Assurer que les indices ne soient pas négatifs
    start_z = max(0, start_z)
    start_x = max(0, start_x)
    
    # Extraire la sous-image
    subimage = image[start_z:end_z, start_x:end_x]
    
    return subimage, (start_z, end_z), (start_x, end_x)

def calculate_new_extent(start_index, end_index, extent, resolution):
    """
    Calculer les nouvelles étendues en mm pour la sous-image extraite.
    
    :param start_index: Indice de départ (z_start, x_start).
    :param end_index: Indice de fin (z_end, x_end).
    :param extent: Étendue des axes en mm [z_extent, x_extent] = [(z_extent_min, z_extent_max), (x_extent_min, x_extent_max)].
    :param resolution: Résolution en mm par pixel (dz, dx).
    :return: Nouvelles étendues en mm (z_extent, x_extent).
    """
    z_start, z_end = start_index
    x_start, x_end = end_index
    
    new_z_extent = (extent[0][0] + z_start * resolution[0], extent[0][0] + z_end * resolution[0])
    new_x_extent = (extent[1][0] + x_start * resolution[1], extent[1][0] + x_end * resolution[1])
    
    return new_z_extent, new_x_extent



def convert_mm_to_pixel(coord_mm, nMB, extent, resolution):
    """
    Convertir des coordonnées en mm en pixels.
    
    :param coord_mm: Coordonnée en mm (z, x).
    :param extent: Étendue des axes en mm [z_extent, x_extent] 
                                        = [(z_extent_min, z_extent_max), (x_extent_min, x_extent_max)].
    :param resolution: Résolution en mm par pixel (dz, dx).
    :return: Coordonnée en pixels (row, col).
    """
    
    
    z_mm, x_mm = coord_mm
    
    # Calcul de la conversion
    x_px = int(round((x_mm - extent[1][0]) / resolution[1]))
    z_px = int(round((z_mm - extent[0][0]) / resolution[0]))
    
    return (z_px, x_px)



def plot_data(image, coord, nMB, myextent, resolution):
    if myextent is None:
        myextent = None
    else:
        [z_extent, x_extent] = myextent
        [dz, dx] = resolution
    
    if nMB > 1 :
        x_coord = coord[0, nMB]
        z_coord = coord[1, nMB]
    elif nMB == 1 :
        x_coord = coord[0]
        z_coord = coord[1]
    
    plt.imshow(image, extent=[x_extent[0]-dx/2, x_extent[1]+dx/2, z_extent[1]+dz/2, z_extent[0]-dz/2])
    plt.scatter(x_coord, z_coord, marker = '+', c='red')
    plt.pause(0.1)
    plt.show()

    
from itertools import combinations

# Fonction pour calculer les distances entre les points
def calculate_distances_from_coords(coords):
    nMB = coords.shape[0]  # Nombre de points
    distances = []
    
    for (i, j) in combinations(range(nMB), 2):
        x_i, z_i = coords[i]
        x_j, z_j = coords[j]
        distance = np.sqrt((x_j - x_i)**2 + (z_j - z_i)**2)
        distances.append(distance)
    
    # Retourne les distances sous forme d'un tableau
    return np.array(distances)

# Fonction pour générer les labels de distances
def generate_distance_labels(coord_labels):
    distance_labels = []
    
    for coords in coord_labels:
        distances = calculate_distances_from_coords(coords)
        distances_sorted = np.sort(distances)  # Trier par ordre croissant
        distance_labels.append(distances_sorted)
    
    return np.array(distance_labels)


