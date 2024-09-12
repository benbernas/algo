# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:13:50 2024

@author: Benoît Bernas 
'my_module' gathers all the functions useful and used for deep learning algos

- extract_subimage_from_scatterer
--> function to extract a sub-image from an original size of a frames (img)

"""
import matplotlib.pyplot as plt
import numpy as np
import random

def normalize_IQ_mb(frames):
    # norm_IQ = np.empty(frames.shape)
    # for i in range(frames.shape[2]):
    im_IQ = abs(frames[:,:,:])
    max_IQ = np.max(im_IQ)
    # norm_IQ[:,:,i] =  im_IQ / max_IQ
    norm_IQ =  im_IQ / max_IQ
    return norm_IQ


def gaussian_noise(frames):
    # noise = np.empty(frames.shape)
    noised_img = np.empty(frames.shape)
    for i in range(frames.shape[2]):
        mu, sigma = 0, np.max(frames[:,:,i])*0.05
        noised_img[:,:,i] = np.random.normal(mu, sigma, frames[:,:,i].shape) 
        # noised_img[:,:,i] = noise      
    return noised_img, mu, sigma


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



def convert_mm_to_pixel(coord_mm, extent, resolution):
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

def check_data(image, coord, myextent, resolution):
    if myextent is None:
        myextent = None
    else:
        [z_extent, x_extent] = myextent
        [dz, dx] = resolution
    
    plt.figure()
    plt.imshow(image, extent=[x_extent[0]-dx/2, x_extent[1]+dx/2, z_extent[1]+dz/2, z_extent[0]-dz/2])
    plt.scatter(coord[:,0], coord[:,1], marker = '+', c='red')
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


# import tensorflow as tf

# Custom loss function
# def custom_loss(y_true, y_pred):
#     # Split y_true and y_pred into bounding box coordinates and class probabilities
#     y_true_coords, y_true_probs = tf.split(y_true, [4, num_classes], axis=-1)
#     y_pred_coords, y_pred_probs = tf.split(y_pred, [4, num_classes], axis=-1)
    
#     # Calculate mean squared error (MSE) for bounding box coordinates
#     mse_coords = tf.losses.mean_squared_error(y_true_coords, y_pred_coords)
    
#     # Calculate binary cross-entropy for class probabilities (if applicable)
#     if num_classes > 0:
#         bce_probs = tf.losses.binary_crossentropy(y_true_probs, y_pred_probs)
#         # You may need to adjust the weights for the two loss components to balance their contributions
#         total_loss = mse_coords + bce_probs
#     else:
#         total_loss = mse_coords
    
#     return total_loss
