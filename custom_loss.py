# -*- coding: utf-8 -*-
"""
Created on Fri May 16 17:50:01 2025

@author: Benoît Bernas


Function used for custom loss function. 
"""


import numpy as np
import itertools
import tensorflow as tf

# ---------- paramètres modifiables ----------
_zExtent  = (-np.inf, np.inf)   # bornes par défaut (aucune contrainte)
_xExtent  = (-np.inf, np.inf)
_nMB      = 5                   # nombre de micro‑bulles par défaut
# -------------------------------------------

def set_matching_params(zExtent, xExtent, nMB):
    """Fonction à appeler dans le script principal pour configurer les limites et nMB."""
    global _zExtent, _xExtent, _nMB
    _zExtent = zExtent
    _xExtent = xExtent
    _nMB     = nMB

# -------- Matching function --------
def matching_mb(x_true, z_true, x_pred, z_pred):
    z_pred = np.clip(z_pred, _zExtent[0], _zExtent[1])
    x_pred = np.clip(x_pred, round(_xExtent[0], 3), round(_xExtent[1], 3))
    # num_MB = x_true.shape[0]

    x_true_exp = np.expand_dims(x_true, axis=1)
    x_pred_exp = np.expand_dims(x_pred, axis=0)
    z_true_exp = np.expand_dims(z_true, axis=1)
    z_pred_exp = np.expand_dims(z_pred, axis=0)

    distance_matrix = np.sqrt(np.square(x_true_exp - x_pred_exp) + np.square(z_true_exp - z_pred_exp))

    best_permutation = None
    min_total_distance = np.inf
    for perm in itertools.permutations(range(_nMB)):
        total_distance = sum(distance_matrix[i, perm[i]] for i in range(_nMB))
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_permutation = perm

    return np.array(best_permutation).astype(np.int32)

# -------- Custom loss --------
def optimal_matching_loss(y_true, y_pred):
    batch_size = tf.shape(y_true)[0]
    loss = 0.0

    for ibatch in range(batch_size):
        x_true, z_true = y_true[ibatch, :_nMB], y_true[ibatch, _nMB:]
        x_pred, z_pred = y_pred[ibatch, :_nMB], y_pred[ibatch, _nMB:]

        best_perm = tf.py_function(matching_mb, [x_true, z_true, x_pred, z_pred], Tout=tf.int32)

        x_pred_matched = [x_pred[best_perm[i]] for i in range(_nMB)]
        z_pred_matched = [z_pred[best_perm[i]] for i in range(_nMB)]

        distance = tf.sqrt(tf.square(x_pred_matched - x_true) + tf.square(z_pred_matched - z_true))
        # loss += tf.reduce_mean(distance)
        # loss += distance

    # return loss / tf.cast(batch_size, tf.float32)
    return distance


"""
Hugarian method to match MBs



"""
from scipy.optimize import linear_sum_assignment
def hungarian_algorithm(cost_matrix):
    """ Applique l'algorithme Hugarian sur la matrice de coût """
    cost_matrix_np = cost_matrix.numpy()  # Conversion en NumPy
    row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
    return col_ind.astype(np.int32)  # Retourne les indices de correspondance

def euclidean_distance_loss_hungarian(y_true, y_pred):
    """
    Custom loss function using the Hungarian algorithm to optimally match predicted points to ground truth.
    """
    batch_size = tf.shape(y_true)[0]  # Nombre d'échantillons dans le batch
    loss = 0.0


    for i in range(batch_size):
        # Séparer les coordonnées x et z pour y_true et y_pred
        x_true, z_true = y_true[i, :_nMB], y_true[i, _nMB:]
        x_pred, z_pred = y_pred[i, :_nMB], y_pred[i, _nMB:]

        
        # Appliquer des contraintes sur les coordonnées prédictes des points = limites étant les bords de l'image
        z_pred = tf.maximum(z_pred, _zExtent[0])  #  si z_pred < 0, alors z_pred = 0
        x_pred = tf.maximum(x_pred, round(_xExtent[0],3)) # si x_pred < -6.985 alors x_pred = -6.985
        x_pred = tf.minimum(x_pred, round(_xExtent[1],3)) # si x_pred > 6.985 --> x_pred = 6.985
        z_pred = tf.minimum(z_pred, _zExtent[1]) # si z_pred > 6.985 --> x_pred = 6.985
        
        

        # Construire la matrice de coût (distance euclidienne entre chaque point vrai et prédit)
        x_true_exp = tf.expand_dims(x_true, axis=1)  # (5,1)
        x_pred_exp = tf.expand_dims(x_pred, axis=0)  # (1,5)
        z_true_exp = tf.expand_dims(z_true, axis=1)
        z_pred_exp = tf.expand_dims(z_pred, axis=0)

        cost_matrix = tf.sqrt(tf.square(x_true_exp - x_pred_exp) + tf.square(z_true_exp - z_pred_exp))

        # Exécuter le calcul Hongrie en mode non différentiable
        col_ind = tf.py_function(hungarian_algorithm, [cost_matrix], Tout=tf.int32)

        # Appliquer la correspondance optimale aux prédictions
        x_pred_matched = tf.gather(x_pred, col_ind)
        z_pred_matched = tf.gather(z_pred, col_ind)

        # Calcul de la distance euclidienne après correspondance
        distance = tf.sqrt(tf.square(x_pred_matched - x_true) + tf.square(z_pred_matched - z_true))

        # Ajouter la perte moyenne pour cet élément du batch
        loss += tf.reduce_mean(distance)

    # Retourner la perte moyenne sur tout le batch
    return loss / tf.cast(batch_size, tf.float32)


