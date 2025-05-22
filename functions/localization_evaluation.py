# -*- coding: utf-8 -*-
"""
Created on Fri May 16 17:20:45 2025

@author: Benoît Bernas

Performance evaluation of MB localization compute by the Deep Learning model
"""

"""
Created on Wed Nov 20 15:11:44 2024

@author: Benoît Bernas 

Compute and plot RMSE 

OUTPUT 
- rmse (array) : Root Mean Square Error (RMSE) in micrometer between 
relative error along x-axis 'e_x' and relative error along z-axis 'e_z'


INPUTS                
- true_coords (array): [x,z] coordinates of the true MB position of the test set (test images)
                ndim = 3, 
                shape = [len(trainImages), 2, 1]
                
- loc_coords (array): [x,z] coordinates of the MB localization
                ndim = 2, 
                shape = [len(trainImages), 2]

OPTIONS
- plot_fig (bool): True or False
            --> plot RMSE for the entire test set 
            overlaid on the local number of MBs in the training set of the CNN model.

"""
import matplotlib.pyplot as plt
import numpy as np

from custom_loss import matching_mb

def global_rmse(true_coords, loc_coords, nMB):
    results = {}
    
    if nMB == 1:    
        # Calculate RMSE
        e_x = loc_coords[:, 0] - true_coords[:, 0]
        e_z = loc_coords[:, 1] - true_coords[:, 1]
        rmse_mb = np.sqrt((e_x**2 + e_z**2)) * 1e3  # micro-meter
    
    elif nMB > 1:
        x_pred_test, z_pred_test = loc_coords[:,:5], loc_coords[:,5:] 
        x_true_test, z_true_test = true_coords[:,0,:],  true_coords[:,1,:]
        
        best_perm_test = []
        x_pred_test_matched = []
        z_pred_test_matched = []

        for iframe in range(x_pred_test.shape[0]):    
            best_perm = matching_mb(x_true_test[iframe,:], z_true_test[iframe,:], x_pred_test[iframe,:], z_pred_test[iframe,:])
            x_pred_test_matchs = [x_pred_test[iframe, best_perm[i]] for i in range(nMB)]
            z_pred_test_matchs = [z_pred_test[iframe, best_perm[i]] for i in range(nMB)]
                # x_true_matched = [x_true[best_perm[i]] for i in range(num_mb)]
                # z_true_matched = [z_true[best_perm[i]] for i in range(num_mb)]
            
            x_pred_test_matched.append(x_pred_test_matchs)
            z_pred_test_matched.append(z_pred_test_matchs)
            
            best_perm_test.append(best_perm)
            
        # test = np.array(z_true_test_matchs)
        # test_perm = np.array(best_perm_test)

        # e_x = 

        # #% evaluation
        ex_mb  = np.array(x_pred_test_matched) - x_true_test # relative error along x for each MB (mm)
        ez_mb  = np.array(z_pred_test_matched) - z_true_test # relative error along z for each MB (mm)

        rmse_mb = np.sqrt((ex_mb**2 + ez_mb**2)) * 1e3  # rmse along [x,z] for each MB (um)
        
        # Stockage structuré
        results["rmse_mb"] = rmse_mb
        results["ex_mb"] = ex_mb
        results["ez_mb"] = ez_mb
        results["rmse_tot_mean"] = np.mean(rmse_mb) # mean rmse over all MBs
        results["rmse_tot_std"] = np.std(rmse_mb) # std rmse over all MBs
      
        results["rmse_mb_mean"] = [np.mean(rmse_mb[:, i]) for i in range(nMB)] # mean rmse for each MBs
        results["rmse_mb_std"] = [np.std(rmse_mb[:, i]) for i in range(nMB)] # std rmse for each MBs
      
        return results
    

"""
Created on Wed Nov 20 15:11:44 2024

@author: benoît bernas 

Mean RMSE over 1 physical parameter (correlation, depth, ...)

- rmse (array): global RMSE computed with the function above
        ndim = 1

- param (array): physical parameter you want to observe
        ndim = 1

- bins_range (array): range of the observation
        ndim = 1

- num_bins (int): number of intervals in the range
        ndim = 1
"""
   
def rmse_over_param(rmse, param, bins_range, num_bins): 
        
    value = param.flatten()
    # num_bins_corr = num_bins
    bins = np.linspace(bins_range[0], bins_range[1], num_bins + 1)
    step = abs((bins[1]-bins[0]))
    
    # Dictionnaires pour stocker les indices pour chaque intervalle de x et z
    indices = {f"{ind} to {ind+step} mm": [] for ind in bins}
    
    all_mean_rmse = []
    
    for idx, val in enumerate(value):
        for start in bins:
            if start <=  val < start + step:
                indices[f"{start} to {start + step} mm"].append(idx)
                break
        
    for ii in range(len(indices)):
        list_inter = list(indices.keys())[ii]
        interval_indices = indices[list_inter]
        
        # CNN model
        mean_rmse_inter = np.mean(rmse[interval_indices])
        all_mean_rmse.append(mean_rmse_inter)
        
    return np.array(all_mean_rmse), bins, step


    
#%% 

def plot_rmse_and_density_map(loc_coords, rmse, train_coords, nMB, extent, resolution, 
                               num_bins_x=11, num_bins_z=10, 
                               c_lim_rmse=[0, 100], c_lim_num_mb=[0, 120], 
                               name_plot=None):
    """
    Trace la densité des MBs et superpose le RMSE pour chaque localisation prédite.
    Supporte plusieurs MBs par image.

    Args:
        loc_coords (np.ndarray): Coordonnées prédites (N, 10) avec 5 x puis 5 z
        rmse (np.ndarray): Valeur RMSE par MB (N, 5)
        train_coords (np.ndarray): Coordonnées de training (N, 2, 5)
        extent (list): [[xmin, xmax], [zmin, zmax]]
        resolution (list): [dx, dz]
        num_bins_x (int): Nombre de bins sur l’axe x
        num_bins_z (int): Nombre de bins sur l’axe z
        c_lim_rmse (list): Limites de couleur pour RMSE
        c_lim_num_mb (list): Limites de couleur pour la densité
        name_plot (str): Titre de la figure
    """
    x_extent, z_extent = extent
    dx, dz = resolution

    x_bins = np.linspace(x_extent[0], x_extent[1], num_bins_x + 1)
    z_bins = np.linspace(z_extent[0], z_extent[1], num_bins_z + 1)
    
    region_num_mb = np.zeros((len(z_bins)-1, len(x_bins)-1))

    # === Densité basée sur le premier MB de train_coords ===
    for i in range(len(x_bins) - 1):
        for j in range(len(z_bins) - 1):
            mask = (train_coords[:,:nMB] >= x_bins[i]) & (train_coords[:,:nMB] < x_bins[i+1]) & \
                   (train_coords[:,nMB:] >= z_bins[j]) & (train_coords[:,nMB:] < z_bins[j+1])
            region_num_mb[j, i] = np.sum(mask)

    # === Plot densité ===
    plt.figure(figsize=(8, 8))
    dens = plt.imshow(region_num_mb, cmap='binary',
                      extent=[x_extent[0]-dx/2, x_extent[1]+dx/2, 
                              z_extent[1]+dz/2, z_extent[0]-dz/2],
                      alpha=1, vmin=c_lim_num_mb[0], vmax=c_lim_num_mb[1])

    plt.xlabel('x (mm)', fontsize=14)
    plt.ylabel('z (mm)', fontsize=14)
    cbar_dens = plt.colorbar(dens)
    cbar_dens.set_label("Number of MB (training set)", fontsize=14)

    # === Extraire x, z, rmse sous forme plate ===
    x_coords = loc_coords[:, :5].flatten()
    z_coords = loc_coords[:, 5:].flatten()
    rmse_vals = rmse.flatten()
    np.max(rmse_vals)

    # === Plot RMSE superposé ===
    sc1 = plt.scatter(x_coords, z_coords, c=rmse_vals, cmap='jet', 
                      s=100, alpha=1, vmin=c_lim_rmse[0], vmax=np.max(rmse_vals))
    cbar = plt.colorbar(sc1)
    cbar.set_label("RMSE (μm)", fontsize=14)

    plt.title(name_plot if name_plot else "RMSE vs MB Density", fontsize=14)
    plt.axis('image')
    plt.tight_layout()
    plt.show()

