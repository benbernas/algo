# -*- coding: utf-8 -*-
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

def global_rmse(true_coords, loc_coords):
# def global_rmse(true_coords, loc_coords, plot_fig, extent=None, resolution=None, num_bins_x = 11, num_bins_z = 10, c_lim_rmse=[0, 20], c_lim_num_mb=[0, 120], name_plot = None):
        
    # Calculate RMSE
    e_x = loc_coords[:, 0] - true_coords[:, 0]
    e_z = loc_coords[:, 1] - true_coords[:, 1]
    rmse = np.sqrt((e_x**2 + e_z**2)) * 1e3  # micro-meter
    
    # # Skip plotting if requested
    # if plot_fig == False:      
    #     return rmse, e_x, e_z
    
    
    # # Check extent and resolution if plotting is enabled
    # if plot_fig == True:
    #     if extent is None or resolution is None:
    #         raise ValueError("If plot_fig=True, 'extent' and 'resolution' must be provided as function inputs.")
    #     [x_extent, z_extent] = extent
    #     [dx, dz] = resolution
    
    #     x_bins = np.linspace(x_extent[0], x_extent[1], num_bins_x + 1)
    #     z_bins = np.linspace(z_extent[0], z_extent[1], num_bins_z + 1)
        
    #     region_num_mb = np.zeros((len(z_bins), len(x_bins)))
    
        # for i in range(len(x_bins)-1):
        #     for j in range(len(z_bins)-1):
        #         mask = (train_coords[:,0,0] >= x_bins[i]) & (train_coords[:,0,0]  < x_bins[i+1]) & (train_coords[:,1,0]  >= z_bins[j]) & (train_coords[:,1,0]  < z_bins[j+1])
        #         region_num_mb[j, i] = len(train_coords[mask,:,:])
                
        
        # fsize = 14
        # plt.figure(figsize=(8, 8))
        # # plt.subplot(2, 1, 2)
        # # density
        # dens = plt.imshow(region_num_mb, cmap='binary', extent=[x_extent[0]-dx/2, x_extent[1]+dx/2, z_extent[1]+dz/2, z_extent[0]-dz/2], alpha = 1, vmin =c_lim_num_mb[0], vmax = c_lim_num_mb[1])
        # plt.xlabel('x (mm)', fontsize=fsize)
        # plt.ylabel('z (mm)', fontsize=fsize)
        # cbar_dens = plt.colorbar(dens) 
        # cbar_dens.ax.tick_params(labelsize=fsize)
        # cbar_dens.set_label("Number of MB (training set)", fontsize=fsize)
        # plt.title(name_plot,fontsize=fsize)
        # # rmse 
        # sc1 = plt.scatter(loc_coords[:,0], loc_coords[:,1], c=np.array(rmse), cmap='jet', s=100, alpha=1, vmin =c_lim_rmse[0], vmax = c_lim_rmse[1])
        # plt.xlabel('x (mm)', fontsize=fsize)
        # plt.ylabel('z (mm)', fontsize=fsize)
        # cbar = plt.colorbar(sc1) 
        # cbar.ax.tick_params(labelsize=fsize)
        # cbar.set_label("RMSE (μm)", fontsize=fsize)
        # plt.axis('image')
        # plt.tight_layout()
        
        # return rmse, e_x, e_z
    return rmse, e_x, e_z
    
    
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