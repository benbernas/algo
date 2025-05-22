# -*- coding: utf-8 -*-
"""
Created on Fri May 16 16:52:08 2025

@author: Benoît Bernas

"""

"""
Visualizes an IQ image depending on the display type: raw, mean, or Doppler.

Available functions:
    
   - check_distribution
   - plot_data_and_loc

Args:
    raw_iq: raw IQ data (2D or 3D array)
    type_im: type of image to show - 'raw', 'mean', or 'pwd' (power Doppler)
    nMB: number of microbubbles (used to parse coords)
    extent_im: [x_extent, z_extent] in mm (image bounds)
    resolution: [dx, dz] in mm/pixel (image resolution)
    coords: optional real coordinates [x, z] (shape varies)
"""


import matplotlib.pyplot as plt
import numpy as np

def check_iq(raw_iq, type_im, nMB, extent_im, resolution, coords=None):
            
    
    [x_extent, z_extent] = extent_im
    [dx, dz] = resolution
    
    # Extract x and z coordinates based on input shape and number of MBs
    if coords is not None and coords.ndim == 3 and coords.shape[2] == 1:
    # Cas 1 MB : coords (N, 2, 1)
        x_coords, z_coords = coords[:, 0, 0], coords[:, 1, 0]
    elif coords is not None and nMB > 1:
    # Cas multi-MBs : coords (N, 2*nMB)
        x_coords, z_coords = coords[:, :nMB], coords[:, nMB:]
    else:
        x_coords, z_coords = None, None

       
    fsize = 20
    # Choose the image to display
    if type_im == 'raw':
        im = raw_iq
        title = 'raw IQ'
        if im.shape[0] > 1:
            print('dim(raw_iq) has to be = 1 to ')
        
    
    elif type_im == 'mean':
        im = np.mean(raw_iq, axis = 0)
        title = 'mean IQ'
        
    elif type_im == 'pwd':
        dop = np.mean(abs(raw_iq)**2, axis = 0)
        im = 20*np.log10(dop/np.max(dop))
        title = 'Power Doppler (dB)'
        

    # Choose the image to display    
    plt.figure(figsize = (15,8))
    pwd = plt.imshow(im, cmap='viridis', extent=(x_extent[0]-dx/2, x_extent[1]+dx/2, z_extent[1]+dz/2, z_extent[0]-dz/2))
    plt.xlabel('x (mm)', fontsize=fsize)
    plt.ylabel('z (mm)', fontsize=fsize)
    plt.scatter(x_coords, z_coords, s = 100, marker = '+', c = 'red', alpha = 0.5, label = 'True coordinates', linewidths = 1.5, edgecolors = 'face')
    plt.tick_params(axis='both', which='major', labelsize=14)
    cbar = plt.colorbar(pwd) 
    cbar.ax.tick_params(labelsize=fsize)
    cbar.set_label(title, fontsize=fsize)
    plt.show()            

"""
Displays a sequence of IQ images frame-by-frame with optional coordinate overlay.

Args:
    nframe: number of frames to display
    iq_img: 3D IQ image sequence [N, H, W]
    nMB: number of microbubbles
    extent_im: [x_extent, z_extent]
    resolution: [dx, dz]
    coords: ground truth coordinates (shape varies)
    show_pred: (optional) enable predicted coordinate overlay
"""

def check_iq_movie(nframe, iq_img, nMB, extent_im, resolution, coords=None, show_pred=None):
    
        
    [xExtent, zExtent] = extent_im
    [dx, dz] = resolution
    
    # Extract x and z coordinates
    if coords is not None and coords.ndim == 3 and coords.shape[2] == 1:
    # Cas 1 MB : coords (N, 2, 1)
        x_coords, z_coords = coords[:, 0, 0], coords[:, 1, 0]
    elif coords is not None and nMB > 1:
    # Cas multi-MBs : coords (N, 2*nMB)
        x_coords, z_coords = coords[:, :nMB], coords[:, nMB:]
    else:
        x_coords, z_coords = None, None
      
    fsize = 20
    
    
    plt.figure(figsize = (15,8))
    for iframe in range(nframe):
        plt.imshow(iq_img[iframe,:,:], extent=[xExtent[0]-dx/2, xExtent[1]+dx/2, zExtent[1]+dz/2, zExtent[0]-dz/2])
        # plt.imshow(xTrain[ii,:,:])
        
        plt.scatter(x_coords[iframe], z_coords[iframe], marker = '+', c='red')
        # if show_pred != None:    
        #     plt.scatter(y_pred_coord[iframe,:5], y_pred_coord[iframe,5:], marker = '+', c='green')
        plt.xlabel('x (mm)', fontsize=fsize)
        plt.ylabel('z (mm)', fontsize=fsize)
        plt.pause(0.5)
        plt.clf()
        plt.show()

"""
Displays descriptive statistics and several plots for a given value array (e.g., RMSE).

Args:
    param: numeric array of values (Ex: array of RMSE values)
    nbins: number of bins for histogram

Returns:
    mean, median, std, min, and max of the input array
"""        


def check_distribution(param, nbins):
    

    # Statistics
    mean_value = np.mean(param)
    median_value = np.median(param)
    std_value = np.std(param)
    min_value = np.min(param)
    max_value = np.max(param)

    print(f"Moyenne: {mean_value}")
    print(f"Médiane: {median_value}")
    print(f"Écart-type: {std_value}")
    print(f"Min: {min_value}")
    print(f"Max: {max_value}")

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(param, bins=nbins, color='skyblue', edgecolor='black')
    plt.title('Histogramme des valeurs RMSE')
    plt.xlabel('RMSE')
    plt.ylabel('Fréquence')
    plt.grid(True)
    plt.show()

    # Boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(param, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title('Boxplot des valeurs RMSE')
    plt.xlabel('RMSE')
    plt.grid(True)
    plt.show()

    # # Line plot 
    plt.figure(figsize=(10, 6))
    plt.plot(param, marker='o', linestyle='-', color='blue', label='RMSE')
    plt.title('Progression des valeurs RMSE')
    plt.xlabel('Index')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return mean_value, median_value, std_value, min_value, max_value


"""
Multi-purpose visualization function to inspect results (train, val, test) and Doppler image.

Args:
    Includes image arrays, coordinates, resolution, display preferences, and frames to inspect.
"""

def plot_data_and_loc(
    iq_shuffled=None, coord_mb_shuffled = None,  
    xTrain=None, yTrain=None, 
    xTest=None, yTest=None, y_pred_coord=None, 
    xVal=None, yVal=None, 
    xExtent=None, zExtent=None, dx=None, dz=None, 
    display_modes=("check_pwd", "train_several_frames", "train_movie_frames", "val_several_frames", "val_movie_frames", "test_several_frames", "test_movie_frames"),
    frame_wanted=[0, 26, 72, 134],
    pause_duration=0.5,
    n_frames=200
):
    def get_extent():
        return [xExtent[0]-dx/2, xExtent[1]+dx/2, zExtent[1]+dz/2, zExtent[0]-dz/2]
    
    # Display specific training frames
    if "train_several_frames" in display_modes and xTrain is not None and yTrain is not None:
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for i, idx in enumerate(frame_wanted):
            row, col = divmod(i, 2)
            ax[row, col].imshow(xTrain[idx], extent=get_extent())
            ax[row, col].scatter(yTrain[idx, :5], yTrain[idx, 5:], marker='+', c='red', label='True')
            # ax[row, col].scatter(y_pred_coord[idx, :5], y_pred_coord[idx, 5:], marker='+', c='green', label='Pred')
            ax[row, col].set_title(f'Test frame {idx}')
            if i == 0:
                ax[row, col].legend()
        plt.tight_layout()
        plt.show()
        
        
    # Animate training frames    
    if "train_movie_frames" in display_modes and xTrain is not None and yTrain is not None:
        plt.figure(figsize = (15,8))
        for ii in range(min(n_frames, len(xTrain))):
            plt.imshow(xTrain[ii], extent=get_extent())
            plt.scatter(yTrain[ii, :5], yTrain[ii, 5:], marker='+', c='red')
            plt.title(f"Training frame {ii}")
            plt.pause(pause_duration)
            plt.clf()
        plt.close()
        
    # Display validation frames
    if "val_several_frames" in display_modes and xVal is not None and yVal is not None:
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for i, idx in enumerate(frame_wanted):
            row, col = divmod(i, 2)
            ax[row, col].imshow(xVal[idx], extent=get_extent())
            ax[row, col].scatter(yVal[idx, :5], yVal[idx, 5:], marker='+', c='red', label='True')
            ax[row, col].set_title(f'Validation frame {idx}')
            if i == 0:
                ax[row, col].legend()
        plt.tight_layout()
        plt.show()
    # Animate validation frames    
    if "val_movie_frames" in display_modes and xVal is not None and yVal is not None:
        plt.figure(figsize = (15,8))
        for ii in range(min(n_frames, len(xVal))):
            plt.imshow(xVal[ii], extent=get_extent())
            plt.scatter(yVal[ii, :5], yVal[ii, 5:], marker='+', c='red')
            # plt.scatter(y_pred_coord[ii, :5], y_pred_coord[ii, 5:], marker='+', c='green')
            plt.title(f"Train pred frame {ii}")
            plt.pause(pause_duration)
            plt.clf()
        plt.close()
        
       
    # Display test frames with predictions    
        #add of predicted coordinates (y_pred_coord) for the test set plot
    if "test_several_frames" in display_modes and xTest is not None and yTest is not None and y_pred_coord is not None:
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for i, idx in enumerate(frame_wanted):
            row, col = divmod(i, 2)
            ax[row, col].imshow(xTest[idx], extent=get_extent())
            ax[row, col].scatter(yTest[idx, 0, :], yTest[idx, 1, :], marker='+', c='red', label='True')
            ax[row, col].scatter(y_pred_coord[idx, :5], y_pred_coord[idx, 5:], marker='+', c='green', label='Pred')
            ax[row, col].set_title(f'Test frame {idx}')
            if i == 0:
                ax[row, col].legend()
        plt.tight_layout()
        plt.show()
    
    # Animate test frames with predictions
    if "test_movie_frames" in display_modes and xTest is not None and yTest is not None and y_pred_coord is not None:
        plt.figure(figsize = (15,8))
        for ii in range(min(n_frames, len(xTest))):
            plt.imshow(xTest[ii], extent=get_extent())
            plt.scatter(yTest[ii, 0, :], yTest[ii, 1, :], marker='+', c='red')
            plt.scatter(y_pred_coord[ii, :5], y_pred_coord[ii, 5:], marker='+', c='green')
            plt.title(f"Train pred frame {ii}")
            plt.pause(pause_duration)
            plt.clf()
        plt.close()
    
    # Display Power Doppler image
    if "check_pwd" in display_modes and iq_shuffled is not None and coord_mb_shuffled is not None :
        dop = np.mean(abs(iq_shuffled)**2, axis = 0)
        im = 20*np.log10(dop/np.max(dop))
        title = 'Power Doppler (dB)'
        
        plt.figure(figsize = (15,8))
        pwd = plt.imshow(im, cmap='viridis', extent=get_extent())
        plt.xlabel('x (mm)')
        plt.ylabel('z (mm)')
        plt.scatter(coord_mb_shuffled[:, 0, :], coord_mb_shuffled[:, 1, :], s = 100, marker = '+', c = 'red', alpha = 0.5, label = 'True coordinates', linewidths = 1.5, edgecolors = 'face')
        plt.tick_params(axis='both', which='major', labelsize=14)
        cbar = plt.colorbar(pwd) 
        # cbar.ax.tick_params(labelsize=fsize)
        cbar.set_label(title)
        plt.tight_layout()
        plt.show() 
        



