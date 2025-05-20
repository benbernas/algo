# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:58:43 2024

@author: Benoît Bernas

Main code to run deep learning model for the detection and the localization of MBs 
"""

#import libraries

import sys
import os
import scipy.io
import h5py


import matplotlib.pyplot as plt
import numpy as np

import time

import tensorflow as tf

from sklearn.model_selection import KFold, StratifiedKFold




from joblib import dump

#%% define pre-param

#% define pre-param

dir_path = r'D:\Benoit\deep_learning_for_ulm'
# dir_path = r'C:\Benoit\deep_learning'
os.chdir(dir_path)
# simu_path = dir_path + r'\matlab\field_simu\realistic_database\2025-05-06_10blocks_5scats\simu_test_noise'
simu_path = dir_path + r'\matlab\field_simu\simple_database\2024-08-23_10blocks_5scats_no_noise'
save_fig = 1

#path database
# D:\Benoit\machine_learning\python\deep_learning\algo\generate_database\database
# load_path = dir_path + r'\python\datasets\5mb\realistic_dataset\dataset_IQ_svd_sizeImg_121_128_5mb_10blocks.h5'
load_path = dir_path + r'\python\datasets\5mb\simple_dataset\dataset_IQ_mb_noise_5per_sizeImg_91_128_5mb_10blocks.h5'
# D:\Benoit\deep_learning_for_ulm\python\GitHub\algo
#path containing functions 
function_path = dir_path + r'\python\GitHub\algo\functions'
# Add the directory to sys.path
sys.path.append(function_path)
from my_module import check_environment
#%% check if you are running the code on GPU or running on CPU
infoSys = check_environment()
# print(runOnGPU) 
#%% load database
    
os.makedirs(os.path.dirname(load_path), exist_ok=True)
BfStruct = scipy.io.loadmat(os.path.join(simu_path, 'BfStruct.mat'))['BfStruct']

# BfStruct = BfStruct['BfStruct']
xExtent = float(BfStruct['x0']), float(BfStruct['x1']); # equivaut to [- (number of probe elements)/2 , +(number of probe elements)/2] * pitch
zExtent = float(BfStruct['z0']), float(BfStruct['z1']);
dz = float(BfStruct['dz']);
dx = float(BfStruct['dx']);
nframes = float(BfStruct['nframes']);

# is_dataset = 'realistic' #realistic or simple
is_dataset = 'simple' #realistic or simple
    
with h5py.File(load_path, 'r') as file:
    
    # norm_IQ_mb_noised = file["norm_IQ_mb_noised"][:,:,:]
    if is_dataset == 'realistic':
        norm_IQ_mb_noised = file["norm_IQ_svd"][:]
    elif is_dataset == 'simple':
        norm_IQ_mb_noised = file["norm_IQ_mb_noised"][:]
        norm_IQ_mb_noised = np.transpose(norm_IQ_mb_noised, (2,0,1))

    # special research into all strings files to know if there is a sub-img or not in the loaded database
    list_key = list(file.keys())[:]
    search_text = 'sub_img_norm_IQ_mb_shuffled'
    found_string = None
    for key in list_key:
        if search_text in key:
            found_string = key
            subimages = file["sub_img_norm_IQ_mb_shuffled"][:]
            break

        else:
            subimages = 'None'
            # norm_IQ_mb_shuffled = file["norm_IQ_mb_shuffled"][:,:,:]      
    coord_mb_shuffled = file["coord_mb_shuffled"][:]
    IQ_train = file["IQ_train"][:]
    coord_train = file["coord_train"][:]
    xTest = file["xTest"][:]
    yTest = file["yTest"][:]
    

#%% not mandatory

from preprocess_mb import clean_and_rank

(IQ_train, coord_train_desc,
 IQ_test,  coord_test_desc,
 infos_train, infos_test) = clean_and_rank(
        train_IQ=IQ_train,
        train_coord_mm=coord_train,
        test_IQ=xTest,
        test_coord_mm=yTest,
        zExtent=zExtent, xExtent=xExtent,
        dz=dz, dx=dx,
        dist_threshold_px=0.105)

# Désormais IQ_train / coord_train_desc nettoyés et triés
       
#%% Show database
if subimages == 'None':
    print('Shape of X database = ', norm_IQ_mb_noised.shape)
else:
    print('Shape of X database = ', subimages.shape)
# print('Shape of X database (sub images) = ', subimages.shape)
print('Shape of Y database = ', coord_mb_shuffled.shape)
print('Shape of IQ Train = ', IQ_train.shape)
print('Shape of coord Train" = ', coord_train.shape)
print('Shape of x test = ', xTest.shape)
print('Shape of y test = ', yTest.shape)

img_depth = norm_IQ_mb_noised.shape[1]
img_width = norm_IQ_mb_noised.shape[2]
sizeConv2D = 3
activationConv2D = 'relu'
sizeMaxPool = 2
dropoutValue = 0
activationDenseLayer = 'sigmoid'
# Loss = 'hugarian_matching_loss' 
Loss = 'optimal_matching_loss' 
Optimizer = 'Adam'
Epoch = 2
learningRate = 1e-4
batchSize = 64
crossVal = 'KFold'
nKFold = 2
nMB = yTest.shape[2] # MB number simulated
nblocks = 10
save_model = 1 
pixel_size = (dx+dz)/2

#%% save path folder 

save_folder = dir_path + r'\python\results\test_simu_5mb\realistic_dataset\vggnet\classic_matching_loss\test'
save_subfolder0 = r'\nblocks_{}'.format(nblocks)
# save_subfolder1 = r'\sizeConv2D_{}'.format(sizeConv2D)+'_{}'.format(sizeConv2D)
save_subfolder2 = r'\cross_val_{}'.format(crossVal) + '_{}'.format(nKFold) 
save_subfolder3 = r'\epoch_{}'.format(Epoch)
# save_subfolder4 = r'\learning_rate_{}'.format(learningRate)
save_subfolder5 =  r'\dropout_{}'.format(dropoutValue)
create_subfolder = os.path.join(save_folder + save_subfolder0 + save_subfolder2 + save_subfolder3 + save_subfolder5)
# create_subfolder = r'D:\Benoit\machine_learning\python\deep_learning\results\test_simu_5mb\cnn_prediction\IQ_mb_no_noise\adding_gaussian_noise_5per\quick_test\hugarian\add_lr_callback_l2_regularizer_0.0001'
# Create the subfolder
os.makedirs(create_subfolder, exist_ok=True)

#%%  import my functions

from custom_loss import matching_mb, optimal_matching_loss, euclidean_distance_loss_hungarian, set_matching_params
from visualize_callback import build_callbacks
from tensorflow.keras.callbacks import EarlyStopping
from my_models import get_model
from localization_evaluation import global_rmse, plot_rmse_and_density_map

# set constant paramater zExtent, xExtent, nMB
set_matching_params(zExtent, xExtent, nMB)

# Set the number of folds
k = nKFold 
# kf = StratifiedKFold(n_splits=k, shuffle=True)
kf = KFold(n_splits=k, shuffle=True)
# kf_y = np.ones(5)
# This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.
# yData_stratified = np.floor(coord_train[:, 0]/0.1).astype(int)

all_fold_train_histories = []
all_fold_y_pred_coord = []
all_fold_rmse = []
n_k = 0

start_time = time.time()
# Perform K-fold cross-validation
for train_index, val_index in kf.split(IQ_train):
    xTrain, xVal = IQ_train[train_index], IQ_train[val_index]
    yTrain, yVal = coord_train_desc[train_index], coord_train_desc[val_index]
    # yTrain, yVal = coord_train[train_index], coord_train[val_index]
     
    # reinitialization of K-Fold
    tf.keras.backend.clear_session()

    


    # paramètres communs
    input_shape = (img_depth, img_width, 1)

    # # Exemple 1 : CNN simple avec régularisation L2 et loss « optimal »
    # model = get_model("cnn",
    #                   input_shape=input_shape,
    #                   nMB=nMB,
    #                   learning_rate=1e-4,
    #                   l2_lambda=1e-4,
    #                   loss_name="optimal")

    # Exemple 2 : VGG‑net avec loss « hungarian »
    model = get_model("vgg",
                      input_shape=input_shape,
                      nMB=nMB,
                      learning_rate=1e-4,
                      loss_name="optimal")  #"optimal" or "hugarian" matching method
    
    n_k = n_k+1
    # set callbacks (add loss history live visuzalization + matching MBs saving + early stopping)
    mycallbacks = build_callbacks(xVal, yVal,
                                matching_mb_func=matching_mb,
                                viz_dir= create_subfolder + r'\grid_viz{}'.format(n_k),
                                max_samples=9,
                                patience=10)

    
    history = model.fit(xTrain, yTrain, epochs=Epoch, batch_size=batchSize, validation_data=(xVal, yVal), verbose=1, callbacks=mycallbacks)
   
    # Evaluate the model on the validation data
    val_scores = model.evaluate(xVal, yVal, verbose=1)
    
    ##### testing
    
    # test and performance --> compute prediction 
    y_pred_coord = model.predict(xTest[:,:,:].reshape(len(xTest), img_depth, img_width, 1))
    
    #% Evaluate the deep learning model on test set 
    # error in prediction --> root mean square error (RMSE)     
    rmse_results = global_rmse(yTest, y_pred_coord, nMB)
    # inputs = def global_rmse(true_coords, loc_coords, nMB)
    # outputs = rmse, e_x, e_z

    all_fold_train_histories.append(history.history)
    all_fold_y_pred_coord.append(y_pred_coord)
    all_fold_rmse.append(rmse_results)

    
end_time = time.time() 
model.summary()

# Calculate elapsed time of the model 
elapsed_time = (end_time - start_time)/60 #elapsed time of model (min)
print(elapsed_time)
    
#%% Plot all train and validation functions for each cross-validation  
n_subplots = nKFold   # Nombre de subplots souhaité
if n_subplots == 5:
    n_rows = 2
    n_cols = 3
elif n_subplots == 2:
    n_rows = 1
    n_cols = 2

fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 6))
ax = ax.flatten()  # Transforme en liste à 1D pour itérer facilement

for n in range(n_subplots):
    min_loss = min(all_fold_train_histories[n]['loss'])
    last_loss = all_fold_train_histories[n]['loss'][-1]
    min_val_loss = min(all_fold_train_histories[n]['val_loss'])
    last_val_loss = all_fold_train_histories[n]['val_loss'][-1]

    ax[n].plot(all_fold_train_histories[n]['loss'], label='loss; min = {:.3f}'.format(min_loss))
    ax[n].plot(all_fold_train_histories[n]['val_loss'], label='val loss; min = {:.3f}'.format(min_val_loss))
    ax[n].set_xlabel('Epoch')
    ax[n].set_xlim(0, Epoch)
    ax[n].grid(True)
    ax[n].legend()

# Supprimer le dernier subplot inutilisé (le 6e)
if n_subplots < len(ax):
    fig.delaxes(ax[-1])  # Supprime l’axe 6 inutile

plt.tight_layout()
plt.show()
plt.savefig(create_subfolder + "\loss.png")

#%% save model and data

if save_model == 1:
    # save_folder = dir_path + r'\results\test_simu_5mb\cnn_prediction\IQ_mb_no_noise\adding_gaussian_noise_5per\hugarian_matching_loss_with_contrainst'

    
    # Save variables to a file
    dump({'IQ': norm_IQ_mb_noised,
          'sub_IQ': subimages,
          'mb_coord': coord_mb_shuffled,
          'IQ_train': IQ_train,
          'mb_coord_train': coord_train, 
          'infos_intensities_train': infos_train,
          'mb_coord_train_descend': coord_train_desc,
          'IQ_test': xTest,
          'mb_coord_test': yTest,
          'infos_intensities_test': infos_test,
          'mb_coord_test_descend': coord_test_desc,
          'xTrain': xTrain,
          'yTrain': yTrain,
          'xVal': xVal,
          'yVal': yVal,
          'BfStruct': BfStruct,
          'nblocks_train': nblocks,
          'nMB_train' : nMB, # MB number simulated
          'simu_path': simu_path, 
          'dataset_path': load_path
          }, create_subfolder + r'\training_dataset.joblib')
    
    model.save(create_subfolder + r'\model.keras')
    
    
    dump({'img_depth': img_depth,
          'img_width': img_width,
          # 'sub_img_depth': sub_img_depth,
          # 'sub_img_width': sub_img_width,
          'sizeConv2D': sizeConv2D,
          'activationConv2D': activationConv2D,
          'sizeMaxPool': sizeMaxPool,
          'dropoutValue': dropoutValue,
          'activationDenseLayer': activationDenseLayer,
          'Loss': Loss,
          'Epoch': Epoch,
          'batchSize': batchSize,
          'Optimizer': Optimizer,
          'learningRate': learningRate,
          'crossVal': crossVal,
          'nKFold': nKFold,
          'runOnGPU': bool(infoSys[0]),
          'python': infoSys[1],
          'TensorFlow': infoSys[2],
          'cuda': infoSys[3],
          'cuDNN': infoSys[4]
          }, create_subfolder + r'\model_param.joblib')
    
    dump({'elapsed_time': elapsed_time,
          'all_fold_train_histories': all_fold_train_histories, 
          'all_fold_y_pred_coord': all_fold_y_pred_coord,
          'pixel_size': pixel_size,
          'all_fold_rmse': all_fold_rmse,
          # 'all_fold_mean_rmse': mean_rmse_tot,
          # 'all_fold_mean_std': std_rmse_tot,
          
          }, create_subfolder + r'\results.joblib')
elif save_model == 0:
    print('no saving param / training / result model')


#%% Run this part if you want to analyze and plot data and results, directly after the training of the model
quick_check = 1

if quick_check == 1:
    
    #------------------------------------------------------------------
    # Quick check of the dataset 
    #------------------------------------------------------------------
    from visualize_data import plot_data_and_loc
    
    """ choose between those (str) : 
        "train_several_frames", "train_movie_frames", 
        "val_several_frames", "val_movie_frames", 
        "test_several_frames", "test_movie_frames",
        "check_pwd"
    """
    
    plot_data_and_loc(
        iq_shuffled=norm_IQ_mb_noised, coord_mb_shuffled=coord_mb_shuffled, 
        xTrain=xTrain, yTrain=yTrain,
        xTest=xTest, yTest=yTest, y_pred_coord=y_pred_coord,
        xVal=xVal, yVal=yVal,
        xExtent=xExtent, zExtent=zExtent, dx=dx, dz=dz,
        display_modes=("train_several_frames"),
        frame_wanted=[0, 26, 72, 134],
        pause_duration=0.5,
        n_frames=100
    )
    
    #%% check histories (train ; val loss)
    from visualize_callback import plot_histories
    plot_histories(all_fold_train_histories, max_epoch=50)
    
    #%% Plot rmse evaluation 
    """
    inputs = def plot_rmse_and_density_map(loc_coords, rmse, train_coords, extent, resolution, 
                                   num_bins_x=11, num_bins_z=10, 
                                   c_lim_rmse=[0, 20], c_lim_num_mb=[0, 120], 
                                   name_plot=None)         
             """
    plot_rmse_and_density_map(y_pred_coord, rmse_results["rmse_mb"], yTrain, nMB, [xExtent, zExtent], [dx ,dz])
    
    #% Make movie of the validation loss callback 
    from visualize_callback import make_movie_epoch
    make_movie_epoch(nKFold, create_subfolder)
    
elif quick_check == 0:
    print('no quick check')
        




