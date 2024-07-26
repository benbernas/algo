# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:58:43 2024

@author: labo
"""

### CNN model for detection and localization of MBs 
# Clean algorithm of "test_cnn_coord"

import sys
import os
import scipy.io
import h5py


import matplotlib.pyplot as plt
import numpy as np
import random
import time
from sklearn.metrics import root_mean_squared_error, accuracy_score


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import root_mean_squared_error, accuracy_score

import pickle

#%% define pre-param

dir_path = r'D:\Benoit\machine_learning\python\deep_learning'
os.chdir(dir_path)
simu_path = dir_path + r'\field_simu\2024-06-26_10blocks_1scats_no_noise'
save_fig = 1

#%% check if you are running the code on GPU or running on CPU
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

#check run on GPU or not 
infoSys = check_environment()
# print(runOnGPU) 
#%% load database
    
# D:\Benoit\machine_learning\python\deep_learning\algo\generate_database\database
load_path = dir_path + r'\algo\generate_database\database\database_complex_IQ_mb_noised_correction.h5'
os.makedirs(os.path.dirname(load_path), exist_ok=True)
BfStruct = scipy.io.loadmat(os.path.join(simu_path, 'BfStruct.mat'))['BfStruct']

# BfStruct = BfStruct['BfStruct']
xExtent = float(BfStruct['x0']), float(BfStruct['x1']); # equivaut to [- (number of probe elements)/2 , +(number of probe elements)/2] * pitch
zExtent = float(BfStruct['z0']), float(BfStruct['z1']);
dz = float(BfStruct['dz']);
dx = float(BfStruct['dx']);
nframes = float(BfStruct['nframes']);
    
with h5py.File(load_path, 'r') as file:
    norm_IQ_mb_noised = file["norm_IQ_mb_noised"][:,:,:]
    mb_coord = file["simu_mb_coord"][:,:]
    IQ_train = file["IQ_train"][:,:,:]
    coord_train = file["coord_train"][:,:]
    # xTrain = file["xTrain"][:,:,:]
    # yTrain = file["yTrain"][:,:]
    xTest = file["xTest"][:,:,:]
    yTest = file["yTest"][:,:]
    # xVal = file["xVal"][:,:,:]
    # yVal = file["yVal"][:,:]


#%%  check xTrain and yTrain
# faire fonction check xTrain yTrain
# plt.figure()   
# # for ii in range(1):     
# for ii in range(200):
    
#     plt.imshow(xTrain[ii,:,:], extent=[xExtent[0]-dx/2, xExtent[1]+dx/2, zExtent[1]+dz/2, zExtent[0]-dz/2])
#     # plt.imshow(xTrain[ii,:,:])
    
#     plt.scatter(yTrain[ii, 0], yTrain[ii, 1], marker = '+', c='red')
#     # plt.hold(True)
#     # plt.hold
#     plt.pause(0.1)
#     plt.show()
    
#%% check xTrain and yTrain after transposing and reshaping 
#
# plt.figure()   
# # for ii in range(1):     
# for ii in range(200):
    
#     plt.imshow(raw_IQ_mb[:,:,ii], extent=[xExtent[0]-dx/2, xExtent[1]+dx/2, zExtent[1]+dz/2, zExtent[0]-dz/2])
#     # plt.imshow(xTrain[ii,:,:])
    
#     plt.scatter(yTrain[ii, 0], yTrain[ii, 1], marker = '+', c='red')
#     # plt.hold(True)
#     # plt.hold
#     plt.pause(0.1)
#     plt.show()
       
#%% Show database

print('Shape of X database = ', norm_IQ_mb_noised.shape)
print('Shape of Y database = ', mb_coord.shape)
print('Shape of IQ Train = ', IQ_train.shape)
print('Shape of coord Train" = ', coord_train.shape)
# print('Shape of IQ test = ', IQ_test.shape)
# print('Shape of x train = ', xTrain.shape)
# print('Shape of y train = ', yTrain.shape)
# print('Shape of x validation = ', xVal.shape)
# print('Shape of y validation = ', yVal.shape)
print('Shape of x test = ', xTest.shape)
print('Shape of y test = ', yTest.shape)

img_depth = norm_IQ_mb_noised.shape[1]
img_width = norm_IQ_mb_noised.shape[2]
sizeConv2D = 15
activationConv2D = 'relu'
sizeMaxPool = 2
dropoutValue = 0.5
activationDenseLayer = 'sigmoid'
Loss = 'mse'
Optimizer = 'Adam'
Epoch = 5000 
learningRate = 1e-3
batchSize = 20
crossVal = 'KFold'
nKFold = 2

save_folder = dir_path + r'\results\test_simu_1bulle\cnn_prediction\IQ_mb_no_noise\adding_gaussian_noise_5_per_correction'
save_subfolder1 = r'\sizeConv2D_{}'.format(sizeConv2D)+'_{}'.format(sizeConv2D)
save_subfolder2 = r'\cross_val_{}'.format(crossVal) + '_{}'.format(nKFold) 
save_subfolder3 = r'\epoch_{}'.format(Epoch)
# save_subfolder5 = r'\nKFold_{}'.format(nKFold)
# save_subfolder5 = r'\dropout_afterSecondMaxPooling'


# Combine the parent folder path with the subfolder name to create the full path
create_subfolder = os.path.join(save_folder + save_subfolder1 + save_subfolder2 + save_subfolder3)

# Create the subfolder
os.makedirs(create_subfolder, exist_ok=True)

print(f"Subfolder '{save_subfolder1}' and  '{save_subfolder2}' created in '{create_subfolder}'")


#%%
# Define model 
def create_model(input_shape):
    cnn = Sequential([
        Conv2D(16, (sizeConv2D, sizeConv2D), activation=activationConv2D, input_shape=input_shape),
        MaxPooling2D((sizeMaxPool, sizeMaxPool)),
        Conv2D(32, (sizeConv2D, sizeConv2D), activation=activationConv2D),
        MaxPooling2D((sizeMaxPool, sizeMaxPool)),
        Dropout(dropoutValue),
        Flatten(),
        Dense(64, activation=activationDenseLayer),
        Dense(2)  # Output layer to obtain [z, x] coordinates
        ])
    opt = tf.keras.optimizers.Adam(learning_rate=learningRate)
    cnn.compile(loss=Loss, optimizer=opt, metrics=['accuracy', 'mse'])
    return cnn


# Set the number of folds
k = nKFold 
# kf = StratifiedKFold(n_splits=k, shuffle=True)
kf = KFold(n_splits=k, shuffle=True)
# kf_y = np.ones(5)
# This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.
# yData_stratified = np.floor(coord_train[:, 0]/0.1).astype(int)


# Lists to store the results
all_fold_val_scores = []
all_fold_train_histories = []
all_fold_y_pred_coord = []
all_fold_dist_x_true_pred = []
all_fold_dist_z_true_pred = []
all_fold_abs_dist_true_pred = []
all_fold_rmse = []


start_time = time.time()
# Perform K-fold cross-validation
for train_index, val_index in kf.split(IQ_train):
    xTrain, xVal = IQ_train[train_index], IQ_train[val_index]
    yTrain, yVal = coord_train[train_index], coord_train[val_index]

    # Create and train the model
    model = create_model((img_depth, img_width, 1))
    history = model.fit(xTrain, yTrain, epochs=Epoch, batch_size=batchSize, validation_data=(xVal, yVal), verbose=1)
    # print(history.history['loss'])
    # print(history.history['loss'])
   
    # Evaluate the model on the validation data
    val_scores = model.evaluate(xVal, yVal, verbose=1)
    all_fold_val_scores.append(val_scores)
    
    
    # test and performance 
    y_pred_coord = model.predict(xTest[:,:,:].reshape(len(xTest), img_depth, img_width, 1))
    all_fold_y_pred_coord.append(y_pred_coord)
    
    #calculate the distance [x,z] between true and predicted coordinates
    dist_x_true_pred = yTest[:,0] - y_pred_coord[:,0]
    dist_z_true_pred = yTest[:,1] - y_pred_coord[:,1]
    abs_dist_true_pred = np.sqrt(dist_x_true_pred**2 + dist_z_true_pred**2)
    
    all_fold_dist_x_true_pred.append(dist_x_true_pred)
    all_fold_dist_z_true_pred.append(dist_z_true_pred)
    all_fold_abs_dist_true_pred.append(abs_dist_true_pred)

    
    # error in prediction --> root mean square error (RMSE)
    # yTest = our y_true 
    RMSE_pred  = np.sqrt(np.mean((yTest[:,:] - y_pred_coord[:,:])**2))
    RMSE_pred_x  = np.sqrt(np.mean((yTest[:,0] - y_pred_coord[:,0])**2))
    RMSE_pred_z  = np.sqrt(np.mean((yTest[:,1] - y_pred_coord[:,1])**2))
    RMSE = [RMSE_pred, RMSE_pred_x, RMSE_pred_z]
    all_fold_rmse.append(RMSE)
    

    # Save the training history for further analysis (optional)
    all_fold_train_histories.append(history.history)
    
end_time = time.time()
model.summary()

#%%
# Loss
fig1, ax = plt.subplots()
ax.plot(history.history['loss'], label='loss')
ax.plot(history.history['val_loss'], label='val loss')
ax.set_xlabel('Epoch')
ax.legend()
# Accuracy
fig2, ax = plt.subplots()
ax.plot(history.history['accuracy'], label='accuracy')
ax.set_xlabel('Epoch')
# fig, ax = plt.subplots()
# ax.plot(history.history['ROC'], label='ROC')
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
ax.legend()

#%%
### PREDICTION ###
# faire fonction check prediction
## check 1 prediction
# idx_test = random.randint(0, xTest.shape[0])
# IQ_mb_test = xTest[idx_test,:,:]
# true_coord = yTest[idx_test,:]

# predicted_coordinates = model.predict(IQ_mb_test.reshape(1, img_depth, img_width, 1))
# fig, ax = plt.subplots()
# ax.imshow(IQ_mb_test, extent=[xExtent[0]-dx/2, xExtent[1]+dx/2, zExtent[1]+dz/2, zExtent[0]-dz/2])
# # ax.imshow(IQ_mb_test, extent=[xExtent[0], xExtent[1], zExtent[1], zExtent[0]])
# ax.scatter(predicted_coordinates[0,0],predicted_coordinates[0,1], marker = '+', c='red', label = 'Localization of the MB predicted by the DL model')
# ax.scatter(true_coord[0], true_coord[1], marker = '+', c='green', label = 'True localization of the MB')
# ax.set_title('1 MB simulated. Frame sent for prediction')
# ax.set_xlabel('x (mm)')
# ax.set_ylabel('z (mm)')
# ax.legend(loc='lower right')
# dist_true_pred_plot = np.sqrt((true_coord[0] - predicted_coordinates[0,0])**2 + (true_coord[1] - predicted_coordinates[0,1])**2)
# print(dist_true_pred_plot)
# ax.show()


#%% show results
# Calculate elapsed time of the model 
elapsed_time = (end_time - start_time)/60 #elapsed time of model (min)

## Calculate mean and standard deviation
# validation scores
avg_val_score = np.mean(all_fold_val_scores, axis=0)
std_val_score = np.std(all_fold_val_scores, axis=0)
# rmse 
avg_rmse = np.mean(all_fold_rmse, axis=0)
std_rmse = np.std(all_fold_rmse, axis=0)


print(f"Elapsed time for training: {elapsed_time:.2f} min")
pixel_size = (dx+dz)/2
print('Pixel size = ',  "{:.3f}".format(pixel_size), 'mm') 
print(f'[mean, std] over {k} folds:')
print(f' -> validation scores = [{avg_val_score[:2]}, {std_val_score[:2]}]')
print(' -> RMSE between true position and predicted position of MBs = ', "[{:.3f}, ".format(avg_rmse[0]), "{:.3f}]".format(std_rmse[0]), 'mm')
print(' -> RMSE over x between true position and predicted position of MBs = ', "[{:.3f}, ".format(avg_rmse[1]), "{:.3f}]".format(std_rmse[1]), 'mm')
print(' -> RMSE over z between true position and predicted position of MBs = ', "[{:.3f}, ".format(avg_rmse[2]), "{:.3f}]".format(std_rmse[2]), 'mm')
#%% Show figs

# distribution x coord
fig3 = plt.figure()
plt.hist(yTrain[:,0], bins=int((np.max(yTrain[:,0]) - np.min(yTrain[:,0]))*10), color='blue', edgecolor='black', label = 'x coordinates of trained images')
# plt.legend('')
plt.xlabel('x (mm)')
plt.legend(loc='upper right')

# distribution z coord
fig4 = plt.figure()
plt.hist(yTrain[:,1], bins=int((np.max(yTrain[:,1]) - np.min(yTrain[:,1]))*10), color='red', edgecolor='black', label = 'z coordinates of trained images')
plt.xlabel('z (mm)')
plt.legend(loc='upper right')


bins_test=int((np.max(yTest[:,0]) - np.min(yTest[:,0]))*10)
fc = 15.6; #MHz
c0 = BfStruct['c0'].astype(float) 
lambda_radius = c0/fc # mm

x_bins = len(np.arange(np.min(yTest[:,0]), np.max(yTest[:,0])+2*lambda_radius, lambda_radius))
z_bins = len(np.arange(np.min(yTest[:,1]), np.max(yTest[:,1])+2*lambda_radius, lambda_radius))

counts_x, bins_coord_x = np.histogram(yTest[:,0], bins= x_bins)
counts_z, bins_coord_z = np.histogram(yTest[:,1], bins= z_bins)

#plot histogram between x true coordinates and x predicted coordinates
fig5 = plt.figure()
plt.hist(yTest[:,0], bins=x_bins, color='green', edgecolor='black', alpha = 0.5, label = 'x true coordinates')
plt.hist(y_pred_coord[:,0], bins=x_bins, color='red', edgecolor='black', alpha = 0.5, label = 'x predicted coordinates')
plt.legend(loc='upper right')
plt.xlabel('x (mm)')
plt.ylabel('Nx coordinates')


#plot histogram between z true coordinates and z predicted coordinates
fig6 = plt.figure()
plt.hist(yTest[:,1], bins=z_bins, color='green', edgecolor='black', alpha = 0.5, label = 'z true coordinates')
plt.hist(y_pred_coord[:,1], bins=z_bins, color='red', edgecolor='black', alpha = 0.5, label = 'z predicted coordinates')
plt.legend(loc='upper right')
plt.xlabel('z (mm)')
plt.ylabel('Nz coordinates')

#plot histogram of the distance between z true coordinates and z predicted coordinates

dist_x_true_pred = yTest[:,0] - y_pred_coord[:,0]
dist_z_true_pred = yTest[:,1] - y_pred_coord[:,1]

dist_true_pred = np.sqrt(dist_x_true_pred**2 + dist_z_true_pred**2)
dist_true_pred_test = np.mean(dist_true_pred)

# dist_true_pred_abs =  np.sqrt(np.mean(np.abs(dist_x_true_pred**2) + np.abs(dist_z_true_pred**2)))

fig7 = plt.figure()
plt.hist(dist_x_true_pred, bins=x_bins, color='orange', edgecolor='black', alpha = 0.5, label = 'distance between x true and x predicted coordinates')
plt.legend(loc='upper right')
plt.xlabel('distance x (mm)')
plt.ylabel('Nx distances')

fig8 = plt.figure()
plt.hist(dist_z_true_pred, bins=x_bins, color='blue', edgecolor='black', alpha = 0.5, label = 'distance between z true and z predicted coordinates')
plt.legend(loc='upper right')
plt.xlabel('distance z (mm)')
plt.ylabel('Nz distances')


fig9 = plt.figure()
plt.hist(dist_true_pred, bins=x_bins, color='red', edgecolor='black', alpha = 0.5, label = 'distance between true coordinates and predicted coordinates')
plt.legend(loc='upper right')
plt.xlabel('distance (mm)')
plt.ylabel('Nzx distances')

#%% save figs 
if save_fig == 1:
    
    savefig_path = r'\fig'
    create_figSubfolder = os.path.join(create_subfolder + savefig_path)
    # Create the subfolder
    os.makedirs(create_figSubfolder, exist_ok=True)
    print(f"figures saved in '{create_figSubfolder}'")
    
    savefig_name = r'\loss'
    file = open(create_figSubfolder + savefig_name + ".mpl", 'wb')
    pickle.dump(fig1, file)
    
    
    savefig_name = r'\accuracy'
    file = open(create_figSubfolder + savefig_name + ".mpl", 'wb')
    pickle.dump(fig2, file)
    
    savefig_name = r'\distribution_xcoord'
    file = open(create_figSubfolder + savefig_name + ".mpl", 'wb')
    pickle.dump(fig3, file)
    
    savefig_name = r'\distribution_zcoord'
    file = open(create_figSubfolder + savefig_name + ".mpl", 'wb')
    pickle.dump(fig4, file)
    
    savefig_name = r'\distribution_xcoord_true_pred'
    file = open(create_figSubfolder + savefig_name + ".mpl", 'wb')
    pickle.dump(fig5, file)
    
    savefig_name = r'\distribution_zcoord_true_pred'
    file = open(create_figSubfolder + savefig_name + ".mpl", 'wb')
    pickle.dump(fig6, file)
    
    savefig_name = r'\distance_xcoord_true_pred'
    file = open(create_figSubfolder + savefig_name + ".mpl", 'wb')
    pickle.dump(fig7, file)
    
    savefig_name = r'\distance_zcoord_true_pred'
    file = open(create_figSubfolder + savefig_name + ".mpl", 'wb')
    pickle.dump(fig8, file)
    
    savefig_name = r'\abs_distance_xzcoord_true_pred'
    file = open(create_figSubfolder + savefig_name + ".mpl", 'wb')
    pickle.dump(fig9, file)
elif save_fig == 0:
    print('no fig saved')
#%%

file_path = r'D:\Benoit\machine_learning\python\deep_learning\results\test_simu_1bulle\cnn_prediction\IQ_mb_no_noise\adding_gaussian_noise_5_per_correction\sizeConv2D_3_3\runGPU_False\cross_val_KFold_2\epoch_5000\dropout_afterSecondMaxPooling\fig'

if os.path.exists(file_path):
    print(f"The file exists: {file_path}")
else:
    print(f"The file does not exist: {file_path}")

#%% save model and data
from joblib import dump, load
save_model = 1 

if save_model == 1:
    # Save variables to a file
    dump({'IQ': norm_IQ_mb_noised,
          'mb_coord': mb_coord,

          'IQ_train': IQ_train,
          'mb_coord_train': coord_train, 
          'IQ_test': xTest,
          'mb_coord_test': yTest,
          
          'xTrain': xTrain,
          'yTrain': yTrain,
          'xVal': xVal,
          'yVal': yVal,
          'BfStruct': BfStruct,
          'dataset_path': load_path
          }, create_subfolder + r'\training_dataset.joblib')
    
    model.save(create_subfolder + r'\model.keras')
    
    
    dump({'img_depth': img_depth,
          'img_width': img_width,
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
          'all_fold_val_scores': all_fold_val_scores,
          'avg_val_score': avg_val_score,
          'std_val_score': std_val_score,
          'all_fold_y_pred_coord': all_fold_y_pred_coord,
          'all_fold_dist_x_true_pred': all_fold_dist_x_true_pred,
          'all_fold_dist_z_true_pred':all_fold_dist_z_true_pred,
          'all_fold_abs_dist_true_pred':all_fold_abs_dist_true_pred,
          'pixel_size': pixel_size,
          'all_fold_rmse': all_fold_rmse,
          'avg_rmse_pred': avg_rmse,
          'std_rmse_pred': std_rmse,
          }, create_subfolder + r'\training_and_evaluation_results.joblib')
elif save_model == 0:
    print('no saving model')
    
    
    # for future plot : 
    #dist_x_true_pred = yTest[:,0] - all_fold_y_pred_coord[:,0]
    #dist_z_true_pred = yTest[:,1] - all_fold_y_pred_coord[:,1]

#%% load figs
load_fig = 0
if load_fig == 1: 
    load_folder = dir_path + r'\results\epoch_1000'
    load_subfolder = r'\fig'
    file = open(load_folder + load_subfolder + r'\loss.mpl', 'rb')
    figure = pickle.load(file)
    figure.show()
    
    file = open(load_folder + load_subfolder  + r'\abs_distance_xzcoord_true_pred.mpl', 'rb')
    figure = pickle.load(file)
    figure.show()
elif load_fig == 0:
    print('no need to load figures')

    
#%% load data
import numpy as np
from joblib import dump, load
from sklearn.metrics import root_mean_squared_error, accuracy_score
import tensorflow as tf
reload_data = 0
if reload_data == 1:
    load_result_path = r'D:\Benoit\machine_learning\python\deep_learning\old\test_simu_1bulle\cnn_prediction\long_test\IQ_mb_no_noise\adding_gaussian_noise_5_per_correction\sizeConv2D_3_3\cross_val_KFold\epoch_1000'
    # load_model = tf.keras.models.load_model(load_result_path + r'\model.keras')
    # load_model.summary()
    
    param_model = load(load_result_path + r'\model_param.joblib')
    # gpuState = param_model['runOnGPU']
    # nKFold = param_model['nKFold']
    eval_result = load(load_result_path + r'\training_and_evaluation_results.joblib')
    # elaps_time = eval_result['elapsed_time']
elif reload_data == 0:
    print('no reload data')

#%% load model
# load_model = tf.keras.models.load_model(create_subfolder + r'\model.keras')