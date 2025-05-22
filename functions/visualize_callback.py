# visualize_callback.py
"""
Prediction and loss function visualization of the deep learning model during training
(particularly useful for the custom loss function) 

@author: Benoît Bernas
"""

""" Function to help the visualization of the matching callback  """



""" live plotting of trainig and validation loss functions """


        

import os, math
import matplotlib.pyplot as plt
import tensorflow as tf

import cv2
from natsort import natsorted
import glob

# ————————————————————————————————————————————————
# 1. Live display of loss curves
# ————————————————————————————————————————————————
class LossHistory(tf.keras.callbacks.Callback):
    """Trace en temps‑réel la loss et la val_loss pendant l’entraînement."""
    def on_train_begin(self, logs=None):
        self.losses, self.val_losses = [], []
        plt.ion()
        self.fig, self.ax = plt.subplots()

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs["loss"])
        self.val_losses.append(logs["val_loss"])

        self.ax.clear()
        self.ax.plot(self.losses,     label="Loss (train)")
        self.ax.plot(self.val_losses, label="Loss (val)")
        self.ax.set(xlabel="Epoch", ylabel="Loss",
                    title="Training loss – live")
        self.ax.legend()
        plt.pause(0.1)

    def on_train_end(self, logs=None):
        plt.ioff()
        plt.show()


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 2. Visualization of the matching MB between predicted and simulated coordinates during model training
#    Written here to visualize the loss function of validation data, but can also work for the train data loss 
# ————————————————————————————————————————————————————————————————————————————————————————————————————————————
def _default_matching_mb(x_t, z_t, x_p, z_p):
    """Fallback s’il n’existe pas encore de matching_mb dans le main script."""
    raise RuntimeError("matching_mb n’a pas été fourni au callback")

class GridMatchingVisualizationCallback(tf.keras.callbacks.Callback):
    """
    Sauvegarde à chaque epoch un montage d’images montrant
    l’appariement optimal entre MB simulées et prédites.
    """
    def __init__(self, x_val, y_val,
                 matching_mb_func=_default_matching_mb,
                 output_dir="grid_viz",
                 max_samples=9):
        super().__init__()
        self.x_val, self.y_val = x_val, y_val
        self.matching_mb = matching_mb_func      # function passed from main script
        self.output_dir  = output_dir
        self.max_samples = max_samples
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.x_val[:self.max_samples], verbose=0)
        n_rows = int(math.ceil(self.max_samples / 3))
        fig, axs = plt.subplots(n_rows, 3, figsize=(12, 3*n_rows))
        axs = axs.flatten()

        colors = ['r', 'g', 'b', 'orange', 'purple']
        for i in range(self.max_samples):
            x_t, z_t = self.y_val[i][:5],  self.y_val[i][5:]
            x_p, z_p = preds[i][:5],       preds[i][5:]

            perm = self.matching_mb(x_t, z_t, x_p, z_p)

            ax = axs[i]
            ax.scatter(x_t, z_t, c='k', marker='x', label='True')
            for j in range(5):
                pj = perm[j]
                ax.scatter(x_p[pj], z_p[pj], c=colors[j])
                ax.plot([x_t[j], x_p[pj]], [z_t[j], z_p[pj]],
                        c=colors[j], ls='--')
            ax.set_xlim(-7, 7);  ax.set_ylim(0, 10)
            ax.set_title(f"Sample {i}")
            ax.axis("off")

        for j in range(self.max_samples, len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/epoch_{epoch+1:03d}.png")
        plt.close()


# ————————————————————————————————————————————————
# 3. Build a stack of callbacks easily
# ————————————————————————————————————————————————
def build_callbacks(x_val, y_val, matching_mb_func,
                    viz_dir="grid_viz", max_samples=9,
                    patience=10):
    """Renvoie la liste standard de callbacks à brancher sur model.fit."""
    loss_hist = LossHistory()
    viz_cb    = GridMatchingVisualizationCallback(
                    x_val, y_val,
                    matching_mb_func=matching_mb_func,
                    output_dir=viz_dir,
                    max_samples=max_samples)
    early     = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True)
    return [loss_hist, viz_cb, early]
        



# —————————————————————————————————————————————————————————————
# 4. Make a movie of the validation data callback .png images
# ————————————————————————————————————————————————————————————

def make_movie_epoch(number_of_kFold, save_path):
# Dossier où se trouvent les images
    
    nKFold = number_of_kFold;
    create_subfolder = save_path;
    for i in range(nKFold):
        
        image_folder = create_subfolder + r'\grid_viz{}'.format(i+1)
        
        # Sorted list of images
        image_files = natsorted(glob.glob(os.path.join(image_folder, "epoch_*.png")))
        
        # Read the dimensions of the first image
        frame = cv2.imread(image_files[0])
        height, width, layers = frame.shape
        
        # Exit path
        video_name = os.path.join(image_folder, 'epoch_video.mp4')
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 0, 255)  # Red color
        thickness = 2
        position = (int(width/2-75), 75)  # (x, y), value 75 arbitrarily set by graphical view
        
        # Add images to video with title
        for idx, image_file in enumerate(image_files):
            frame = cv2.imread(image_file)
        
            # Add text to frame
            epoch_text = f"Epoch {idx + 1}"
            cv2.putText(frame, epoch_text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
        
            video.write(frame)
        
        video.release()
        print(f" Video with titles saved : {video_name}")
        
        return
    

def plot_histories(histories, max_epoch=None):
    """
    Loads and displays loss curves from a .joblib file containing historical data.

    Args:
        histories (list): list of the all K-Fold train - val loss (ex: “model_param.joblib”)
        max_epoch (int): Max number of epochs to display (optional)
    """

    n_folds = len(histories)-1
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

