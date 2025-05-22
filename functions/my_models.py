# my_models.py
"""
Created on Mon May 19 15:57:29 2025

@author: Benoît Bernas
"""

# ─────────────────────────────────────────────────────────────────────────
"""
Contient les architectures de réseaux et une fonction utilitaire
get_model(name, **kwargs) pour les importer facilement
depuis le script principal.

/!\ pour l'utilisation de cette fonction avec la localisation d'une seule MB (cnn_1mb_localisation.py) 
        --> sizeConv2D = 7
"""
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.regularizers import l2


# -----------------------------------------------------------------------
# 1. Fonctions auxiliaires
# -----------------------------------------------------------------------
def _build_optimizer(learning_rate):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def _select_loss(loss_name):
    """
    Permet de choisir la loss au moment de l’appel :
        loss='hungarian'   → euclidean_distance_loss_hungarian
        loss='optimal'     → optimal_matching_loss
    """
    if loss_name == "hungarian":
        from custom_loss import euclidean_distance_loss_hungarian
        return euclidean_distance_loss_hungarian
    elif loss_name == "optimal":
        from custom_loss import optimal_matching_loss
        return optimal_matching_loss
    else:
        raise ValueError(f"Loss inconnue : {loss_name}")


# -----------------------------------------------------------------------
# 2. Architecture CNN (simple)
# -----------------------------------------------------------------------
def _build_cnn(input_shape, nMB,
               sizeConv2D=3, sizeMaxPool=2,
               activationConv2D="relu",
               activationDense="sigmoid",
               learning_rate=1e-3,
               loss_name="optimal",
               l2_lambda=None):
    reg = l2(l2_lambda) if l2_lambda else None
    
    model = Sequential([
        layers.Conv2D(16, (sizeConv2D, sizeConv2D),
                      activation=activationConv2D,
                      kernel_regularizer=reg,
                      input_shape=input_shape),
        layers.MaxPooling2D((sizeMaxPool, sizeMaxPool)),

        layers.Conv2D(32, (sizeConv2D, sizeConv2D),
                      activation=activationConv2D,
                      kernel_regularizer=reg),
        layers.MaxPooling2D((sizeMaxPool, sizeMaxPool)),

        layers.Flatten(),
        layers.Dense(64, activation=activationDense, kernel_regularizer=reg),
        layers.Dense(nMB * 2)          # sortie plate : [x1..x5, z1..z5]
    ])

    model.compile(optimizer=_build_optimizer(learning_rate),
                  loss=_select_loss(loss_name),
                  metrics=["mae"])
    return model


# -----------------------------------------------------------------------
# 3. Architecture type VGG‑net
# -----------------------------------------------------------------------
def _build_vgg(input_shape, nMB,
               sizeConv2D=3, sizeMaxPool=2,
               learning_rate=1e-3,
               loss_name="optimal",
               l2_lambda=None):
    reg = l2(l2_lambda) if l2_lambda else None
    inp = layers.Input(shape=input_shape)

    # Bloc 1
    x = layers.Conv2D(64, (sizeConv2D, sizeConv2D),
                      activation="relu", padding="same",
                      kernel_regularizer=reg)(inp)
    x = layers.Conv2D(64, (sizeConv2D, sizeConv2D),
                      activation="relu", padding="same",
                      kernel_regularizer=reg)(x)
    x = layers.MaxPooling2D((sizeMaxPool, sizeMaxPool))(x)

    # Bloc 2
    x = layers.Conv2D(128, (sizeConv2D, sizeConv2D),
                      activation="relu", padding="same",
                      kernel_regularizer=reg)(x)
    x = layers.Conv2D(128, (sizeConv2D, sizeConv2D),
                      activation="relu", padding="same",
                      kernel_regularizer=reg)(x)
    x = layers.MaxPooling2D((sizeMaxPool, sizeMaxPool))(x)

    # Bottleneck
    x = layers.Conv2D(256, (sizeConv2D, sizeConv2D),
                      activation="relu", padding="same",
                      kernel_regularizer=reg)(x)
    x = layers.Conv2D(256, (sizeConv2D, sizeConv2D),
                      activation="relu", padding="same",
                      kernel_regularizer=reg)(x)

    # # Tête de régression
    # x = layers.Flatten()(x)
    # out = layers.Dense(nMB * 2)(x)
    
    
    x = layers.Flatten()(x)                    # on prend le goulot
    out = layers.Dense(nMB * 2,
                          activation="linear",
                          kernel_regularizer=reg)(x)
    # out = layers.Reshape((nMB * 2))(out)     # (nMB, [x,z])

    model = models.Model(inp, out)
    model.compile(optimizer=_build_optimizer(learning_rate),
                  loss=_select_loss(loss_name),
                  metrics=["mae"])
    return model


# -----------------------------------------------------------------------
# 4. Architecture U‑Net with coordinates (servira pour plus tard)
# -----------------------------------------------------------------------
def _build_unet(input_shape, nMB,
                learning_rate=1e-3,
                loss_name="optimal",
                l2_lambda=None):

    reg = l2(l2_lambda) if l2_lambda else None
    inputs = layers.Input(shape=input_shape)

    # Down‑sampling
    c1 = layers.Conv2D(64, 3, activation="relu", padding="same",
                       kernel_regularizer=reg)(inputs)
    c1 = layers.Conv2D(64, 3, activation="relu", padding="same",
                       kernel_regularizer=reg)(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(128, 3, activation="relu", padding="same",
                       kernel_regularizer=reg)(p1)
    c2 = layers.Conv2D(128, 3, activation="relu", padding="same",
                       kernel_regularizer=reg)(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    # Bottleneck
    b = layers.Conv2D(256, 3, activation="relu", padding="same",
                      kernel_regularizer=reg)(p2)
    b = layers.Conv2D(256, 3, activation="relu", padding="same",
                      kernel_regularizer=reg)(b)

    # Up‑sampling
    u2 = layers.Conv2DTranspose(128, 2, strides=2, padding="valid")(b)
    u2 = layers.Resizing(45, 64)(u2)             # aligne la taille
    u2 = layers.concatenate([u2, c2])

    c3 = layers.Conv2D(128, 3, activation="relu", padding="same",
                       kernel_regularizer=reg)(u2)
    c3 = layers.Conv2D(128, 3, activation="relu", padding="same",
                       kernel_regularizer=reg)(c3)

    u1 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c3)
    u1 = layers.Resizing(91, 128)(u1)
    u1 = layers.concatenate([u1, c1])

    c4 = layers.Conv2D(64, 3, activation="relu", padding="same",
                       kernel_regularizer=reg)(u1)
    c4 = layers.Conv2D(64, 3, activation="relu", padding="same",
                       kernel_regularizer=reg)(c4)

    # Tête de régression (coordonnées)
    flat = layers.Flatten()(b)                    # on prend le goulot
    coords = layers.Dense(nMB * 2,
                          activation="linear",
                          kernel_regularizer=reg)(flat)
    coords = layers.Reshape((nMB, 2))(coords)     # (nMB, [x,z])

    model = models.Model(inputs, coords)
    model.compile(optimizer=_build_optimizer(learning_rate),
                  loss=_select_loss(loss_name),
                  metrics=["mae"])
    return model


# -----------------------------------------------------------------------
# 5. Fabrique (factory) – interface publique
# -----------------------------------------------------------------------
def get_model(name,
              input_shape,
              nMB,
              **kwargs):
    """
    Ex :
        model = get_model("cnn",  input_shape=(91,128,1), nMB=5,
                          learning_rate=1e-4, l2_lambda=1e-4)

        model = get_model("vgg", input_shape=(91,128,1), nMB=5,
                          loss_name="hungarian")
    """
    name = name.lower()
    if name in ("cnn", "simple"):
        return _build_cnn(input_shape, nMB, **kwargs)
    elif name in ("vgg", "vggnet"):
        return _build_vgg(input_shape, nMB, **kwargs)
    elif name in ("unet", "unet_coords"):
        return _build_unet(input_shape, nMB, **kwargs) 
    else:
        raise ValueError(f"Architecture inconnue : {name}")


#%% Unet
# def unet_with_coords(input_shape=(img_depth, img_width, 1), num_classes=nMB):
#     inputs = layers.Input(input_shape)

#     # Downsampling
#     c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
#     c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
#     p1 = layers.MaxPooling2D((2, 2))(c1)

#     c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
#     c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
#     p2 = layers.MaxPooling2D((2, 2))(c2)

#     # Bottleneck
#     b = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
#     b = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(b)

#     # # Upsampling
#     u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='valid')(b)
#     u2 = layers.Resizing(45, 64)(u2)  # Alignement des tailles
#     u2 = layers.concatenate([u2, c2])

#     c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
#     c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)

#     u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
#     u1 = layers.Resizing(91, 128)(u1)
#     u1 = layers.concatenate([u1, c1])

#     c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
#     c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

#     # Feature vector for coordinates
#     flattened = layers.Flatten()(b)
#     # coord_output = layers.Dense(num_classes * 2)(flattened)
#     coord_output = layers.Dense(num_classes * 2, activation='linear')(flattened)
#     coord_output = layers.Reshape((num_classes, 2))(coord_output)

#     # Map output
#     # map_output = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c4)

#     model = models.Model(inputs, [coord_output])
    
#     opt = tf.keras.optimizers.Adam(learning_rate=learningRate)
    
#     # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     # initial_learning_rate=learningRate, decay_steps=100, decay_rate=0.96, staircase=True)
#     # opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

#     # cnn.compile(loss=euclidean_distance_loss_hungarian, optimizer=opt, metrics=['mae'])
#     model.compile(loss=optimal_matching_loss, optimizer=opt, metrics=['mae'])
#     return model