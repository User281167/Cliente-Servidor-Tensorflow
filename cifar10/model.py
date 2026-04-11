"""
Modelo CIFAR-10 para clasificación de imágenes.

Permite usar el model.fit() de Keras.
"""

import tensorflow as tf


def create_model(gray=True, conv=False):
    """
    Modelo CIFAR-10.

    Args:
        gray (bool): Si es True, usa imágenes en escala de grises.
        conv (bool): Si es True, usa una red convolucional.
    """

    if conv:
        return tf.keras.Sequential(
            [
                tf.keras.Input(shape=(32, 32, 1 if gray else 3)),
                tf.keras.layers.Conv2D(32, 3, padding="same"),
                tf.keras.layers.LeakyReLU(0.01),
                tf.keras.layers.SpatialDropout2D(0.3),
                tf.keras.layers.MaxPooling2D(2),
                tf.keras.layers.Conv2D(64, 3, padding="same"),
                tf.keras.layers.LeakyReLU(0.01),
                tf.keras.layers.MaxPooling2D(2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.LeakyReLU(0.01),
                tf.keras.layers.Dense(10),
            ]
        )

    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(32, 32, 1 if gray else 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(0.01),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(32),
            tf.keras.layers.LeakyReLU(0.01),
            tf.keras.layers.Dense(10),
        ]
    )
