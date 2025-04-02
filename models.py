import numpy as np
import tensorflow as tf
import keras


def hotdog_model(input_shape):
    """
    Convolutional Neural Network (CNN) model for detecting hotdogs.

    The model is compiled using the Adam optimizer, binary crossentropy loss,
    and accuracy metric.

    Parameters
    ----------
    input_shape : tuple
      The shape of the input data, including the height, width, and
      channels of the input image.

    Returns
    -------
    model : keras.Sequential
      A compiled Keras Sequential model with the CNN architecture.
    """
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape),
        keras.layers.Conv2D(16, 3, activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.5),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.5),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', 
        loss='binary_crossentropy',
        metrics='accuracy')

    return model
