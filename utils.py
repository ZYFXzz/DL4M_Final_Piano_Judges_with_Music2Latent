import os
import numpy as np
import tensorflow as tf
import keras
import matplotlib


def load_training_data(data_path):
    """
    Load data for training hotdog detector model.

    Parameters
    ----------
    data_path : str
        The root directory where the dataset is stored.

    Returns
    -------
    image_file_paths : list of str
        A list of image file paths from the specified dataset.
    labels : list of int
        A list of corresponding labels for the image files.
    """
    image_file_paths = []
    labels = []

    # TODO

    return image_file_paths, labels


def plot_loss(history):
    """
    Plot the training and validation loss and accuracy.

    Parameters
    ----------
    history : keras.callbacks.History
      The history object returned by the `fit` method of a Keras model.

    Returns
    -------
    None
    """
    # TODO
