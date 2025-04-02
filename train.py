import os
import numpy as np
import tensorflow as tf
import keras
from models import hotdog_model
from utils import load_data, plot_loss
from constants import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS


def train_model(model, X_train, y_train):

    # TODO
    
    pass


if __name__ == '__main__':
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)
    model = hotdog_model(input_shape)

    # TODO ...

    model, history = train_model(model)

    plot_loss(history)
