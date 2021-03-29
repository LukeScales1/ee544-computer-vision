import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils.conv_utils import normalize_tuple


def build_model(input_dims, n_clases):
    if input_dims is None:
        input_dims = (244, 244)  # default to VGG-16 input dimensions
    else:
        input_dims = normalize_tuple(value=input_dims, n=2, name="input_dims")
    model = Sequential(name="VGG-liter")
    model.add(keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(input_dims[0], input_dims[1], 1))),
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(n_clases, activation='relu'))
    return model

