import os
import random
from datetime import datetime
import itertools

import numpy as np
import tensorflow as tf
import torch
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, AveragePooling2D, Dropout
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout
# from tensorflow.keras.models import Sequential
# from tensorflow.python.keras.utils.conv_utils import normalize_tuple
from sklearn.metrics import classification_report, confusion_matrix

import utils
import load_data

import os
import utils


SEED = 42


# for reproducability
def reset_random():
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True


reset_random()


class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y


if __name__ == "__main__":
    from nvidia_unet import Unet

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    data_fldr = utils.data_fldr + "/oxford-iiit-pet"
    input_dir = f"{data_fldr}/images/"
    target_dir = f"{data_fldr}/annotations/trimaps/"
    img_size = (572, 572)  # original U-Net dims
    num_classes = 3
    batch_size = 1  # original U-Net batch size "...to make maximum use of GPU..." with large images

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )
    model = Unet()
    import random

    # Split our img paths into a training and a validation set
    val_samples = 1000
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    train_gen = OxfordPets(
        batch_size, img_size, train_input_img_paths, train_target_img_paths
    )
    val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    model.compile(optimizer="SGD", loss="sparse_categorical_crossentropy")  # SGD used in original paper

    callbacks = [
        keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
    ]

    # Train the model, doing validation at the end of each epoch.
    epochs = 15
    model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

