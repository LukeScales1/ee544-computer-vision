import os
import random
from datetime import datetime
import itertools

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils.conv_utils import normalize_tuple
from sklearn.metrics import classification_report, confusion_matrix

# from keras_flops import get_flops

import utils
import load_data

print(tf.__version__)

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


def baseline_model(input_dims, n_classes, include_rescale=False):
    if input_dims is None:
        input_dims = (244, 244)  # default to VGG-16 input dimensions
    else:
        input_dims = normalize_tuple(value=input_dims, n=2, name="input_dims")
    model = Sequential(name="VGG-liter-baseline")
    if include_rescale:
        # to support using tf.keras.preprocessing.image_dataset_from_directory which doesn't have rescale built in
        model.add(
            keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(input_dims[0], input_dims[1], 1))
        )
        model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    else:
        model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
                         input_shape=(input_dims[0], input_dims[1], 1)))

    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(n_classes, activation='relu'))
    return model


def build_model(input_dims, n_classes, include_rescale=False, activation_func=None, output_activation=None, 
                dropout=True, batch_norm=True):
    if input_dims is None:
        input_dims = (244, 244)  # default to VGG-16 input dimensions
    else:
        input_dims = normalize_tuple(value=input_dims, n=2, name="input_dims")

    if activation_func is None:
        activation_func = "relu"
    if output_activation is None:
        output_activation = "relu"

    model = Sequential(name="VGG-lite")
    if include_rescale:
        # to support using tf.keras.preprocessing.image_dataset_from_directory which doesn't have rescale built in
        model.add(
            keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(input_dims[0], input_dims[1], 1))
        )
        model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation=activation_func,
                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))
    else:
        model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation=activation_func,
                         input_shape=(input_dims[0], input_dims[1], 1),
                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))
    if batch_norm:
        model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation=activation_func,
                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))
    if batch_norm:
        model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(pool_size=(2, 2)))
    if dropout:
        model.add(Dropout(0.25, seed=SEED))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation=activation_func,
                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))
    if batch_norm:
        model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation=activation_func,
                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))
    if batch_norm:
        model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=512, activation=activation_func, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))
    if batch_norm:
        model.add(BatchNormalization(axis=-1))
    if dropout:
        model.add(Dropout(0.5, seed=SEED))
    model.add(Dense(n_classes, activation=output_activation, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))
    return model


if __name__ == "__main__":
    import json

    # EXPERIMENT CONFIG #
    save = False
    activation = 'elu'
    output_activation = 'softmax'
    use_dropout = True
    use_batch_norm = True
    experiment_name = "image_aug_lrschedule_elu_softmax_dropout0.25-0.5_batchnorm"

    epochs = 5
    # learning_rate = 0.001
    # decay = learning_rate/epochs
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=1500,
        decay_rate=0.96,
        staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # opt = tf.keras.optimizers.Adam()

    exp_stamp = f"{experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    checkpoint_filepath = f'model_checkpoints/{exp_stamp}'

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    # logdir = f"logs/{exp_stamp}"
    # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    callbacks = []

    # tf.config.set_soft_device_placement(True)
    # tf.debugging.set_log_device_placement(True)

    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     # Restrict TensorFlow to only use the first GPU
    #     try:
    #         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    #     except RuntimeError as e:
    #         # Visible devices must be set before GPUs have been initialized
    #         print(e)

    # if gpu_devices:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpu_devices:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpu_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)

    data_gen_options = {
        "rotation_range": 30,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "shear_range": 0.2,
        "zoom_range": 0.1,
        "horizontal_flip": True
    }
    # data_gen_options = {
    #     "rescale": 1. / 255
    #     # use this alone to *not* augment the dataset and train/test models quicker - for prototyping
    # }

    reset_random()

    input_dims = 32
    # (train_data, val_data, test_data) = load_data.load(input_dims)
    # model = build_model(input_dims, n_classes=len(train_data.class_names), include_rescale=True)
    (train_data, val_data, test_data) = load_data.load_data_gen(task=1, img_dims=input_dims, seed=SEED, shuffle=True, **data_gen_options)
    model = build_model(input_dims=input_dims, n_classes=len(train_data.class_indices.keys()),
                        activation_func=activation, output_activation=output_activation)

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    reset_random()

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        # callbacks=[tboard_callback]
        callbacks=callbacks
    )

    # model.load_weights(checkpoint_filepath)

    print(experiment_name)
    utils.print_best_and_last(history)
    val_result = model.evaluate(val_data)
    print(f"Final validation results: {dict(zip(model.metrics_names, val_result))}")
    result = model.evaluate(test_data)
    print(f"Final test results: {dict(zip(model.metrics_names, result))}")
    utils.plot_training(history)
    # print(f"FLOPS: {get_flops(model)}")
    Y_pred = model.predict(test_data)
    y_pred = np.argmax(Y_pred, axis=1)
    cm = confusion_matrix(test_data.classes, y_pred)
    print(classification_report(test_data.classes, y_pred, target_names=test_data.class_indices.keys()))
    utils.plot_confusion_matrix(cm, test_data.class_indices.keys())

    if save:
        model.save(f"saved_models/{exp_stamp}")
        json.dump(history.history, open(f"history_logging/{exp_stamp}", 'w'))
