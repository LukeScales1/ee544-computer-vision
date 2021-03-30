import os
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils.conv_utils import normalize_tuple
import matplotlib.pyplot as plt

print(tf.__version__)

SEED = 42
# for reproducability
def reset_random():
  os.environ['PYTHONHASHSEED'] = str(SEED)
  np.random.seed(SEED)
  tf.random.set_seed(SEED)
  random.seed(SEED)
reset_random()


def build_model(input_dims, n_classes, include_rescale=False):
    if input_dims is None:
        input_dims = (244, 244)  # default to VGG-16 input dimensions
    else:
        input_dims = normalize_tuple(value=input_dims, n=2, name="input_dims")
    model = Sequential(name="VGG-liter")
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


def plot_training(h):
    acc = h.history['accuracy']
    val_acc = h.history['val_accuracy']

    loss = h.history['loss']
    val_loss = h.history['val_loss']

    # epochs_range = range(epochs)
    epochs_range = h.epoch

    plt.style.use("dark_background")

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range[0:len(acc)], acc, label='Training Accuracy')
    plt.plot(epochs_range[0:len(val_acc)], val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    # plt.setp(title, color='w')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range[0:len(loss)], loss, label='Training Loss')
    plt.plot(epochs_range[0:len(val_loss)], val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    # plt.setp(title, color='w')
    plt.show()


def print_best_and_last(h):
    print(f"Final training loss: {h.history['loss'][-1]}")
    print(f"Final training acc: {h.history['accuracy'][-1]}")
    print(f"Final validation loss: {h.history['val_loss'][-1]}")
    print(f"Final validation acc: {h.history['val_accuracy'][-1]}")
    print(f"Best training loss: {np.min(h.history['loss'])}")
    print(f"Best training acc: {np.max(h.history['accuracy'])}")
    print(f"Best validation loss: {np.min(h.history['val_loss'])}")
    print(f"Best validation acc: {np.max(h.history['val_accuracy'])}")


def get_flops(model):
    import tensorflow.python.keras.backend as K
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops


if __name__ == "__main__":
    import load_data
    save = True
    experiment_name = "early_stopping10"
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    callbacks = [early_stop_callback]

    # logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
    #                                                  histogram_freq=1,
    #                                                  profile_batch='500,520')

    # Failed attempts at getting tf 2.4.1 to work with latest CUDA drivers
    # gpu_devices = tf.config.list_physical_devices('GPU')
    # if not gpu_devices:
    #     raise ValueError('Cannot detect physical GPU device in TF')
    # device_name = tf.test.gpu_device_name()
    # if not device_name:
    #     raise SystemError('GPU device not found')
    # print('Found GPU at: {}'.format(device_name))

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

    epochs = 50
    input_dims = 32
    (train_data, val_data, test_data) = load_data.load_data_gen(input_dims, seed=SEED)

    model = build_model(input_dims=input_dims, n_classes=len(train_data.class_indices.keys()))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        # callbacks=[tboard_callback]
        callbacks=callbacks
    )
    print_best_and_last(history)
    result = model.evaluate(test_data)
    print(f"Final test results: {dict(zip(model.metrics_names, result))}")
    plot_training(history)
    # print(f"FLOPS: {get_flops(model)}")
    if save:
        model.save(f"saved_models/{experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
