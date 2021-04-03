import os
import random
from datetime import datetime
import itertools

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils.conv_utils import normalize_tuple
from sklearn.metrics import classification_report, confusion_matrix

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


def build_model(input_dims, n_classes, include_rescale=False, activation_func=None, output_activation=None, dropout=None):
    if input_dims is None:
        input_dims = (244, 244)  # default to VGG-16 input dimensions
    else:
        input_dims = normalize_tuple(value=input_dims, n=2, name="input_dims")

    if activation_func is None:
        activation_func = "relu"
    if output_activation is None:
        output_activation = "relu"

    model = Sequential(name="VGG-liter")
    if include_rescale:
        # to support using tf.keras.preprocessing.image_dataset_from_directory which doesn't have rescale built in
        model.add(
            keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(input_dims[0], input_dims[1], 1))
        )
        model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))
    else:
        model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
                         input_shape=(input_dims[0], input_dims[1], 1),
                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))

    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu',
                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu',
                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))
    # model.add(Dropout(0.4, seed=SEED))
    model.add(Dense(n_classes, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))
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


def plot_confusion_matrix(cm, class_names):
    """
    Credit to the Tensorflow Team, from https://www.tensorflow.org/tensorboard/image_summaries
      Returns a matplotlib figure containing the plotted confusion matrix.

      Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
      """
    plt.figure(figsize=(8, 8))

    plt.style.use("dark_background")
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
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
    import json
    import load_data

    save = True
    experiment_name = "baseline_244"
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
    # early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

    callbacks = [model_checkpoint]



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

    # data_gen_options = {
    #     "rotation_range": 20,
    #     "width_shift_range": 0.2,
    #     "height_shift_range": 0.2,
    #     "horizontal_flip": True
    # }
    data_gen_options = {
        "rescale": 1. / 255
        # use this alone to *not* augment the dataset and train/test models quicker - for prototyping
    }

    reset_random()

    input_dims = 244
    (train_data, val_data, test_data) = load_data.load(input_dims)
    model = build_model(input_dims, n_classes=len(train_data.class_names), include_rescale=True)
    # (train_data, val_data, test_data) = load_data.load_data_gen(input_dims, seed=SEED, shuffle=True, **data_gen_options)
    # model = build_model(input_dims=input_dims, n_classes=len(train_data.class_indices.keys()))

    epochs = 10
    # learning_rate = 0.001
    # decay = learning_rate/epochs
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps=1500,
    #     decay_rate=0.96,
    #     staircase=True)
    # opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    opt = tf.keras.optimizers.Adam()
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

    model.load_weights(checkpoint_filepath)

    print_best_and_last(history)
    val_result = model.evaluate(val_data)
    print(f"Final validation results: {dict(zip(model.metrics_names, val_result))}")
    result = model.evaluate(test_data)
    print(f"Final test results: {dict(zip(model.metrics_names, result))}")
    plot_training(history)
    # print(f"FLOPS: {get_flops(model)}")
    Y_pred = model.predict(test_data)
    y_pred = np.argmax(Y_pred, axis=1)
    cm = confusion_matrix(test_data.classes, y_pred)
    print(classification_report(test_data.classes, y_pred, target_names=test_data.class_indices.keys()))
    plot_confusion_matrix(cm, test_data.class_indices.keys())

    if save:
        model.save(f"saved_models/{exp_stamp}")
        json.dump(history.history, open(f"history_logging/{exp_stamp}", 'w'))
