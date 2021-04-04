import os

import numpy as np
import matplotlib.pyplot as plt

username = "luken" if os.path.exists("C:/Users/luken") else "luke_"  # facilitate working on my personal and work PCs
project_fldr = f"C:/Users/{username}/Desktop/Dev/ee544-computer-vision"
data_fldr = f"{project_fldr}/data"


def convert_pb_to_h5(model_name, saved_models_path):
    import tensorflow as tf
    os.chdir(saved_models_path)
    model = tf.keras.models.load_model(model_name)
    model.save(f"../final_models/{model_name}.h5")


def get_flops(model_h5_path):
    """Credit to @driedler https://github.com/tensorflow/tensorflow/issues/32809#issuecomment-577234513"""
    import tensorflow as tf
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)

            return flops.total_float_ops


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
    import itertools

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
