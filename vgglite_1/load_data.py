import tensorflow as tf
try:
    from utils import data_fldr
except ModuleNotFoundError:
    # in case you forget to mark my vgglite_1 folder as a source root
    from .utils import data_fldr


def build_dataset(fldr):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory=f"{data_fldr}/{fldr}",
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        image_size=(244, 244),
    )


def load():
    return (
        build_dataset("train"),
        build_dataset("validation"),
        build_dataset("test"),
    )
