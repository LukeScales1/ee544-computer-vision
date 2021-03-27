import tensorflow as tf
from tensorflow.python.keras.utils.conv_utils import normalize_tuple
try:
    from utils import data_fldr
except ModuleNotFoundError:
    # in case you forget to mark my vgglite_1 folder as a source root
    from .utils import data_fldr


def build_dataset(fldr, dims=None):
    if dims is None:
        dims = (244, 244)  # default to VGG-16 input dimensions
    else:
        dims = normalize_tuple(value=dims, n=2, name="input_dims")
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory=f"{data_fldr}/{fldr}",
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        image_size=dims,
    )


def load(img_dims):
    return (
        build_dataset("train", img_dims),
        build_dataset("validation", img_dims),
        build_dataset("test", img_dims),
    )
