import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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


def load_data_gen(img_dims, seed=42, batch_size=32, **kwargs):
    if img_dims is None:
        img_dims = (244, 244)  # default to VGG-16 input dimensions
    else:
        img_dims = normalize_tuple(value=img_dims, n=2, name="input_dims")
    if "rescale" not in kwargs.keys():
        kwargs["rescale"] = 1./255
    train_datagen = ImageDataGenerator(**kwargs)
    test_datagen = ImageDataGenerator(rescale=kwargs["rescale"])
    train_generator = train_datagen.flow_from_directory(
        directory=f"{data_fldr}/train",
        target_size=img_dims,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        seed=seed,
    )
    validation_generator = test_datagen.flow_from_directory(
        directory=f"{data_fldr}/train",
        target_size=img_dims,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        seed=seed,
    )
    test_generator = test_datagen.flow_from_directory(
        directory=f"{data_fldr}/train",
        target_size=img_dims,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
    )
    return train_generator, validation_generator, test_generator


