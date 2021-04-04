import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.conv_utils import normalize_tuple
import utils


# def build_dataset(fldr, dims=None):
#     if dims is None:
#         dims = (244, 244)  # default to ResNet50 default input dimensions
#     else:
#         dims = normalize_tuple(value=dims, n=2, name="input_dims")
#     return tf.keras.preprocessing.image_dataset_from_directory(
#         directory=f"{data_fldr}/{fldr}",
#         labels="inferred",
#         label_mode="categorical",
#         color_mode="grayscale",
#         image_size=dims,
#     )
#
#
# def load(img_dims):
#     return (
#         build_dataset("train", img_dims),
#         build_dataset("validation", img_dims),
#         build_dataset("test", img_dims),
#     )


def load_data_gen(task, img_dims, seed=42, batch_size=32, shuffle=True, **kwargs):
    if task == 1:
        data_fldr = utils.data_fldr + "/imagenette_4class"
    elif task == 2:
        data_fldr = utils.data_fldr + "/imagewoof-320"
    elif task == 3:
        data_fldr = utils.data_fldr + "/imagewoof-320"

    if img_dims is None:
        img_dims = (244, 244)  # default to ResNet50 default input dimensions
    else:
        img_dims = normalize_tuple(value=img_dims, n=2, name="input_dims")
    if "rescale" not in kwargs.keys():
        kwargs["rescale"] = 1./255
    train_datagen = ImageDataGenerator(
        validation_split=0.1,
        **kwargs)
    val_datagen = ImageDataGenerator(validation_split=0.1, rescale=kwargs["rescale"])
    test_datagen = ImageDataGenerator(rescale=kwargs["rescale"])
    train_generator = train_datagen.flow_from_directory(
        directory=f"{data_fldr}/train",
        target_size=img_dims,
        color_mode="grayscale" if task == 1 else "rgb",
        batch_size=batch_size,
        class_mode="categorical",
        subset='training',
        seed=seed,
        shuffle=shuffle
    )
    validation_generator = val_datagen.flow_from_directory(
        directory=f"{data_fldr}/train",
        target_size=img_dims,
        color_mode="grayscale" if task == 1 else "rgb",
        batch_size=batch_size,
        class_mode="categorical",
        subset='validation',
        seed=seed,
        shuffle=shuffle
    )
    test_generator = test_datagen.flow_from_directory(
        directory=f"{data_fldr}/val",
        target_size=img_dims,
        color_mode="grayscale" if task == 1 else "rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    return train_generator, validation_generator, test_generator


