import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.conv_utils import normalize_tuple
from tensorflow.keras.applications.resnet50 import preprocess_input

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

    if img_dims is None:
        img_dims = (244, 244)  # default to ResNet50 default input dimensions
    else:
        img_dims = normalize_tuple(value=img_dims, n=2, name="input_dims")

    train_generator, validation_generator, test_generator = None, None, None

    # def get_default_flow_kwargs(ddir, color="grayscale", **dkwargs):
    #     return {
    #         "directory": ddir,
    #         "target_size": img_dims,
    #         "color_mode": color,
    #         "batch_size": batch_size,
    #         "class_mode": "categorical",
    #         "seed": seed,
    #         **dkwargs
    #     }

    if task == 1:
        if "rescale" not in kwargs.keys():
            kwargs["rescale"] = 1. / 255
        data_fldr = utils.data_fldr + "/imagenette_4class"
        train_datagen = ImageDataGenerator(**kwargs)
        train_generator = train_datagen.flow_from_directory(
            f"{data_fldr}/train",
            batch_size=batch_size,
            target_size=img_dims,
            color_mode="grayscale",
            shuffle=True,
            seed=seed,
            class_mode='categorical')
        test_datagen = ImageDataGenerator(rescale=kwargs["rescale"])
        validation_generator = test_datagen.flow_from_directory(
            f"{data_fldr}/validation",
            batch_size=batch_size,
            target_size=img_dims,
            color_mode="grayscale",
            shuffle=True,
            seed=seed,
            class_mode='categorical')
        test_datagen = ImageDataGenerator(rescale=kwargs["rescale"])
        test_generator = test_datagen.flow_from_directory(
            f"{data_fldr}/test",
            batch_size=batch_size,
            target_size=img_dims,
            color_mode="grayscale",
            shuffle=False,
            class_mode='categorical')

    elif task == 2:
        data_fldr = utils.data_fldr + "/imagewoof-320"
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.1, **kwargs)
        train_generator = train_datagen.flow_from_directory(
            f"{data_fldr}/train",
            batch_size=batch_size,
            target_size=img_dims,
            shuffle=True,
            seed=seed,
            subset="training",
            class_mode='categorical')
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.1)
        validation_generator = test_datagen.flow_from_directory(
            f"{data_fldr}/train",
            batch_size=batch_size,
            target_size=img_dims,
            shuffle=False,
            seed=seed,
            subset="validation",
            class_mode='categorical')
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_generator = test_datagen.flow_from_directory(
            f"{data_fldr}/val",
            batch_size=batch_size,
            target_size=img_dims,
            shuffle=False,
            class_mode='categorical')

    elif task == 3:
        data_fldr = utils.data_fldr + "/oxford-iit-pet"

    return train_generator, validation_generator, test_generator


