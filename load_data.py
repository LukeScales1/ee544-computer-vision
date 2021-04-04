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

    if img_dims is None:
        img_dims = (244, 244)  # default to ResNet50 default input dimensions
    else:
        img_dims = normalize_tuple(value=img_dims, n=2, name="input_dims")
    if "rescale" not in kwargs.keys():
        kwargs["rescale"] = 1./255

    val_kwargs = {"rescale": 1./255}

    def get_default_flow_kwargs(ddir, color="grayscale", **dkwargs):
        return {
            "directory": ddir,
            "target_size": img_dims,
            "color_mode": color,
            "batch_size": batch_size,
            "class_mode": "categorical",
            "seed": seed,
            **dkwargs
        }

    if task == 1:
        data_fldr = utils.data_fldr + "/imagenette_4class"
        train_dir = f"{data_fldr}/train"
        train_flow_kwargs = get_default_flow_kwargs(train_dir, shuffle=shuffle)

        val_dir = f"{data_fldr}/validation"
        val_flow_kwargs = get_default_flow_kwargs(val_dir, shuffle=shuffle)

        test_dir = f"{data_fldr}/test"
        test_flow_kwargs = get_default_flow_kwargs(test_dir, shuffle=False)

    elif task == 2:
        data_fldr = utils.data_fldr + "/imagewoof-320"
        kwargs["validation_split"] = 0.1
        val_kwargs["validation_split"] = 0.1

        train_dir = f"{data_fldr}/train"
        train_flow_kwargs = get_default_flow_kwargs(train_dir, color="rgb", shuffle=shuffle)

        val_dir = f"{data_fldr}/train"
        val_flow_kwargs = get_default_flow_kwargs(val_dir, color="rgb", shuffle=shuffle)

        test_dir = f"{data_fldr}/val"
        test_flow_kwargs = get_default_flow_kwargs(test_dir, color="rgb", shuffle=False)

    elif task == 3:
        data_fldr = utils.data_fldr + "/imagewoof-320"

    train_datagen = ImageDataGenerator(**kwargs)
    val_datagen = ImageDataGenerator(**val_kwargs)
    test_datagen = ImageDataGenerator(rescale=kwargs["rescale"])
    train_generator = train_datagen.flow_from_directory(**train_flow_kwargs)
    validation_generator = val_datagen.flow_from_directory(**val_flow_kwargs)
    test_generator = test_datagen.flow_from_directory(**test_flow_kwargs)
    return train_generator, validation_generator, test_generator


