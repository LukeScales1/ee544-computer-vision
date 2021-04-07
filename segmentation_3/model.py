import os
import random
from datetime import datetime
import itertools

import numpy as np
import tensorflow as tf
import torch
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, AveragePooling2D, Dropout
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout
# from tensorflow.keras.models import Sequential
# from tensorflow.python.keras.utils.conv_utils import normalize_tuple
from sklearn.metrics import classification_report, confusion_matrix

import utils
import load_data

import os
import utils


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


class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y


def get_xception_style_unet(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


def conv_block(input_tensor, num_filters, padding="same"):
     encoder = layers.Conv2D(num_filters, (3, 3), padding=padding, kernel_initializer='he_normal')(input_tensor)
     encoder = layers.BatchNormalization()(encoder)
     encoder = layers.Activation('relu')(encoder)
     encoder = layers.Conv2D(num_filters, (3, 3), padding=padding, kernel_initializer='he_normal')(encoder)
     encoder = layers.BatchNormalization()(encoder)
     encoder = layers.Activation('relu')(encoder)
     return encoder


def encoder_block(input_tensor, num_filters, incl_dropout=False):
    encoder = conv_block(input_tensor, num_filters)
    if incl_dropout:
        encoder = layers.Dropout(0.5)(encoder)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters, padding="same"):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding=padding, kernel_initializer='he_normal')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding=padding, kernel_initializer='he_normal')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding=padding, kernel_initializer='he_normal')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder


def get_pauls_model(img_dims, num_classes):
    """Modified from Paul Whelan's code provided in EE544 course notes"""
    inputs = layers.Input(shape=img_dims + (3,))
    x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    # 256
    encoder0_pool, encoder0 = encoder_block(x, 32)
    # 128
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    # 64
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    # 32
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    # 16
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512, incl_dropout=True)
    # 8
    center = conv_block(encoder4_pool, 1024)
    center = layers.Dropout(0.5)(center)
    # center
    decoder4 = decoder_block(center, encoder4, 512)
    # 16
    decoder3 = decoder_block(decoder4, encoder3, 256)
    # 32
    decoder2 = decoder_block(decoder3, encoder2, 128)
    # 64
    decoder1 = decoder_block(decoder2, encoder1, 64)
    # 128
    decoder0 = decoder_block(decoder1, encoder0, 32)
    # 256
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(decoder0)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_charlychiu_model(img_dims):
    concat_axis = 3

    inputs = Input(shape=img_dims + (3,))
    conv1_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv1_1')(inputs)
    conv1_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv1_2')(conv1_1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), name='maxpooling_1')(conv1_2)

    conv2_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv2_1')(pool1)
    conv2_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv2_2')(conv2_1)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), name='maxpooling_2')(conv2_2)

    conv3_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv3_1')(pool2)
    conv3_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv3_2')(conv3_1)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2), name='maxpooling_3')(conv3_2)

    conv4_1 = layers.Conv2D(512, (3, 3), activation='relu', padding='valid', name='conv4_1')(pool3)
    conv4_2 = layers.Conv2D(512, (3, 3), activation='relu', padding='valid', name='conv4_2')(conv4_1)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2), name='maxpooling_4')(conv4_2)

    conv5_1 = layers.Conv2D(1024, (3, 3), activation='relu', padding='valid', name='conv5_1')(pool4)
    conv5_2 = layers.Conv2D(1024, (3, 3), activation='relu', padding='valid', name='conv5_2')(conv5_1)

    upsampling1 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='valid', name='upsampling1')(conv5_2)
    crop_conv4_2 = layers.Cropping2D(cropping=((4, 4), (4, 4)), name='cropped_conv4_2')(conv4_2)
    up6 = layers.concatenate([upsampling1, crop_conv4_2], axis=concat_axis, name='skip_connection1')
    conv6_1 = layers.Conv2D(512, (3, 3), activation='relu', padding='valid', name='conv6_1')(up6)
    conv6_2 = layers.Conv2D(512, (3, 3), activation='relu', padding='valid', name='conv6_2')(conv6_1)

    upsampling2 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='valid', name='upsampling2')(conv6_2)
    crop_conv3_2 = layers.Cropping2D(cropping=((16, 16), (16, 16)), name='cropped_conv3_2')(conv3_2)
    up7 = layers.concatenate([upsampling2, crop_conv3_2], axis=concat_axis, name='skip_connection2')
    conv7_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv7_1')(up7)
    conv7_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv7_2')(conv7_1)

    upsampling3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='valid', name='upsampling3')(conv7_2)
    crop_conv2_2 = layers.Cropping2D(cropping=((40, 40), (40, 40)), name='cropped_conv2_2')(conv2_2)
    up8 = layers.concatenate([upsampling3, crop_conv2_2], axis=concat_axis, name='skip_connection3')
    conv8_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv8_1')(up8)
    conv8_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv8_2')(conv8_1)

    upsampling4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid', name='upsampling4')(conv8_2)
    crop_conv1_2 = layers.Cropping2D(cropping=((88, 88), (88, 88)), name='cropped_conv1_2')(conv1_2)
    up9 = layers.concatenate([upsampling4, crop_conv1_2], axis=concat_axis, name='skip_connection4')
    conv9_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv9_1')(up9)
    conv9_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv9_2')(conv9_1)

    conv10 = layers.Conv2D(3, (1, 1), activation='softmax', name='conv10')(conv9_2)
    model = Model(inputs=[inputs], outputs=[conv10])
    return model


if __name__ == "__main__":
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras import layers, Model

    # experiment_name = "xception_style_unet_15"
    experiment_name = "unet_init"

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    data_fldr = utils.data_fldr + "/oxford-iiit-pet"
    input_dir = f"{data_fldr}/images/"
    target_dir = f"{data_fldr}/annotations/trimaps/"

    epochs = 15
    img_size = (128, 128)  # original U-Net dims was 572, 572 but hitting GPU OOM without tiling strategy
    num_classes = 3
    batch_size = 32  # original U-Net batch size "...to make maximum use of GPU..." with large images
    PADDING = "same"  # original U-net "unpadded" = "valid" but using "same" with input size change

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    # # train first 1000 images
    # input_img_paths = input_img_paths[0:1000]
    # target_img_paths = target_img_paths[0:1000]


    # # modified from Paul Whelan's EE544 course notes:
    # def conv_block(input_tensor, num_filters):
    #     encoder = layers.Conv2D(num_filters, (3, 3), padding=PADDING)(input_tensor)
    #     encoder = layers.Activation('relu')(encoder)
    #     encoder = layers.Conv2D(num_filters, (3, 3), padding=PADDING)(encoder)
    #     encoder = layers.Activation('relu')(encoder)
    #     return encoder
    #
    #
    # def encoder_block(input_tensor, num_filters, incl_dropout=False):
    #     encoder = conv_block(input_tensor, num_filters)
    #     if incl_dropout:
    #         encoder = layers.Dropout(0.5)(encoder)
    #     encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    #     return encoder_pool, encoder
    #
    #
    # def center_block(input_tensor, num_filters):
    #     # DROPOUT dropout_param { dropout_ratio: 0.5 } include: { phase: TRAIN }}
    #     # POOLING  pooling_param { pool: MAX kernel_size: 2 stride: 2 } }
    #     # ---- prev 512 conv block with dropout ^ ----
    #     # CONVOLUTION  { num_output: 1024 pad: 0 ...
    #     # RELU }
    #     # CONVOLUTION  { num_output: 1024 pad: 0 ...
    #     # RELU }
    #     # DROPOUT dropout_param { dropout_ratio: 0.5 } include: { phase: TRAIN }}
    #     # center = layers.Dropout(0.5)(input_tensor)
    #     center = conv_block(input_tensor, num_filters)
    #     center = layers.Dropout(0.5)(center)
    #     return center
    #
    #
    # def decoder_block(input_tensor, concat_tensor, num_filters):
    #     # decoder = layers.UpSampling2D(2)(input_tensor)
    #     decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding=PADDING)(input_tensor)
    #     # decoder = layers.Activation('relu')(decoder)
    #     decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    #     # decoder = layers.Activation('relu')(decoder)  # no relu after concat in original architecture
    #     decoder = layers.Conv2D(num_filters, (3, 3), padding=PADDING)(decoder)
    #     decoder = layers.Activation('relu')(decoder)
    #     decoder = layers.Conv2D(num_filters, (3, 3), padding=PADDING)(decoder)
    #     decoder = layers.Activation('relu')(decoder)
    #     return decoder


    def get_model(img_size, num_classes):
        inputs = keras.Input(shape=img_size + (3,))
        x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)

        # # inputs = layers.Conv2D(32, 3, strides=2, padding=PADDING)(inputs)
        # # inputs = layers.Activation("relu")(inputs)
        # 
        # ### [First half of the network: downsampling inputs] ###
        # 
        # # Entry block
        # # x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        # # x = layers.Activation("relu")(x)
        # encoder0_pool, encoder0 = encoder_block(inputs, 64)
        # encoder1_pool, encoder1 = encoder_block(encoder0_pool, 128)
        # encoder2_pool, encoder2 = encoder_block(encoder1_pool, 256)
        # encoder3_pool, encoder3 = encoder_block(encoder2_pool, 512, incl_dropout=True)
        # # encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
        # 
        # # encoder0_pool, encoder0 = encoder_block(inputs, 32)
        # # # 128
        # # encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
        # # # 64
        # # encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
        # # # 32
        # # encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
        # # # 16
        # # encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512, incl_dropout=True)
        # 
        # center = center_block(encoder3_pool, 1024)
        # 
        # # decoder4 = decoder_block(center, encoder4, 512)
        # # # 16
        # # decoder3 = decoder_block(decoder4, encoder3, 256)
        # # # 32
        # # decoder2 = decoder_block(decoder3, encoder2, 128)
        # # # 64
        # # decoder1 = decoder_block(decoder2, encoder1, 64)
        # # # 128
        # # decoder0 = decoder_block(decoder1, encoder0, 32)
        # 
        # decoder3 = decoder_block(center, encoder2, 512)
        # decoder2 = decoder_block(decoder3, encoder1, 256)
        # decoder1 = decoder_block(decoder2, encoder0, 128)
        # decoder0 = decoder_block(decoder1, encoder0, 64)
        # # decoder0 = decoder_block(decoder1, encoder0, 32)
        # # 256
        # outputs = conv_block(decoder1, 64)(decoder1)
        # outputs = layers.Conv2D(3, (1, 1), activation='softmax')(decoder0)
        # 
        # 
        # # previous_block_activation = x  # Set aside residual
        # #
        # # # Blocks 1, 2, 3 are identical apart from the feature depth.
        # # for filters in [64, 128, 256]:
        # #     x = layers.Conv2D(filters, 3, padding="same")(x)
        # #     x = layers.Activation("relu")(x)
        # #     x = layers.Conv2D(filters, 3, padding="same")(x)
        # #     x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        # #
        # #     # Project residual
        # #     residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
        # #         previous_block_activation
        # #     )
        # #     x = layers.add([x, residual])  # Add back residual
        # #     previous_block_activation = x  # Set aside next residual
        # #
        # # ### [Second half of the network: upsampling inputs] ###
        # #
        # # for filters in [256, 128, 64, 32]:
        # #     x = layers.Activation("relu")(x)
        # #     x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        # #     x = layers.BatchNormalization()(x)
        # #
        # #     x = layers.Activation("relu")(x)
        # #     x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        # #     x = layers.BatchNormalization()(x)
        # #
        # #     x = layers.UpSampling2D(2)(x)
        # #
        # #     # Project residual
        # #     residual = layers.UpSampling2D(2)(previous_block_activation)
        # #     residual = layers.Conv2D(filters, 1, padding="same")(residual)
        # #     x = layers.add([x, residual])  # Add back residual
        # #     previous_block_activation = x  # Set aside next residual
        # 
        # # Add a per-pixel classification layer
        # # outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
        # 
        # # Define the model
        # model = keras.Model(inputs, outputs)
        # print(outputs.summary())
        # inputs = Input(img_size)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(x)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(pool1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(pool2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(pool3)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(conv4)
        drop4 = layers.Dropout(0.5)(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = layers.Conv2D(1024, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(pool4)
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(conv5)
        drop5 = layers.Dropout(0.5)(conv5)

        up6 = layers.Conv2D(512, 2, activation='relu', padding=PADDING, kernel_initializer='he_normal')(
            layers.UpSampling2D(size=(2, 2))(drop5))
        merge6 = layers.concatenate([drop4, up6], axis=3)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(merge6)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(conv6)

        up7 = layers.Conv2D(256, 2, activation='relu', padding=PADDING, kernel_initializer='he_normal')(
            layers.UpSampling2D(size=(2, 2))(conv6))
        merge7 = layers.concatenate([conv3, up7], axis=3)
        conv7 = layers.Conv2D(256, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(merge7)
        conv7 = layers.Conv2D(256, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(conv7)

        up8 = layers.Conv2D(128, 2, activation='relu', padding=PADDING, kernel_initializer='he_normal')(
            layers.UpSampling2D(size=(2, 2))(conv7))
        merge8 = layers.concatenate([conv2, up8], axis=3)
        conv8 = layers.Conv2D(128, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(merge8)
        conv8 = layers.Conv2D(128, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(conv8)

        up9 = layers.Conv2D(64, 2, activation='relu', padding=PADDING, kernel_initializer='he_normal')(
            layers.UpSampling2D(size=(2, 2))(conv8))
        merge9 = layers.concatenate([conv1, up9], axis=3)
        conv9 = layers.Conv2D(64, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(merge9)
        conv9 = layers.Conv2D(64, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(conv9)
        conv9 = layers.Conv2D(2, 3, activation='relu', padding=PADDING, kernel_initializer='he_normal')(conv9)
        conv10 = layers.Conv2D(num_classes, 1, activation='softmax')(conv9)

        model = tf.keras.Model(inputs=inputs, outputs=conv10)
        return model


    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    # model = get_model(img_size, num_classes)
    # model = get_charlychiu_model(img_size)
    model = get_pauls_model(img_size, num_classes)
    # model = get_xception_style_unet(img_size, num_classes)

    # model = Unet()
    import random

    # Split our img paths into a training, validation & sets
    val_samples = 1000
    test_samples = 500
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-(val_samples+test_samples)]
    train_target_img_paths = target_img_paths[:-(val_samples+test_samples)]
    val_input_img_paths = input_img_paths[-(val_samples+test_samples):-test_samples]
    val_target_img_paths = target_img_paths[-(val_samples+test_samples):-test_samples]

    test_input_img_paths = input_img_paths[-test_samples:]
    test_target_img_paths = target_img_paths[-test_samples:]

    # Instantiate data Sequences for each split
    train_gen = OxfordPets(
        batch_size, img_size, train_input_img_paths, train_target_img_paths
    )
    val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    # SGD used in original paper with momentum of 0.99
    # so that updates are determined by "...a large number of previously seen training examples..."
    opt = keras.optimizers.SGD(momentum=0.99)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy",
                  metrics=[
                      "accuracy",
                  ])

    callbacks = [
        keras.callbacks.ModelCheckpoint(f"{experiment_name}.h5", save_best_only=True)
    ]

    # Train the model, doing validation at the end of each epoch.

    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
    model.save(f"saved_models/{experiment_name}.h5")

    utils.plot_training(history)

