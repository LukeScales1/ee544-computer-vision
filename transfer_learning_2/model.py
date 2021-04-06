import os
import random
from datetime import datetime
import itertools

import numpy as np
import tensorflow as tf
import torch
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, AveragePooling2D, Dropout, BatchNormalization
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout
# from tensorflow.keras.models import Sequential
# from tensorflow.python.keras.utils.conv_utils import normalize_tuple
from sklearn.metrics import classification_report, confusion_matrix

import utils
import load_data


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


if __name__ == "__main__":
    exp_name = "undersamp_long_run_small_lr"

    # EXPERIMENT CONFIG #
    # save = False
    # activation = 'elu'
    # output_activation = 'softmax'
    # use_dropout = True
    # use_batch_norm = True
    # experiment_name = "image_aug_lrschedule_elu_softmax_dropout0.25-0.5_batchnorm"
    # epochs = 30
    # # learning_rate = 0.001
    # # decay = learning_rate/epochs
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps=1500,
    #     decay_rate=0.96,
    #     staircase=True)
    # opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # # opt = tf.keras.optimizers.Adam()
    #
    # exp_stamp = f"{experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # checkpoint_filepath = f'model_checkpoints/{exp_stamp}'
    #
    # model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_weights_only=True,
    #     monitor='val_loss',
    #     mode='min',
    #     save_best_only=True)
    # # logdir = f"logs/{exp_stamp}"
    # # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    # early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    #
    # callbacks = []
    #
    # data_gen_options = {
    #     "rotation_range": 30,
    #     "width_shift_range": 0.2,
    #     "height_shift_range": 0.2,
    #     "shear_range": 0.2,
    #     "zoom_range": 0.1,
    #     "horizontal_flip": True
    # }

    # Step 1 - train new classifier with frozen base model
    resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    resnet.trainable = False

    # following Adrian Rosebock's example @
    # https://www.pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/

    class_model = resnet.output
    class_model = BatchNormalization()(class_model)
    class_model = AveragePooling2D(pool_size=(7, 7))(class_model)
    class_model = Flatten(name="flatten")(class_model)
    class_model = Dense(256, activation="relu")(class_model)
    class_model = BatchNormalization()(class_model)
    class_model = Dropout(0.5)(class_model)
    class_model = Dense(10, activation="softmax")(class_model)

    model = Model(inputs=resnet.input, outputs=class_model)

    # load data with resnet preprocessing function
    # (train_data, val_data, test_data) = load_data.load_data_gen(task=2, img_dims=244)
    # second run - balancing classes via weights and augmenting data to fight overfitting, updating batch size:
    BATCH_SIZE = 124

    # third run - balanced dataset by undersampling, only
    # BATCH_SIZE = 32
    # class_weight = {
    #     0: 1.,
    #     1: 1.,
    #     2: 1.,
    #     3: 1.,
    #     4: 1.73,
    #     5: 1.,
    #     6: 1.,
    #     7: 1.,
    #     8: 1.,
    #     9: 1.,
    # }
    data_gen_params = {
        "rotation_range": 40,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "shear_range": 0.2,
        "zoom_range": 0.2,
        "horizontal_flip": True,
    }
    (train_data, val_data, test_data) = load_data.load_data_gen(task=2, img_dims=244, batch_size=BATCH_SIZE, **data_gen_params)

    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

    opt = tf.keras.optimizers.Adam(lr=2e-5)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=100,
        # class_weight=class_weight,
        callbacks=[early_stop_callback]
    )
    # save initial model and log results
    val_result = model.evaluate(val_data)
    print(f"Final validation results: {dict(zip(model.metrics_names, val_result))}")
    result = model.evaluate(test_data)
    print(f"Final test results: {dict(zip(model.metrics_names, result))}")
    utils.plot_training(history)
    Y_pred = model.predict(test_data)
    y_pred = np.argmax(Y_pred, axis=1)
    cm = confusion_matrix(test_data.classes, y_pred)
    print(classification_report(test_data.classes, y_pred, target_names=test_data.class_indices.keys()))
    utils.plot_confusion_matrix(cm, test_data.class_indices.keys())

    model.save(f"saved_models/initial_{exp_name}")

    # Step 2 - unfreeze res5c block and retrain final layers and added classifier
    resnet.trainable = True

    res5c = 0
    trainable_layer_count = 0
    for layer in resnet.layers:
        if layer.name[0:5] == "conv5":
            res5c = 1
        if not res5c:
            layer.trainable = False
        if layer.trainable:
            trainable_layer_count += 1

    print(f"{trainable_layer_count} trainable layers")

    # as per Chollet's advice, v small learning rate when fine-tuning
    # https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb

    # model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-5),
    #               loss=tf.keras.losses.CategoricalCrossentropy(),
    #               metrics=['accuracy'])

    # second pass - using Adam as less affected by class_weights changing range of the loss
    # ref: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#train_a_model_with_class_weights
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    train_data.reset()
    val_data.reset()

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=100,
        # class_weight=class_weight,
        callbacks=[early_stop_callback]
    )

    test_data.reset()

    # save fine-tuned model and log results
    val_result = model.evaluate(val_data)
    print(f"Final validation results: {dict(zip(model.metrics_names, val_result))}")
    result = model.evaluate(test_data)
    print(f"Final test results: {dict(zip(model.metrics_names, result))}")
    utils.plot_training(history)
    Y_pred = model.predict(test_data)
    y_pred = np.argmax(Y_pred, axis=1)
    cm = confusion_matrix(test_data.classes, y_pred)
    print(classification_report(test_data.classes, y_pred, target_names=test_data.class_indices.keys()))
    utils.plot_confusion_matrix(cm, test_data.class_indices.keys())

    model.save(f"saved_models/fine-tuned_{exp_name}")
