import os
username = "luken" if os.path.exists("C:/Users/luken") else "luke_"  # facilitate working on my personal and work PCs
project_fldr = f"C:/Users/{username}/Desktop/Dev/ee544-computer-vision"
data_fldr = f"{project_fldr}/data"


def convert_pb_to_h5(model_name, saved_models_path):
    import tensorflow as tf
    os.chdir(saved_models_path)
    model = tf.keras.models.load_model(model_name)
    model.save(f"../final_models/{model_name}.h5")
