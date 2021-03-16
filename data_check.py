import os

import numpy as np
from skimage import io
import itertools
import multiprocessing as mp


def check_image_sizes(arguments):
    project_fldr = 'C:/Users/luken/Desktop/Dev/ee544-computer-vision'
    data_root = f"{project_fldr}/data/imagenette_4class"
    print("checking")
    shapes = []
    fldr = arguments[0]
    class_name = arguments[1]
    print(f"processing for {fldr}/{class_name}")
    files = [f for f in os.listdir(f"{data_root}/{fldr}/{class_name}") if os.path.isfile(os.path.join(f"{data_root}/{fldr}/{class_name}", f))]
    print(f"{len(files)} images to check")
    for f in files[1:]:
        img = io.imread(os.path.join(f"{data_root}/{fldr}/{class_name}", f))
        shapes.append(img.shape)
    print(f"{fldr}-{class_name}, unique shapes are: {np.unique(shapes)}")
    print(f"smallest shape in {fldr}-{class_name}: {np.min}")


if __name__ == "__main__":
    options = list(
        itertools.product(["train", "test", "validation"], ["church", "garbage_truck", "gas_pump", "parachute"]))
    print(options)
    if len(options) <= mp.cpu_count():
        n_workers = len(options)
    else:
        n_workers = mp.cpu_count()
    print(f"num workers = {n_workers}")
    sizes = mp.Array()
    with mp.Pool(n_workers) as p:
        p.map(check_image_sizes, options)
