#read tiff files from the folder and print the shape of the arrays
import os
from tifffile import imread
import numpy as np
from tqdm import tqdm
import random
from scipy.ndimage import rotate
from tifffile import imwrite

path = f"/group/jug/Anirban/Datasets/AllenNeuron/all"
files = os.listdir(path)
files.sort()
minimum_shape = 512


def pad_crop(swc, minimum_shape):
    pad_x = ((minimum_shape - swc.shape[1]) //
             2) + 1 if swc.shape[1] < minimum_shape else 0
    pad_y = ((minimum_shape - swc.shape[2]) //
             2) + 1 if swc.shape[2] < minimum_shape else 0
    swc = np.pad(swc, ((0, 0), (pad_x, pad_x), (pad_y, pad_y)),
                 mode='constant',
                 constant_values=0)
    return swc


def pad_remove_channels(swc, channels):
    if swc.shape[0] < channels:
        swc = np.pad(swc, ((0, channels - swc.shape[0]), (0, 0), (0, 0)),
                     mode='constant',
                     constant_values=0)
    elif swc.shape[0] > channels:
        #center crop
        swc = swc[(swc.shape[0] - channels) // 2:(swc.shape[0] + channels) //
                  2, :, :]
    elif swc.shape[0] == channels:
        return swc
    return swc


def find_coordinates_crop(swc, minimum_shape=512, minimum_count=0.8):
    swc_max = np.max(swc, axis=0, keepdims=True)
    for _ in range(50):
        random_x = random.randint(0, swc_max.shape[1] - minimum_shape)
        random_y = random.randint(0, swc_max.shape[2] - minimum_shape)
        crop = swc_max[:, random_x:random_x + minimum_shape,
                       random_y:random_y + minimum_shape]
        count_nonzero = np.count_nonzero(crop)
        if count_nonzero > minimum_count * minimum_shape * minimum_shape:
            break
    swc = swc[:, random_x:random_x + minimum_shape,
              random_y:random_y + minimum_shape]
    return swc

def rotate_randomly(array):
    angle = random.uniform(60, 300)
    axes = [(0, 1), (0, 2), (1, 2)]
    axis = random.choice(axes)
    rotated_array = rotate(array,
                           angle,
                           axes=axis,
                           reshape=False,
                           mode='reflect')

    return rotated_array


for i in tqdm(range(len(files))):
    swc = imread(f"{path}/{files[i]}")
    swc = pad_crop(swc, minimum_shape)
    swc = find_coordinates_crop(swc)
    for batch in tqdm(range(9), leave=False):
        rotated_swcs = np.zeros_like(swc)
        rotated_swcs += rotate_randomly(swc)
        for _ in range(2):
            random_specimen = random.choice(files[:i] + files[i + 1:])
            new_swc = imread(f"{path}/{random_specimen}")
            new_swc = pad_remove_channels(new_swc, swc.shape[0])
            new_swc = pad_crop(new_swc, minimum_shape)
            new_swc = find_coordinates_crop(new_swc)
            rotated_swcs += rotate_randomly(new_swc)
        imwrite(f"/group/jug/Anirban/Datasets/AllenNeuron/rotated/{files[i].split('.')[0]}_{batch}.tif",rotated_swcs)
