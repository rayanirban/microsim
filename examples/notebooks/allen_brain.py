from microsim.allen import Specimen
from microsim.util import ndview
from tifffile import imwrite
import numpy as np
import numpy as np
from scipy.ndimage import rotate
import random
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import argparse


def rotate_randomly(array):
    # Generate random angle between 50 and 300 degrees
    angle = random.uniform(50, 300)
    axes = [(0, 1), (0, 2), (1, 2)]
    axis = random.choice(axes)
    rotated_array = rotate(array,angle,axes=axis,reshape=False,mode='reflect')
    return rotated_array


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--start", type=int)
    args = argparser.parse_args()
    start = args.start

    specimen_ids = np.loadtxt("specimen_ids.csv", delimiter=",", dtype=int)

    for k in tqdm(range(len(specimen_ids))):
        spec = Specimen.fetch(specimen_ids[k])
        swc = spec.neuron_reconstructions[0].create_truth_space(voxel_size=2,scale_factor=1)
        #add gaussian blur to the line
        #for i in range(swc.shape[0]):
        #    swc[i] = gaussian_filter(swc[i], sigma=1.0)
        #rotate the swc 3 times and add the results to new array
        # for batch in range(7):
        #     rotated_swcs = np.zeros_like(swc)
        #     for i in range(3):
        #         rotated_swcs += rotate_randomly(swc)
        #imwrite(f"/group/jug/Anirban/Datasets/AllenNeuron/all/{specimen_ids[k]}.tif",swc.astype(np.float32))