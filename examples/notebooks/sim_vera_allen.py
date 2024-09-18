from pathlib import Path
from microsim import schema as ms
from microsim.util import ortho_plot
import numpy as np
from tqdm import tqdm
import math
from tifffile import imwrite, imread
import argparse
import os

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--start", type=int, default=0)
    args = argparser.parse_args()
    start = args.start
    path = f"/group/jug/Anirban/Datasets/AllNeuron_Combined/GT_Volumes"
    files = os.listdir(path)
    files.sort()
    generated_images = []
    pinholes = np.linspace(0.5, 8.75, 22)[2::6]
    downscale = 4

    for images in tqdm(range(start, start + 100)):
        sample = imread(f"{path}/{files[images]}")
        #if the dim 0 of sample is not divisible by 4, then remove the last few rows
        if sample.shape[0] % 4 != 0:
            sample = sample[:-(sample.shape[0] % downscale)]
        sim = ms.Simulation.from_ground_truth(
            ground_truth=sample,
            scale=(0.02, 0.01, 0.01),
            output_space={"downscale": downscale},
            modality=ms.Confocal(pinhole_au=0.5),
            detector=ms.CameraCCD(qe=0.82,full_well=18000,read_noise=6,bit_depth=12,offset=100),
            settings=ms.Settings(random_seed=None, max_psf_radius_aus=4, np_backend="numpy", device='cpu'))
        gt = sim.ground_truth()
        all_images_noisy = []
        all_images_clean = []
        for _, au in enumerate(pinholes):
            sim = sim.model_copy(update=dict(modality=ms.Confocal(pinhole_au=au)))
            sim.run()
            optical_image = sim.optical_image(gt)

            #noisy image
            generated_images_noisy = sim.digital_image(optical_image, with_detector_noise=True)
            generated_images_noisy = np.array(generated_images_noisy)
            generated_image_noisy = generated_images_noisy[generated_images_noisy.shape[0] // 2]
            all_images_noisy.append(generated_image_noisy)

            #clean image
            generated_images_clean = sim.digital_image(optical_image, with_detector_noise=False)
            generated_images_clean = np.array(generated_images_clean)
            generated_image_clean = generated_images_clean[generated_images_clean.shape[0] // 2]
            all_images_clean.append(generated_image_clean)

        all_images_noisy = np.array(all_images_noisy)
        all_images_clean = np.array(all_images_clean)
        imwrite(f"/group/jug/Anirban/Datasets/AllNeuron_Combined/train_noisy/{files[images]}",all_images_noisy.astype(np.float32))
        imwrite(f"/group/jug/Anirban/Datasets/AllNeuron_Combined/train_clean/{files[images]}",all_images_clean.astype(np.float32))
