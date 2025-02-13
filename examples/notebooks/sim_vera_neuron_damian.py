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
    pinholes = [0.5, 5.0]#np.linspace(0.5, 5.5, 21)
    downscale = 8

    for images in tqdm(range(start, start + 100)):
        sample = imread(f"{path}/{files[images]}")
        sim = ms.Simulation.from_ground_truth(
            ground_truth=sample,
            scale=(0.04, 0.02, 0.02),
            output_space={"downscale": downscale},
            modality=ms.Confocal(pinhole_au=0.5),
            detector=ms.CameraCCD(qe=0.82,full_well=18000,read_noise=6,bit_depth=12,offset=100),
            settings=ms.Settings(random_seed=None, max_psf_radius_aus=4, np_backend="numpy", device='cpu'))
        gt = sim.ground_truth()
        all_images_noisy_2D = []
        all_images_clean_2D = []
        all_images_noisy_3D = []
        all_images_clean_3D = []
        for _, au in enumerate(pinholes):
            sim = sim.model_copy(update=dict(modality=ms.Confocal(pinhole_au=au)))
            sim.run()
            optical_image = sim.optical_image(gt)
            optical_image = np.array(optical_image)

            #noisy image
            generated_images_noisy = sim.digital_image(optical_image, with_detector_noise=True, photons_pp_ps_max=2000)
            generated_images_noisy = np.array(generated_images_noisy)
            all_images_noisy_3D.append(generated_images_noisy)
            generated_image_noisy = generated_images_noisy[(generated_images_noisy.shape[0] // 2)-1]
            all_images_noisy_2D.append(generated_image_noisy)

            #clean image
            generated_images_clean = sim.digital_image(optical_image, with_detector_noise=False)
            generated_images_clean = np.array(generated_images_clean)
            all_images_clean_3D.append(generated_images_clean)
            generated_image_clean = generated_images_clean[(generated_images_clean.shape[0] // 2)-1]
            all_images_clean_2D.append(generated_image_clean)

        all_images_noisy_2D = np.array(all_images_noisy_2D)
        all_images_clean_2D = np.array(all_images_clean_2D)
        all_images_noisy_3D = np.array(all_images_noisy_3D)
        all_images_clean_3D = np.array(all_images_clean_3D)
        if images >= 5900  and images < 6000:
            imwrite(f"/group/jug/Anirban/Datasets/AllNeuron_Combined_3D_2D/2D/test_noisy/{files[images]}",all_images_noisy_2D.astype(np.float32))
            imwrite(f"/group/jug/Anirban/Datasets/AllNeuron_Combined_3D_2D/2D/test_clean/{files[images]}",all_images_clean_2D.astype(np.float32))
            imwrite(f"/group/jug/Anirban/Datasets/AllNeuron_Combined_3D_2D/3D/test_noisy/{files[images]}",all_images_noisy_3D.astype(np.float32), imagej=True, metadata={"axes": "TZYX"})
            imwrite(f"/group/jug/Anirban/Datasets/AllNeuron_Combined_3D_2D/3D/test_clean/{files[images]}",all_images_clean_3D.astype(np.float32), imagej=True, metadata={"axes": "TZYX"})

        elif images >= 5800 and images < 5900:
            imwrite(f"/group/jug/Anirban/Datasets/AllNeuron_Combined_3D_2D/2D/val_noisy/{files[images]}",all_images_noisy_2D.astype(np.float32))
            imwrite(f"/group/jug/Anirban/Datasets/AllNeuron_Combined_3D_2D/2D/val_clean/{files[images]}",all_images_clean_2D.astype(np.float32))
            imwrite(f"/group/jug/Anirban/Datasets/AllNeuron_Combined_3D_2D/3D/val_noisy/{files[images]}",all_images_noisy_3D.astype(np.float32), imagej=True, metadata={"axes": "TZYX"})
            imwrite(f"/group/jug/Anirban/Datasets/AllNeuron_Combined_3D_2D/3D/val_clean/{files[images]}",all_images_clean_3D.astype(np.float32), imagej=True, metadata={"axes": "TZYX"})

        else:
            imwrite(f"/group/jug/Anirban/Datasets/AllNeuron_Combined_3D_2D/2D/train_noisy/{files[images]}",all_images_noisy_2D.astype(np.float32))
            imwrite(f"/group/jug/Anirban/Datasets/AllNeuron_Combined_3D_2D/2D/train_clean/{files[images]}",all_images_clean_2D.astype(np.float32))
            imwrite(f"/group/jug/Anirban/Datasets/AllNeuron_Combined_3D_2D/3D/train_noisy/{files[images]}",all_images_noisy_3D.astype(np.float32), imagej=True, metadata={"axes": "TZYX"})
            imwrite(f"/group/jug/Anirban/Datasets/AllNeuron_Combined_3D_2D/3D/train_clean/{files[images]}",all_images_clean_3D.astype(np.float32), imagej=True, metadata={"axes": "TZYX"})
