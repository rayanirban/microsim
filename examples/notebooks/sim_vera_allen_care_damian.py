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
    path = f"/group/jug/Anirban/Datasets/care_florian/N2V_Processed/All_data_N2V"
    path_original = f"/group/jug/Anirban/Datasets/care_florian/All_data"
    files = os.listdir(path)
    files.sort()
    generated_images = []
    pinholes = [30.0]#np.linspace(5.0, 5.5, 2)
    downscale = 1
    repeat_size = 52

    for images in tqdm(range(start, len(files))):
        sample = imread(f"{path}/{files[images]}")
        sample_original = imread(f"{path_original}/{files[images]}")
        
        #convert the image to uint8
        # Normalize to range [0, 255] and convert to uint8 and then back to uint16
        min_val = np.min(sample_original, axis=(1,2))
        max_val = np.max(sample_original, axis=(1,2))
        sample_original = ((sample_original - min_val[:, None, None]) / (max_val - min_val)[:, None, None] * 255).astype(np.uint8).astype(np.uint16)
        
        sample = sample.astype(np.uint16)
            
        sim = ms.Simulation.from_ground_truth(
            ground_truth=sample,
            # scale=(0.04, 0.02, 0.02), #.6, .2, .2
            scale=(0.4, 0.2, 0.2),
            output_space={"downscale": downscale},
            modality=ms.Confocal(pinhole_au=0.5),
            detector=ms.CameraCCD(qe=0.82,full_well=18000,read_noise=6,bit_depth=12,offset=100),
            settings=ms.Settings(random_seed=None, np_backend="numpy", device='cpu'))
        gt = sim.ground_truth()
        all_images_noisy_2D = []
        all_images_noisy_3D = []
        
        #append the middle slice of gt the 3D volume to the list
        all_images_noisy_2D.append(sample_original[sample_original.shape[0] // 2])
        all_images_noisy_3D.append(sample_original)
        for _, au in enumerate(pinholes):
            sim = sim.model_copy(update=dict(modality=ms.Confocal(pinhole_au=au)))
            sim.run()
            optical_image = sim.optical_image(gt)
            generated_images_noisy = sim.digital_image(optical_image, with_detector_noise=True, photons_pp_ps_max=10000)
            generated_images_noisy = np.array(generated_images_noisy)
            all_images_noisy_3D.append(generated_images_noisy)
            generated_image_noisy = generated_images_noisy[generated_images_noisy.shape[0] // 2]
            all_images_noisy_2D.append(generated_image_noisy)


        all_images_noisy_2D = np.array(all_images_noisy_2D)
        all_images_noisy_3D = np.array(all_images_noisy_3D)
        imwrite(f"/group/jug/Anirban/Datasets/care_florian/N2V_Processed/microsim/2D_3D/2D/train_noisy/{files[images]}",all_images_noisy_2D.astype(np.float32))
        imwrite(f"/group/jug/Anirban/Datasets/care_florian/N2V_Processed/microsim/2D_3D/3D/train_noisy/{files[images]}",all_images_noisy_3D.astype(np.float32), imagej=True, metadata={"axes": "TZYX"})
