""" adapted from: https://gist.github.com/veegalinova/4fda9622474826aa0f4589b9386ba9cf"""
import numpy as np
import tifffile
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from microsim.allen import Specimen
from microsim.allen._swc import Compartment, SWC, SWCType
from tqdm import tqdm
import numpy as np
import argparse
import os

RNG = np.random.default_rng(seed=21)


def _shift_points(point_array: NDArray, offsets: NDArray) -> NDArray:
    shifted_points = point_array + offsets
    return shifted_points


def _scale_points(point_array: NDArray, scales: NDArray) -> NDArray:
    scaled_points = point_array * scales
    return scaled_points


def _rotate_points(points: NDArray, rotation_angles: NDArray) -> NDArray:
    rotation = Rotation.from_euler("zyx", rotation_angles)
    rotated_points = rotation.apply(points)
    return rotated_points

def _shift_combined_swc_origin(swc: SWC):
    points = swc.coords
    min_point = points.min(axis=0)
    transformed_points = _shift_points(points, -min_point)
    return _create_swc_with_new_coordinates(swc, transformed_points)

def _scale_combined_swc(swc: SWC, scale: float) -> SWC:
    points = swc.coords
    scaled_points = _scale_points(points, scale)
    return _create_swc_with_new_coordinates(swc, scaled_points)

def _create_combined_swc(swc_list: list[SWC]) -> SWC:
    new_compartments: list[Compartment] = []

    compartment_id_offset = 0
    for swc in swc_list:
        for compartment in swc.compartments:
            if compartment_id_offset:
                new_id = compartment.id + compartment_id_offset
                new_c = (
                    compartment.c + compartment_id_offset if compartment.c != -1 else -1
                )
                compartment = compartment._replace(id=new_id, c=new_c)
            new_compartments.append(compartment)
        compartment_id_offset = new_compartments[-1].id

    combined_swc = SWC(compartments=new_compartments)
    return _shift_combined_swc_origin(combined_swc)


def _create_swc_with_new_coordinates(swc: SWC, new_coordinates_list: NDArray) -> SWC:
    updated_compartments: list[Compartment] = []

    for compartment, new_coordinates in zip(swc.compartments, new_coordinates_list):
        z, y, x = new_coordinates
        new_compartment = compartment._replace(z=z, y=y, x=x)
        updated_compartments.append(new_compartment)

    return SWC(updated_compartments)

def _randomly_transform_swc(
    original_swc: SWC,
    scale_range: tuple[float, float],
    rotation_range: tuple[float, float],
    sample_output_size: np.ndarray,
) -> SWC:
    scales = RNG.uniform(low=scale_range[0], high=scale_range[1], size=3)
    rotations = RNG.uniform(low=rotation_range[0], high=rotation_range[1], size=3)
    points = original_swc.coords
    somas = original_swc._node_types[SWCType.SOMA]
    soma_center = somas[0].coord() if len(somas) > 0 else np.mean(points, axis=0)
    location_array = RNG.integers(low=sample_output_size//4, high=sample_output_size//3, size=3)
    soma_offsets = location_array - soma_center
    transformed_points = _shift_points(points, soma_offsets)
    transformed_points = _scale_points(transformed_points, scales)
    transformed_points = _rotate_points(transformed_points, rotations)
        
    return _create_swc_with_new_coordinates(original_swc, transformed_points)


def load_and_combine_specimens(
    specimen_ids: list[int],
    scale_range: tuple[float, float],
    rotation_range: tuple[float, float],
    sample_output_size: np.ndarray
) -> SWC:
    swc_to_combine: list[SWC] = []

    for specimen_id in specimen_ids:
        specimen = Specimen.fetch(specimen_id)
        for reconstruction in specimen.neuron_reconstructions:
            original_swc = reconstruction.swc
            transformed_swc = _randomly_transform_swc(
                original_swc, scale_range, rotation_range, sample_output_size
            )
            swc_to_combine.append(transformed_swc)

    return _create_combined_swc(swc_to_combine)


def pad_mask_if_needed(mask: NDArray, min_size=512) -> NDArray:
    padding = []
    for dim in mask.shape:
        if dim < min_size:
            pad = min_size - dim
            left_pad = pad // 2
            right_pad = pad - left_pad
            padding.append([left_pad, right_pad])
        else:
            padding.append([0, 0])
    res = np.pad(mask, padding, mode="constant")
    return res

def center_crop_mask(mask: NDArray, crop_size: NDArray, centers: NDArray) -> NDArray:
    left_side = centers - crop_size // 2  
    left_side = np.maximum(left_side, 0)
   
    cropped_mask = mask[
        left_side[0]: left_side[0]+crop_size[0],
        left_side[1]: left_side[1]+crop_size[1],
        left_side[2]: left_side[2]+crop_size[2],
    ]
    return cropped_mask


def create_combined_mask_for_simulation(
    specimen_ids: list[int],
    specimen_line_scale_factor: int,
    scale_range: tuple[float, float],
    rotation_range: tuple[float, float],
    sample_output_size: np.ndarray,
    voxel_size: float = 1.0,
) -> NDArray:
    combined_swc = load_and_combine_specimens(
        specimen_ids=specimen_ids,
        scale_range=scale_range,
        rotation_range=rotation_range,
        sample_output_size=sample_output_size,
    )
    # combined_swc = _scale_combined_swc(combined_swc, 2.0)
    sample_mask = combined_swc.binary_mask(voxel_size=voxel_size, scale_factor=specimen_line_scale_factor)
    resized_mask = pad_mask_if_needed(sample_mask, min_size=sample_output_size[1])
    resized_mask = center_crop_mask(resized_mask, crop_size=sample_output_size,centers=combined_swc.coords.mean(axis=0).astype(int))

    return resized_mask


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--start", type=int)
    args = argparser.parse_args()
    start = args.start
    all_specimen_ids = np.loadtxt("specimen_ids.csv", delimiter=",", dtype=int)
    sample_output_size = np.array([256, 512, 512])
    i = start
    with tqdm(total=100) as pbar:
        while i < start + 100:
            sample_size = np.random.randint(8, 12)  # random sample size between 8 and 12 specimens
            specimen_ids = all_specimen_ids[np.random.choice(all_specimen_ids.shape[0], sample_size, replace=False)]
            try:
                output = create_combined_mask_for_simulation(
                    specimen_ids=specimen_ids,
                    specimen_line_scale_factor=1,
                    scale_range=(0.8, 1.2),
                    rotation_range=(-2 * np.pi, 2 * np.pi),
                    sample_output_size=sample_output_size,  # z,x,y the size of the output volume
                )
                if output.shape != (sample_output_size[0], sample_output_size[1], sample_output_size[2]):
                    print(output.shape, (sample_output_size[0], sample_output_size[1], sample_output_size[2]))
                    print(f"Output shape is not correct for specimen_ids: {specimen_ids}")
                    continue
            except Exception as e:
                print(f"{e} for specimen_ids: {specimen_ids}")
                continue
            tifffile.imwrite(f"/group/jug/Anirban/Datasets/AllNeuron_Combined/GT_Volumes/{i:04d}_.tif", output.astype(np.float32))
            i += 1
            pbar.update(1)