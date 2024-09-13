from __future__ import annotations

from collections import defaultdict
from enum import IntEnum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from microsim.util import http_get

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


class SWCType(IntEnum):
    """Constants for the SWC node types."""

    UNDEFINED = 0
    SOMA = 1
    AXON = 2
    BASAL_DENDRITE = 3
    APICAL_DENDRITE = 4
    CUSTOM = 5
    UNSPECIFIED_NEURITE = 6
    GLIA_PROCESSES = 7


class Compartment(NamedTuple):
    """A compartment in an SWC file."""

    id: int
    t: int  # type
    x: float
    y: float
    z: float
    r: float  # radius
    c: int  # parent id or "connectivity"

    def coord(self) -> np.ndarray:
        return np.array([self.z, self.y, self.x])

    def transformed_coord(self,
                          coords,
                          min_coords=None,
                          scale=None) -> np.ndarray:
        """Returns: Transformed coordinates to the new extent"""
        if scale is None:
            return coords
        else:
            transformed_coord = (coords - min_coords) * scale
            transformed_coord = transformed_coord.astype(int)
            return transformed_coord

    def shifted_coord(self,
                      origin: np.ndarray,
                      resolution: float = 1,
                      scale=None,
                      min_coords=None) -> tuple[int, int, int]:
        if scale is None and min_coords is None:
            shifted_coord = (self.coord() - origin) / resolution
            return tuple(c for c in shifted_coord.astype(int))
        else:
            shifted_coord = (self.transformed_coord(
                self.coord(), min_coords=min_coords, scale=scale) -
                             origin) / resolution
            return tuple(c for c in shifted_coord.astype(int))


class SWC:
    """Object representing an SWC file.

    Provides methods to parse SWC files and render them as binary masks.
    """

    @classmethod
    def from_path(cls, path: str | Path) -> SWC:
        if str(path).startswith(("http://", "https://")):
            response = http_get(str(path))
            content = response.decode()
        else:
            content = Path(path).expanduser().read_text()
        return cls.from_string(content)

    @classmethod
    def from_string(cls, content: str | bytes) -> SWC:
        if isinstance(content, bytes):
            content = content.decode()

        compartments = []
        for num, line in enumerate(content.splitlines()):
            if line.startswith("#"):
                continue
            try:
                a, b, c, d, e, f, g = line.split()
                comp = Compartment(int(a), int(b), float(c), float(d), float(e), float(f), int(g))
            except ValueError as e:
                raise ValueError(f"Invalid SWC line {num}: {line}") from e
            compartments.append(comp)
        return cls(compartments)

    def __init__(self, compartments: Sequence[Compartment] = ()):
        self.compartments = compartments

        self._id_map: dict[int, Compartment] = {}
        self._children_of: defaultdict[int, list[Compartment]] = defaultdict(list)
        self._node_types: defaultdict[int, list[Compartment]] = defaultdict(list)
        for comp in compartments:
            self._id_map[comp.id] = comp
            self._children_of[comp.c].append(comp)
            self._node_types[comp.t].append(comp)

    def iter_pairs(self, *types: int) -> Iterator[tuple[Compartment, Compartment]]:
        seen = set()
        for comp_id, children in self._children_of.items():
            if comp_id == -1:
                continue
            comp = self._id_map[comp_id]
            if not types or comp.t in types:
                for child in children:
                    if (comp, child) not in seen:
                        yield comp, child
                        seen.add((comp, child))

    @cached_property
    def coords(self) -> np.ndarray:
        """Return the coordinates of the compartments as (M, 3) array."""
        return np.array([(c.z, c.y, c.x)
                         for c in self.compartments]).astype("float32")

    def new_extent(self, new_extent: int | tuple[int, int, int] = 512) -> np.ndarray:
        """Transform all the SWC coordinates to a new extent"""
        
        for i in range(len(self._node_types[SWCType.SOMA])):
            soma = self._node_types[SWCType.SOMA][i]
            soma_center = np.array([soma.z, soma.y, soma.x])
            soma_center = soma_center.astype(int)
            min_coords = np.min(self.coords, axis=0)
            max_coords = np.max(self.coords, axis=0)
            scale = new_extent / (max_coords - min_coords)
            transformed_soma = (soma_center - min_coords) * scale
            self._node_types[SWCType.SOMA][i] = self._node_types[
                SWCType.SOMA][i]._replace(z=transformed_soma[0],
                                        y=transformed_soma[1],
                                        x=transformed_soma[2])
        for i in range(len(self._node_types[SWCType.AXON])):
            axon = self._node_types[SWCType.AXON][i]
            axon_center = np.array([axon.z, axon.y, axon.x])
            axon_center = axon_center.astype(int)
            min_coords = np.min(self.coords, axis=0)
            max_coords = np.max(self.coords, axis=0)
            scale = new_extent / (max_coords - min_coords)
            transformed_axon = (axon_center - min_coords) * scale
            self._node_types[SWCType.AXON][i] = self._node_types[
                SWCType.AXON][i]._replace(z=transformed_axon[0],
                                        y=transformed_axon[1],
                                        x=transformed_axon[2])
        for i in range(len(self._node_types[SWCType.BASAL_DENDRITE])):
            basal = self._node_types[SWCType.BASAL_DENDRITE][i]
            basal_center = np.array([basal.z, basal.y, basal.x])
            basal_center = basal_center.astype(int)
            min_coords = np.min(self.coords, axis=0)
            max_coords = np.max(self.coords, axis=0)
            scale = new_extent / (max_coords - min_coords)
            transformed_basal = (basal_center - min_coords) * scale
            self._node_types[SWCType.BASAL_DENDRITE][i] = self._node_types[
                SWCType.BASAL_DENDRITE][i]._replace(z=transformed_basal[0],
                                        y=transformed_basal[1],
                                        x=transformed_basal[2])
        for i in range(len(self._node_types[SWCType.APICAL_DENDRITE])):
            apical = self._node_types[SWCType.APICAL_DENDRITE][i]
            apical_center = np.array([apical.z, apical.y, apical.x])
            apical_center = apical_center.astype(int)
            min_coords = np.min(self.coords, axis=0)
            max_coords = np.max(self.coords, axis=0)
            scale = new_extent / (max_coords - min_coords)
            transformed_apical = (apical_center - min_coords) * scale
            self._node_types[SWCType.APICAL_DENDRITE][i] = self._node_types[
                SWCType.APICAL_DENDRITE][i]._replace(z=transformed_apical[0],
                                        y=transformed_apical[1],
                                        x=transformed_apical[2])

    def root(self) -> Compartment:
        """Return the root compartment of the SWC."""
        return self._id_map[1]

    def origin(self) -> tuple[float, float, float]:
        """Return the (Z,Y,X) coordinate of the root node."""
        root = self.root()
        return root.z, root.y, root.x

    def _empty_grid(self,
                    voxel_size: float = 1.0,
                    coordinates=None) -> np.ndarray:
        """Create an empty 3D grid for the binary mask."""
        coordinates = self.coords if coordinates is None else coordinates
        extent = np.ptp(coordinates, axis=0)
        size = np.ceil(extent / voxel_size).astype(int)
        return np.zeros(size, dtype=bool)

    def get_empty_grid(self,
                       voxel_size: float = 1.0,
                       coordinates=None) -> np.ndarray:
        """Create an empty 3D grid for the binary mask."""
        return self._empty_grid(voxel_size, coordinates)

    def binary_mask(
        self,
        voxel_size: float = 1,
        scale_factor: float = 3,
        empty_grid: np.ndarray | None = None,
        *,
        include_types: Iterable[int] = (
            SWCType.BASAL_DENDRITE,
            SWCType.APICAL_DENDRITE,
            SWCType.AXON,
        ),
    ) -> np.ndarray:
        """Render a binary mask of the neuron reconstruction."""
        from microsim._draw import draw_line_3d, draw_sphere

        if empty_grid is None:
            grid = self._empty_grid(voxel_size)
        origin = np.min(self.coords, axis=0)

        dend_scale: float = 1
        max_r = float(np.sum(grid.shape))

        for par, child in self.iter_pairs(*include_types):
            r = int(max(1,0.5 * scale_factor * dend_scale * (par.r + child.r)))
            pz, py, px = par.shifted_coord(origin, voxel_size)
            cz, cy, cx = child.shifted_coord(origin, voxel_size)
            draw_line_3d(px, py, pz, cx, cy, cz, grid, max_r=max_r, width=r)

        soma_scale: float = 1
        for comp in self._node_types[SWCType.SOMA]:
            z, y, x = comp.shifted_coord(origin, voxel_size)
            r = int(0.5 * scale_factor * soma_scale * comp.r)
            draw_sphere(grid, x, y, z, r)

        return grid.astype(np.uint8)

    def soma_location(self) -> tuple[float, float, float]:
        """Return the location of the soma for this neuron reconstruction."""
        return self._node_types[SWCType.SOMA]

    def coords_location(self) -> np.ndarray:
        """Return the coordinates for this neuron reconstruction."""
        return self._node_types[SWCType.AXON], self._node_types[SWCType.BASAL_DENDRITE], self._node_types[SWCType.APICAL_DENDRITE]

    def offset_coords(self, offset: tuple[float, float, float]) -> np.ndarray:
        """Offset the coordinates of the neuron reconstruction."""

        for i in range(len(self._node_types[SWCType.SOMA])):
            soma = self._node_types[SWCType.SOMA][i]
            new_soma = soma._replace(z=soma.z + offset[0],
                                    y=soma.y + offset[1],
                                    x=soma.x + offset[2])
            self._node_types[SWCType.SOMA][i] = new_soma

        for i in range(len(self._node_types[SWCType.AXON])):
            axon = self._node_types[SWCType.AXON][i]
            new_axon = axon._replace(z=axon.z + offset[0],
                                    y=axon.y + offset[1],
                                    x=axon.x + offset[2])
            self._node_types[SWCType.AXON][i] = new_axon
        for i in range(len(self._node_types[SWCType.BASAL_DENDRITE])):
            basal = self._node_types[SWCType.BASAL_DENDRITE][i]
            new_basal = basal._replace(z=basal.z + offset[0],
                                    y=basal.y + offset[1],
                                    x=basal.x + offset[2])
            self._node_types[SWCType.BASAL_DENDRITE][i] = new_basal
        for i in range(len(self._node_types[SWCType.APICAL_DENDRITE])):
            apical = self._node_types[SWCType.APICAL_DENDRITE][i]
            new_apical = apical._replace(z=apical.z + offset[0],
                                    y=apical.y + offset[1],
                                    x=apical.x + offset[2])
            self._node_types[SWCType.APICAL_DENDRITE][i] = new_apical
            
        # return self._node_types[SWCType.AXON], self._node_types[SWCType.BASAL_DENDRITE], self._node_types[SWCType.APICAL_DENDRITE]            

    def rotate_coords(self, rotation: tuple[float, float, float]) -> np.ndarray:
        """Rotate the coordinates of the neuron reconstruction using a rotation translation matrix."""

        #create rotation-translation matrix
        rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(rotation[0]), -np.sin(rotation[0]), 0],
                       [0, np.sin(rotation[0]),
                        np.cos(rotation[0]), 0], [0, 0, 0, 1]])
        ry = np.array([[np.cos(rotation[1]), 0,
                        np.sin(rotation[1]), 0], [0, 1, 0, 0],
                       [-np.sin(rotation[1]), 0,
                        np.cos(rotation[1]), 0], [0, 0, 0, 1]])
        rz = np.array([[np.cos(rotation[2]), -np.sin(rotation[2]), 0, 0],
                       [np.sin(rotation[2]),
                        np.cos(rotation[2]), 0, 0], [0, 0, 1, 0], [0, 0, 0,1]])

        rotation_matrix = rx.dot(ry).dot(rz)

        for i in range(len(self._node_types[SWCType.SOMA])):
            soma = self._node_types[SWCType.SOMA][i]
            soma_center = np.array([soma.z, soma.y, soma.x])
            soma_center = np.append(soma_center, 1)
            rotated_soma = rotation_matrix.dot(soma_center)
            self._node_types[SWCType.SOMA][i] = self._node_types[
                SWCType.SOMA][i]._replace(z=rotated_soma[0],
                                        y=rotated_soma[1],
                                        x=rotated_soma[2])     
        
        for i in range(len(self._node_types[SWCType.AXON])):
            axon = self._node_types[SWCType.AXON][i]
            axon_center = np.array([axon.z, axon.y, axon.x])
            axon_center = np.append(axon_center, 1)
            rotated_axon = rotation_matrix.dot(axon_center)
            self._node_types[SWCType.AXON][i] = self._node_types[
                SWCType.AXON][i]._replace(z=rotated_axon[0],
                                        y=rotated_axon[1],
                                        x=rotated_axon[2])
        for i in range(len(self._node_types[SWCType.BASAL_DENDRITE])):
            basal = self._node_types[SWCType.BASAL_DENDRITE][i]
            basal_center = np.array([basal.z, basal.y, basal.x])
            basal_center = np.append(basal_center, 1)
            rotated_basal = rotation_matrix.dot(basal_center)
            self._node_types[SWCType.BASAL_DENDRITE][i] = self._node_types[
                SWCType.BASAL_DENDRITE][i]._replace(z=rotated_basal[0],
                                        y=rotated_basal[1],
                                        x=rotated_basal[2])
        for i in range(len(self._node_types[SWCType.APICAL_DENDRITE])):
            apical = self._node_types[SWCType.APICAL_DENDRITE][i]
            apical_center = np.array([apical.z, apical.y, apical.x])
            apical_center = np.append(apical_center, 1)
            rotated_apical = rotation_matrix.dot(apical_center)
            self._node_types[SWCType.APICAL_DENDRITE][i] = self._node_types[
                SWCType.APICAL_DENDRITE][i]._replace(z=rotated_apical[0],
                                        y=rotated_apical[1],
                                        x=rotated_apical[2])
                

    def scale_coords(self, scale: float) -> np.ndarray:
        """Scale the coordinates of the neuron reconstruction."""

        for i in range(len(self._node_types[SWCType.SOMA])):
            soma = self._node_types[SWCType.SOMA][i]
            soma_center = np.array([soma.z, soma.y, soma.x])
            soma_center = soma_center * scale
            self._node_types[SWCType.SOMA][i] = self._node_types[SWCType.SOMA][i]._replace(
                z=soma_center[0], y=soma_center[1], x=soma_center[2])        
       
        for i in range(len(self._node_types[SWCType.AXON])):
            axon = self._node_types[SWCType.AXON][i]
            axon_center = np.array([axon.z, axon.y, axon.x])
            axon_center = axon_center * scale
            self._node_types[SWCType.AXON][i] = self._node_types[SWCType.AXON][i]._replace(
                z=axon_center[0], y=axon_center[1], x=axon_center[2])

        for i in range(len(self._node_types[SWCType.BASAL_DENDRITE])):
            basal = self._node_types[SWCType.BASAL_DENDRITE][i]
            basal_center = np.array([basal.z, basal.y, basal.x])
            basal_center = basal_center * scale
            self._node_types[SWCType.BASAL_DENDRITE][i] = self._node_types[SWCType.BASAL_DENDRITE][i]._replace(
                z=basal_center[0], y=basal_center[1], x=basal_center[2])

        for i in range(len(self._node_types[SWCType.APICAL_DENDRITE])):
            apical = self._node_types[SWCType.APICAL_DENDRITE][i]
            apical_center = np.array([apical.z, apical.y, apical.x])
            apical_center = apical_center * scale
            self._node_types[SWCType.APICAL_DENDRITE][i] = self._node_types[SWCType.APICAL_DENDRITE][i]._replace(
                z=apical_center[0], y=apical_center[1], x=apical_center[2])        

    def bounding_box(
        self
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Return the bounding box for the neuron reconstruction."""
        min_coords = np.min(self.coords, axis=0)
        max_coords = np.max(self.coords, axis=0)
        return (min_coords[0], max_coords[0]), (min_coords[1],max_coords[1]), (min_coords[2],max_coords[2])