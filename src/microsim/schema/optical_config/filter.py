from collections.abc import Sequence
from enum import Enum
from functools import cached_property
from typing import Annotated, Any, Literal, Self

import numpy as np
from annotated_types import Interval
from pydantic import Field, computed_field

from microsim import fpbase
from microsim._field_types import Nanometers
from microsim.schema._base_model import SimBaseModel
from microsim.schema.spectrum import Spectrum

Transmission = Annotated[float, Interval(ge=0, le=1.0)]


class Placement(Enum):
    EX_PATH = "EX"
    EM_PATH = "EM"
    BS = "BS"
    BS_REFLECTS_EM = "BSi"
    ALL = "ALL"  # implies that the filter will be used reg


class _FilterBase(SimBaseModel):
    type: str = ""
    name: str = ""
    placement: Placement = Placement.ALL

    @computed_field  # type: ignore
    @cached_property
    def spectrum(self) -> Spectrum:
        return self._get_spectrum()

    def _get_spectrum(self) -> Spectrum:
        raise NotImplementedError()

    # @property
    # def reflects_emission(self) -> bool:
    #     return self.placement == Placement.BS_REFLECTS_EM

    @classmethod
    def from_fpbase(
        cls,
        filter: str | fpbase.FilterPlacement | fpbase.FilterSpectrum,
    ) -> "FullSpectrumFilter":
        if isinstance(filter, str):
            filter = fpbase.get_filter(filter)  # noqa
        elif isinstance(filter, fpbase.FilterPlacement):
            return FullSpectrumFilter(
                name=filter.name,
                placement=filter.path,
                transmission=Spectrum.from_fpbase(filter.spectrum),
            )
        if not isinstance(filter, fpbase.FilterSpectrum):
            raise TypeError(
                "filter must be a string, FilterPlacement, or FilterSpectrum, "
                f"not {type(filter)}"
            )

        return FullSpectrumFilter(
            name=filter.ownerFilter.name,
            placement=filter.subtype,
            transmission=Spectrum.from_fpbase(filter),
        )


class Bandpass(_FilterBase):
    type: Literal["bandpass"] = "bandpass"
    bandcenter: Nanometers
    bandwidth: Nanometers
    transmission: Transmission = 1.0

    def _get_spectrum(self) -> Spectrum:
        min_wave = min(300, self.bandcenter - self.bandwidth)
        max_wave = max(800, self.bandcenter + self.bandwidth)
        wavelength = np.arange(min_wave, max_wave, 1)
        return Spectrum(
            wavelength=wavelength,
            intensity=bandpass(
                wavelength,
                center=self.bandcenter.magnitude,
                bandwidth=self.bandwidth.magnitude,
                transmission=self.transmission,
            ),
        )


class Shortpass(_FilterBase):
    type: Literal["shortpass"] = "shortpass"
    cutoff: Nanometers
    slope: float | None = None
    transmission: Transmission = 1.0
    placement: Placement = Placement.EX_PATH

    def _get_spectrum(self) -> Spectrum:
        min_wave = min(300, self.cutoff - 50)
        max_wave = max(800, self.cutoff + 50)
        wavelength = np.arange(min_wave, max_wave, 1)
        return Spectrum(
            wavelength=wavelength,
            intensity=sigmoid(
                np.arange(300, 800, 1),
                self.cutoff.magnitude,
                slope=self.slope or 5,
                up=False,
                max=self.transmission,
            ),
        )


class Longpass(_FilterBase):
    type: Literal["longpass"] = "longpass"
    cuton: Nanometers
    slope: float | None = None
    transmission: Transmission = 1.0
    placement: Placement = Placement.EM_PATH

    def _get_spectrum(self) -> Spectrum:
        min_wave = min(300, self.cuton - 50)
        max_wave = max(800, self.cuton + 50)
        wavelength = np.arange(min_wave, max_wave, 1)
        return Spectrum(
            wavelength=wavelength,
            intensity=sigmoid(
                np.arange(300, 800, 1),
                self.cuton.magnitude,
                slope=self.slope or 5,
                up=True,
                max=self.transmission,
            ),
        )


class FullSpectrumFilter(_FilterBase):
    type: Literal["spectrum"] = "spectrum"
    transmission: Spectrum = Field(..., repr=False)  # because of spectrum on super()

    def _get_spectrum(self) -> Spectrum:
        return self.transmission


Filter = Bandpass | Shortpass | Longpass | FullSpectrumFilter


def sigmoid(
    x: Any, cutoff: float, slope: float = 1, max: float = 1, up: bool = True
) -> Any:
    if not up:
        slope = -slope
    return max / (1 + np.exp(slope * (x - cutoff)))


def bandpass(
    x: Any,
    center: float | Sequence[float],
    bandwidth: float | Sequence[float],
    slope: float = 5,
    transmission: float = 1,
) -> Any:
    if isinstance(center, Sequence):
        if isinstance(bandwidth, Sequence):
            if len(center) != len(bandwidth):
                raise ValueError("center and bandwidth must have the same length")
        else:
            bandwidth = [bandwidth] * len(center)

        segments = [
            bandpass(x, c, b, slope=slope, transmission=transmission)
            for c, b in zip(center, bandwidth, strict=False)
        ]
        return np.prod(segments, axis=0)
    elif isinstance(bandwidth, Sequence):
        raise ValueError("center and bandwidth must have the same shape")

    left = sigmoid(x, center - bandwidth / 2, slope=slope)
    right = sigmoid(x, center + bandwidth / 2, slope=slope, up=False)
    return left * right * transmission
