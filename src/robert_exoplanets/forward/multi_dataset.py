"""Native-spectrum response orchestration for plotting multiple datasets.

Retrievals should normally prepare correlated-k opacity with ``exo_k`` for
each instrument mode and evaluate one mode-specific forward model per dataset.
This module retains the shared native-spectrum path for visualization and
diagnostics, where a continuous high-resolution curve is useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from robert_exoplanets.core import Spectrum
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.instruments import (
    ObservationCollection,
    PreparedTopHatObservationResponse,
    TopHatObservationResponse,
)

NativeSpectrumEvaluator = Callable[[Mapping[str, float]], Spectrum]


@dataclass(frozen=True)
class MultiDatasetPrediction:
    """Named observed-grid spectra from one shared physical model."""

    native_spectrum: Spectrum
    spectra: Mapping[str, Spectrum]

    def __post_init__(self) -> None:
        object.__setattr__(self, "spectra", immutable_mapping(self.spectra))


@dataclass(frozen=True)
class MultiDatasetForwardModel:
    """Map one native spectrum to datasets for visualization or diagnostics.

    This is not the default correlated-k retrieval path.  Inference should use
    mode-specific opacity tables prepared with ``exo_k`` before radiative
    transfer; ``exo_k`` cannot recompress an already evaluated flux spectrum.
    """

    native_model: NativeSpectrumEvaluator
    observations: ObservationCollection
    response: TopHatObservationResponse = field(default_factory=TopHatObservationResponse)
    _prepared: tuple[tuple[str, PreparedTopHatObservationResponse], ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_prepared",
            tuple(
                (dataset.name, self.response.prepare(dataset.observation))
                for dataset in self.observations.datasets
            ),
        )

    def __call__(self, parameters: Mapping[str, float]) -> MultiDatasetPrediction:
        native = self.native_model(parameters)
        spectra = {name: response.observe(native) for name, response in self._prepared}
        return MultiDatasetPrediction(native_spectrum=native, spectra=spectra)


__all__ = ["MultiDatasetForwardModel", "MultiDatasetPrediction"]
