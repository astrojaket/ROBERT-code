"""Instrument response helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from robert_exoplanets.core import RobertCoverageError, RobertValidationError, SpectralGrid, Spectrum

from .observation import Observation


@dataclass(frozen=True)
class LinearObservationResponse:
    """Prepare a linear interpolation response onto observation wavelengths."""

    name: str = "linear-observation-grid"

    def prepare(self, observation: Observation) -> "PreparedObservationResponse":
        """Bind this response to an observation."""

        return PreparedObservationResponse(observation=observation, name=self.name)


@dataclass(frozen=True)
class PreparedObservationResponse:
    """Observation-bound response that maps native spectra to data bins."""

    observation: Observation
    name: str = "linear-observation-grid"

    def observe(self, spectrum: Spectrum) -> Spectrum:
        """Interpolate a native model spectrum onto the observation grid."""

        if spectrum.spectral_grid.unit != self.observation.wavelength_unit:
            raise RobertValidationError("spectral units must match before response mapping")
        if spectrum.unit != self.observation.flux_unit:
            raise RobertValidationError("spectrum and observation value units must match")
        if spectrum.observable != self.observation.observable:
            raise RobertValidationError("spectrum and observation observables must match")

        source_x = np.array(spectrum.spectral_grid.values, dtype=float, copy=True)
        source_y = np.array(spectrum.values, dtype=float, copy=True)
        target_x = self.observation.wavelength
        if source_x.size > 1 and source_x[0] > source_x[-1]:
            source_x = source_x[::-1]
            source_y = source_y[::-1]

        min_source = float(source_x[0])
        max_source = float(source_x[-1])
        if np.any(target_x < min_source) or np.any(target_x > max_source):
            raise RobertCoverageError("observation wavelengths extend outside native spectrum")

        values = np.interp(target_x, source_x, source_y)
        observed_grid = SpectralGrid(
            values=target_x,
            unit=self.observation.wavelength_unit,
            name=self.observation.instrument,
            role="observed",
        )
        return Spectrum(
            spectral_grid=observed_grid,
            values=values,
            unit=spectrum.unit,
            observable=spectrum.observable,
            metadata={
                "response": self.name,
                "instrument": self.observation.instrument or "",
            },
        )


@dataclass(frozen=True)
class TopHatObservationResponse:
    """Flux-conserving top-hat integration over published wavelength bins."""

    name: str = "top-hat-observation-bins"

    def prepare(self, observation: Observation) -> "PreparedTopHatObservationResponse":
        if observation.wavelength_bin_edges is None:
            raise RobertValidationError("top-hat response requires observation wavelength bin edges")
        return PreparedTopHatObservationResponse(observation=observation, name=self.name)


@dataclass(frozen=True)
class PreparedTopHatObservationResponse:
    """Observation-bound piecewise-linear bin integration."""

    observation: Observation
    name: str = "top-hat-observation-bins"

    def observe(self, spectrum: Spectrum) -> Spectrum:
        if spectrum.spectral_grid.unit != self.observation.wavelength_unit:
            raise RobertValidationError("spectral units must match before response mapping")
        if spectrum.unit != self.observation.flux_unit:
            raise RobertValidationError("spectrum and observation value units must match")
        if spectrum.observable != self.observation.observable:
            raise RobertValidationError("spectrum and observation observables must match")
        if self.observation.wavelength_bin_edges is None:  # pragma: no cover - constructor guard
            raise RobertValidationError("top-hat response requires wavelength bin edges")

        source_x = np.asarray(spectrum.spectral_grid.values, dtype=float)
        source_y = np.asarray(spectrum.values, dtype=float)
        if source_x[0] > source_x[-1]:
            source_x = source_x[::-1]
            source_y = source_y[::-1]
        edges = np.asarray(self.observation.wavelength_bin_edges, dtype=float)
        reversed_observation = edges[0] > edges[-1]
        if reversed_observation:
            edges = edges[::-1]
        if edges[0] < source_x[0] or edges[-1] > source_x[-1]:
            raise RobertCoverageError("observation bins extend outside native spectrum")

        values = np.empty(self.observation.n_points, dtype=float)
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            interior = (source_x > lower) & (source_x < upper)
            integration_x = np.concatenate(([lower], source_x[interior], [upper]))
            integration_y = np.interp(integration_x, source_x, source_y)
            values[index] = np.trapezoid(integration_y, integration_x) / (upper - lower)
        if reversed_observation:
            values = values[::-1]
        observed_grid = SpectralGrid(
            values=self.observation.wavelength,
            unit=self.observation.wavelength_unit,
            name=self.observation.instrument,
            role="observed",
            bin_edges=self.observation.wavelength_bin_edges,
        )
        return Spectrum(
            spectral_grid=observed_grid,
            values=values,
            unit=spectrum.unit,
            observable=spectrum.observable,
            metadata={
                "response": self.name,
                "instrument": self.observation.instrument or "",
                "bin_integration": "piecewise_linear_top_hat",
            },
        )


@dataclass(frozen=True)
class StratifiedSamplingObservationResponse:
    """Select deterministic physical samples inside each observation bin."""

    samples_per_bin: int = 4
    name: str = "stratified-opacity-sampling"

    def __post_init__(self) -> None:
        if (
            isinstance(self.samples_per_bin, bool)
            or int(self.samples_per_bin) != self.samples_per_bin
            or self.samples_per_bin < 1
        ):
            raise RobertValidationError("samples_per_bin must be a positive integer")
        object.__setattr__(self, "samples_per_bin", int(self.samples_per_bin))

    def prepare(
        self,
        observation: Observation,
        source_grid: SpectralGrid,
    ) -> "PreparedStratifiedSamplingObservationResponse":
        """Select source-grid points nearest equal-width bin strata midpoints."""

        if observation.wavelength_bin_edges is None:
            raise RobertValidationError(
                "stratified sampling requires observation wavelength bin edges"
            )
        if source_grid.unit != observation.wavelength_unit:
            raise RobertValidationError(
                "source and observation spectral units must match"
            )
        source = np.asarray(source_grid.values, dtype=float)
        if source[0] > source[-1]:
            source = source[::-1]
        edges = np.asarray(observation.wavelength_bin_edges, dtype=float)
        reversed_observation = edges[0] > edges[-1]
        ascending_edges = edges[::-1] if reversed_observation else edges
        selected_values = []
        selected_bins = []
        for ascending_bin, (lower, upper) in enumerate(
            zip(ascending_edges[:-1], ascending_edges[1:], strict=True)
        ):
            available = source[(source >= lower) & (source <= upper)]
            if available.size < self.samples_per_bin:
                raise RobertCoverageError(
                    "source opacity grid has fewer points than requested in an observation bin"
                )
            stratum_edges = np.linspace(lower, upper, self.samples_per_bin + 1)
            midpoints = 0.5 * (stratum_edges[:-1] + stratum_edges[1:])
            insertion = np.searchsorted(available, midpoints)
            right = np.clip(insertion, 0, available.size - 1)
            left = np.clip(insertion - 1, 0, available.size - 1)
            choose_left = np.abs(available[left] - midpoints) <= np.abs(
                available[right] - midpoints
            )
            chosen = np.where(choose_left, available[left], available[right])
            if np.unique(chosen).size != chosen.size:
                raise RobertCoverageError(
                    "stratified sampling selected duplicate source points; reduce samples_per_bin"
                )
            original_bin = (
                observation.n_points - 1 - ascending_bin
                if reversed_observation
                else ascending_bin
            )
            selected_values.extend(chosen.tolist())
            selected_bins.extend([original_bin] * self.samples_per_bin)
        order = np.argsort(selected_values)
        values = np.asarray(selected_values, dtype=float)[order]
        bin_indices = np.asarray(selected_bins, dtype=np.int64)[order]
        return PreparedStratifiedSamplingObservationResponse(
            observation=observation,
            spectral_grid=SpectralGrid.from_array(
                values,
                unit=source_grid.unit,
                name=f"{observation.instrument or 'observation'} stratified samples",
                role="opacity_sampling",
            ),
            sample_bin_indices=bin_indices,
            samples_per_bin=self.samples_per_bin,
            name=self.name,
        )


@dataclass(frozen=True)
class PreparedStratifiedSamplingObservationResponse:
    """Fixed equal-weight mapping from physical samples to observation bins."""

    observation: Observation
    spectral_grid: SpectralGrid
    sample_bin_indices: ArrayLike
    samples_per_bin: int
    name: str = "stratified-opacity-sampling"

    def __post_init__(self) -> None:
        indices = np.array(self.sample_bin_indices, dtype=np.int64, copy=True)
        if indices.shape != (self.spectral_grid.size,):
            raise RobertValidationError(
                "sample_bin_indices must match the stratified spectral grid"
            )
        counts = np.bincount(indices, minlength=self.observation.n_points)
        if not np.array_equal(
            counts, np.full(self.observation.n_points, self.samples_per_bin)
        ):
            raise RobertValidationError(
                "each observation bin must contain samples_per_bin samples"
            )
        indices.setflags(write=False)
        object.__setattr__(self, "sample_bin_indices", indices)

    def observe(self, spectrum: Spectrum) -> Spectrum:
        """Average sampled model values independently within each data bin."""

        if spectrum.spectral_grid.unit != self.spectral_grid.unit:
            raise RobertValidationError("sampled spectrum unit does not match response")
        if not np.array_equal(
            spectrum.spectral_grid.values, self.spectral_grid.values
        ):
            raise RobertValidationError(
                "sampled spectrum grid does not match the prepared response"
            )
        if spectrum.unit != self.observation.flux_unit:
            raise RobertValidationError(
                "spectrum and observation value units must match"
            )
        if spectrum.observable != self.observation.observable:
            raise RobertValidationError(
                "spectrum and observation observables must match"
            )
        totals = np.bincount(
            self.sample_bin_indices,
            weights=np.asarray(spectrum.values, dtype=float),
            minlength=self.observation.n_points,
        )
        values = totals / float(self.samples_per_bin)
        return Spectrum(
            spectral_grid=self.observation.spectral_grid,
            values=values,
            unit=spectrum.unit,
            observable=spectrum.observable,
            metadata={
                "response": self.name,
                "instrument": self.observation.instrument or "",
                "samples_per_bin": str(self.samples_per_bin),
                "sampling": "fixed_equal_width_strata_midpoints",
            },
        )
