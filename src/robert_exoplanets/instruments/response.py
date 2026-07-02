"""Instrument response helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
