"""Minimal forward-model pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from robert_exoplanets.atmosphere import AtmosphereBuilder, AtmosphereState
from robert_exoplanets.core import RobertValidationError, SpectralGrid, Spectrum
from robert_exoplanets.instruments import PreparedObservationResponse
from robert_exoplanets.opacity import EvaluatedOpacity, FixtureOpacityProvider, PreparedOpacity


@dataclass(frozen=True)
class PlaceholderEmissionBackend:
    """Deterministic emission placeholder for forward-model wiring.

    This backend intentionally does not perform radiative transfer. It produces
    a smooth baseline-plus-slope spectrum so the forward-model pipeline can be
    tested before validated emission physics exists.
    """

    name: str = "placeholder-gray-emission"
    unit: str = "eclipse_depth"
    observable: str = "eclipse_depth"
    baseline_parameter: str = "emission_baseline"
    slope_parameter: str = "emission_slope"
    default_baseline: float = 1.0e-3
    default_slope: float = 0.0

    def evaluate(
        self,
        spectral_grid: SpectralGrid,
        atmosphere: AtmosphereState,
        opacity: EvaluatedOpacity,
        parameters: Mapping[str, float],
    ) -> Spectrum:
        """Return a placeholder native-grid emission spectrum."""

        baseline = float(parameters.get(self.baseline_parameter, self.default_baseline))
        slope = float(parameters.get(self.slope_parameter, self.default_slope))
        if not np.isfinite(baseline) or not np.isfinite(slope):
            raise RobertValidationError("emission placeholder parameters must be finite")

        centered_wavelength = spectral_grid.values - np.mean(spectral_grid.values)
        values = baseline + slope * centered_wavelength
        return Spectrum(
            spectral_grid=spectral_grid,
            values=values,
            unit=self.unit,
            observable=self.observable,
            metadata={
                "backend": self.name,
                "physics": "placeholder",
            },
        )


@dataclass(frozen=True)
class ModelPrediction:
    """Forward-model output on native and observed spectral grids."""

    native_spectrum: Spectrum
    observed_spectrum: Spectrum
    atmosphere: AtmosphereState
    opacity: EvaluatedOpacity
    diagnostics: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True)
class ForwardModel:
    """Compose atmosphere, opacity fixture, emission placeholder, and response."""

    atmosphere_builder: AtmosphereBuilder
    native_spectral_grid: SpectralGrid
    instrument_response: PreparedObservationResponse
    opacity_provider: FixtureOpacityProvider = field(default_factory=FixtureOpacityProvider)
    emission_backend: PlaceholderEmissionBackend = field(default_factory=PlaceholderEmissionBackend)
    prepared_opacity: PreparedOpacity = field(init=False)

    def __post_init__(self) -> None:
        prepared_opacity = self.opacity_provider.prepare(
            spectral_grid=self.native_spectral_grid,
            pressure_grid=self.atmosphere_builder.pressure_grid,
            species=self.atmosphere_builder.species,
        )
        object.__setattr__(self, "prepared_opacity", prepared_opacity)

    def predict(self, parameters: Mapping[str, float] | None = None) -> ModelPrediction:
        """Evaluate the full non-retrieval forward-model pipeline."""

        parameter_values = {} if parameters is None else parameters
        atmosphere = self.atmosphere_builder.build(parameter_values)
        opacity = self.opacity_provider.evaluate(atmosphere, self.prepared_opacity)
        native_spectrum = self.emission_backend.evaluate(
            self.native_spectral_grid,
            atmosphere,
            opacity,
            parameter_values,
        )
        observed_spectrum = self.instrument_response.observe(native_spectrum)
        return ModelPrediction(
            native_spectrum=native_spectrum,
            observed_spectrum=observed_spectrum,
            atmosphere=atmosphere,
            opacity=opacity,
            diagnostics={
                "forward_model": "v0.3-minimal-placeholder",
                "emission_backend": self.emission_backend.name,
                "opacity_provider": self.opacity_provider.name,
            },
        )
