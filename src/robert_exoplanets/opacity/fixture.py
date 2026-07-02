"""Fixture opacity provider for v0.3 pipeline wiring."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.core import PressureGrid, RobertCoverageError, RobertValidationError, SpectralGrid


@dataclass(frozen=True)
class CoverageReport:
    """Opacity coverage validation result."""

    valid: bool
    message: str
    species: tuple[str, ...]


@dataclass(frozen=True)
class PreparedOpacity:
    """Run-specific fixture opacity preparation state."""

    provider_name: str
    spectral_grid: SpectralGrid
    pressure_grid: PressureGrid
    species: tuple[str, ...]
    cache_key: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider_name:
            raise RobertValidationError("provider_name must not be empty")
        if not self.species:
            raise RobertValidationError("prepared opacity must include at least one species")
        if any(not species for species in self.species):
            raise RobertValidationError("prepared opacity species names must not be empty")
        if not self.cache_key:
            raise RobertValidationError("prepared opacity cache_key must not be empty")
        object.__setattr__(self, "species", tuple(self.species))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class EvaluatedOpacity:
    """Evaluated fixture opacity arrays.

    The array is deliberately zero-valued. It exists to exercise the opacity
    interface before ROBERT has validated opacity data or radiative transfer.
    """

    prepared: PreparedOpacity
    extinction: NDArray[np.float64]
    unit: str = "placeholder"

    def __post_init__(self) -> None:
        extinction = np.array(self.extinction, dtype=float, copy=True)
        expected_shape = (
            len(self.prepared.species),
            self.prepared.pressure_grid.n_layers,
            self.prepared.spectral_grid.size,
        )
        if extinction.shape != expected_shape:
            raise RobertValidationError("opacity extinction shape must be species x layers x wavelengths")
        if np.any(extinction < 0.0) or not np.all(np.isfinite(extinction)):
            raise RobertValidationError("opacity extinction must be finite and non-negative")
        if not self.unit:
            raise RobertValidationError("opacity unit must not be empty")
        extinction.setflags(write=False)
        object.__setattr__(self, "extinction", extinction)


@dataclass(frozen=True)
class FixtureOpacityProvider:
    """Zero-opacity fixture provider for interface and pipeline tests."""

    name: str = "fixture-zero-opacity"

    def inspect(self) -> Mapping[str, str]:
        """Return inspectable provider metadata."""

        return {
            "name": self.name,
            "physics": "placeholder",
            "description": "zero-valued opacity fixture for pipeline wiring",
        }

    def prepare(
        self,
        spectral_grid: SpectralGrid,
        pressure_grid: PressureGrid,
        species: Iterable[str],
    ) -> PreparedOpacity:
        """Prepare immutable fixture opacity state for one run."""

        species_tuple = tuple(str(item) for item in species)
        if not species_tuple:
            raise RobertValidationError("species must contain at least one item")
        if any(not item for item in species_tuple):
            raise RobertValidationError("species names must not be empty")

        cache_key = _fixture_cache_key(self.name, spectral_grid, pressure_grid, species_tuple)
        return PreparedOpacity(
            provider_name=self.name,
            spectral_grid=spectral_grid,
            pressure_grid=pressure_grid,
            species=species_tuple,
            cache_key=cache_key,
            metadata={"physics": "placeholder"},
        )

    def coverage(
        self,
        atmosphere: AtmosphereState,
        prepared: PreparedOpacity,
    ) -> CoverageReport:
        """Validate that prepared fixture state matches the atmosphere."""

        missing_species = tuple(
            species for species in prepared.species if species not in atmosphere.composition
        )
        if missing_species:
            return CoverageReport(
                valid=False,
                message="atmosphere is missing prepared opacity species",
                species=missing_species,
            )
        if not np.array_equal(prepared.pressure_grid.centers, atmosphere.pressure_grid.centers):
            return CoverageReport(
                valid=False,
                message="prepared pressure grid does not match atmosphere pressure grid",
                species=prepared.species,
            )
        return CoverageReport(valid=True, message="covered", species=prepared.species)

    def evaluate(
        self,
        atmosphere: AtmosphereState,
        prepared: PreparedOpacity,
    ) -> EvaluatedOpacity:
        """Return zero-valued fixture opacity after coverage validation."""

        report = self.coverage(atmosphere, prepared)
        if not report.valid:
            raise RobertCoverageError(report.message)

        extinction = np.zeros(
            (
                len(prepared.species),
                prepared.pressure_grid.n_layers,
                prepared.spectral_grid.size,
            ),
            dtype=float,
        )
        return EvaluatedOpacity(prepared=prepared, extinction=extinction)


def _fixture_cache_key(
    provider_name: str,
    spectral_grid: SpectralGrid,
    pressure_grid: PressureGrid,
    species: tuple[str, ...],
) -> str:
    payload = "|".join(
        (
            provider_name,
            spectral_grid.unit,
            np.array2string(spectral_grid.values, precision=16),
            pressure_grid.unit,
            np.array2string(pressure_grid.centers, precision=16),
            ",".join(species),
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:16]
