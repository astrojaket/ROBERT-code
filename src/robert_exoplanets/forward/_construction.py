"""Private preparation primitives used by the typed forward factories.

The public factory functions assemble atmosphere and radiative-transfer model
objects.  This module owns the source-facing preparation that must happen
before those objects are assembled: loading a configured opacity source,
choosing a pressure grid, and applying correlated-k spectral binning.

No radiative-transfer code is imported here.  The returned provider is still
prepared once by the forward-model constructor, where the model's full set of
species and grids is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from robert_exoplanets.core import (
    PressureGrid,
    RobertConfigError,
    RobertValidationError,
    SpectralGrid,
)
from robert_exoplanets.opacity import (
    CorrelatedKOpacityProvider,
    LineByLineOpacityProvider,
    OpacityProvider,
    OpacitySamplingProvider,
)


class _OpacitySource(Protocol):
    """Minimal protocol for a typed, file-backed opacity source."""

    species: tuple[str, ...]

    def load(self) -> OpacityProvider:
        """Load the source into an opacity provider."""


class _OpacityBinning(Protocol):
    """Minimal protocol for correlated-k spectral binning configuration."""

    def apply(
        self,
        provider: CorrelatedKOpacityProvider,
        spectral_grid: SpectralGrid,
    ) -> CorrelatedKOpacityProvider:
        """Return a provider on the requested observed spectral grid."""


@dataclass(frozen=True)
class PreparedOpacityInputs:
    """Opacity and pressure inputs ready for forward-model assembly.

    ``native_provider`` records the source result before optional spectral
    binning.  ``provider`` is the provider passed to the forward model.  The
    forward-model constructor performs its normal one-time preparation on the
    selected model species after this step.
    """

    native_provider: OpacityProvider
    provider: OpacityProvider
    pressure_grid: PressureGrid


_PROVIDER_TYPES = (
    CorrelatedKOpacityProvider,
    LineByLineOpacityProvider,
    OpacitySamplingProvider,
)


def _load_opacity_source(
    source: OpacityProvider | _OpacitySource,
) -> OpacityProvider:
    """Resolve a source configuration or keep an already loaded provider."""

    if isinstance(source, _PROVIDER_TYPES):
        return source
    loader = getattr(source, "load", None)
    if callable(loader):
        return loader()
    # Factory config validation restricts this path to OpacityProvider objects.
    # Keep the original object here so invalid custom values fail at the same
    # preparation boundary as before, instead of changing their exception type.
    return source  # type: ignore[return-value]


def _prepare_provider(
    provider: OpacityProvider,
    binning: _OpacityBinning | None,
    spectral_grid: SpectralGrid,
) -> OpacityProvider:
    """Apply only the preparation supported by the selected opacity mode."""

    if isinstance(provider, (OpacitySamplingProvider, LineByLineOpacityProvider)):
        # These providers retain exact physical wavelength samples.  Exo-k
        # binning/recompression applies only to correlated-k providers.
        return provider
    return provider if binning is None else binning.apply(provider, spectral_grid)


def prepare_opacity_inputs(
    source: OpacityProvider | _OpacitySource,
    *,
    spectral_grid: SpectralGrid,
    pressure_grid: PressureGrid | None,
    species: str,
    opacity_binning: _OpacityBinning | None,
) -> PreparedOpacityInputs:
    """Prepare source and grid inputs before atmosphere/RT model assembly.

    The operation is deterministic for a fixed source, spectral grid, and
    binning configuration.  It performs no atmosphere evaluation or
    radiative-transfer work.
    """

    native_provider = _load_opacity_source(source)
    resolved_pressure_grid = pressure_grid or pressure_grid_from_opacity(
        native_provider,
        species=species,
    )
    provider = _prepare_provider(
        native_provider,
        opacity_binning,
        spectral_grid,
    )
    return PreparedOpacityInputs(
        native_provider=native_provider,
        provider=provider,
        pressure_grid=resolved_pressure_grid,
    )


def pressure_grid_from_opacity(
    provider: OpacityProvider,
    *,
    species: str | None = None,
    name: str | None = None,
) -> PressureGrid:
    """Construct layer edges around one opacity table's pressure centers."""

    selected_species = provider.species[0] if species is None else str(species)
    try:
        table = provider.tables[selected_species]
    except KeyError as exc:
        raise RobertConfigError(
            f"cannot derive pressure grid: opacity species {selected_species!r} is unavailable"
        ) from exc
    centers = np.asarray(table.pressure_bar, dtype=float)
    if centers.size < 2:
        raise RobertValidationError(
            "at least two opacity pressure points are required to infer layer edges; "
            "provide pressure_grid explicitly"
        )
    if np.any(centers <= 0.0) or not (
        np.all(np.diff(centers) > 0.0) or np.all(np.diff(centers) < 0.0)
    ):
        raise RobertValidationError(
            "opacity pressure centers must be positive and monotonic"
        )
    log_centers = np.log(centers)
    inner_edges = 0.5 * (log_centers[:-1] + log_centers[1:])
    first_edge = log_centers[0] - (inner_edges[0] - log_centers[0])
    last_edge = log_centers[-1] + (log_centers[-1] - inner_edges[-1])
    edges = np.exp(np.concatenate(([first_edge], inner_edges, [last_edge])))
    return PressureGrid(
        edges=edges,
        centers=centers,
        unit="bar",
        name=name or f"{selected_species} opacity pressure grid",
        metadata={
            "source": (
                "line-by-line opacity"
                if isinstance(provider, LineByLineOpacityProvider)
                else "correlated-k opacity"
            ),
            "species": selected_species,
        },
    )


__all__ = [
    "PreparedOpacityInputs",
    "prepare_opacity_inputs",
    "pressure_grid_from_opacity",
]
