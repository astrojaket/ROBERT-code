"""Shared atmosphere-to-optical-depth evaluation for forward models."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.opacity import (
    LineByLineOpacityProvider,
    OpacityProvider,
    OpacitySamplingProvider,
    PreparedLineByLineOpacity,
    PreparedOpacity,
    PreparedOpacitySampling,
)
from robert_exoplanets.rt import (
    CiaTable,
    GasOpticalDepth,
    HMinusContinuumConfig,
    LayerOpticalDepth,
    assemble_gas_optical_depth,
    assemble_opacity_sampling_gas_optical_depth,
    cia_optical_depth,
    hminus_optical_depth,
    rayleigh_scattering_optical_depth,
)
from robert_exoplanets.rt.optical_depth import (
    assemble_line_by_line_gas_optical_depth,
)

from .clouds import ParameterizedCloudModel


def evaluate_gas_optical_depth(
    provider: OpacityProvider,
    prepared: PreparedOpacity,
    atmosphere: AtmosphereState,
    *,
    gravity_m_s2: float | ArrayLike,
    gas_combination: str,
    retain_species_tau: bool = True,
) -> GasOpticalDepth:
    """Evaluate prepared opacity and assemble gas optical depth consistently."""

    if isinstance(provider, OpacitySamplingProvider):
        if not isinstance(prepared, PreparedOpacitySampling):
            raise RobertValidationError(
                "opacity-sampling provider requires prepared opacity sampling"
            )
        return assemble_opacity_sampling_gas_optical_depth(
            atmosphere,
            provider,
            prepared,
            gravity_m_s2=gravity_m_s2,
        )
    if isinstance(provider, LineByLineOpacityProvider):
        if not isinstance(prepared, PreparedLineByLineOpacity):
            raise RobertValidationError(
                "line-by-line provider requires prepared line-by-line opacity"
            )
        if not retain_species_tau:
            return assemble_line_by_line_gas_optical_depth(
                atmosphere,
                provider,
                prepared,
                gravity_m_s2=gravity_m_s2,
            )
    evaluated = provider.evaluate(atmosphere, prepared)
    return assemble_gas_optical_depth(
        atmosphere,
        evaluated,
        gravity_m_s2=gravity_m_s2,
        gas_combination=gas_combination,
        retain_species_tau=retain_species_tau,
    )


def evaluate_additional_optical_depths(
    gas_optical_depth: GasOpticalDepth,
    *,
    cia_tables: Sequence[CiaTable] = (),
    include_rayleigh: bool = True,
    cia_normal_hydrogen: bool = True,
    cia_temperature_extrapolation: str = "clip",
    cia_spectral_extrapolation: str = "zero",
    hminus_continuum: HMinusContinuumConfig | None = None,
    cloud_model: ParameterizedCloudModel | None = None,
    parameters: Mapping[str, float] | None = None,
) -> tuple[object, ...]:
    """Evaluate geometry-independent continuum and cloud extinction.

    Emission and transmission consume this exact optical-depth sequence. Their
    implementations diverge only when the common atmospheric extinction is
    mapped through the selected radiative-transfer geometry.
    """

    contributions: list[object] = [
        cia_optical_depth(
            gas_optical_depth,
            table,
            normal_hydrogen=cia_normal_hydrogen,
            temperature_extrapolation=cia_temperature_extrapolation,
            spectral_extrapolation=cia_spectral_extrapolation,
        )
        for table in cia_tables
    ]
    if hminus_continuum is not None:
        contributions.append(hminus_optical_depth(gas_optical_depth, hminus_continuum))
    if include_rayleigh:
        contributions.append(rayleigh_scattering_optical_depth(gas_optical_depth))
    if cloud_model is not None:
        if parameters is None:
            raise RobertValidationError(
                "parameterized cloud evaluation requires model parameters"
            )
        contributions.extend(cloud_model.evaluate(gas_optical_depth, parameters))
    return tuple(contributions)


def evaluate_combined_additional_optical_depth(
    gas_optical_depth: GasOpticalDepth,
    *,
    cia_tables: Sequence[CiaTable] = (),
    include_rayleigh: bool = True,
    cia_normal_hydrogen: bool = True,
    cia_temperature_extrapolation: str = "clip",
    cia_spectral_extrapolation: str = "zero",
    hminus_continuum: HMinusContinuumConfig | None = None,
) -> LayerOpticalDepth | None:
    """Evaluate continuum extinction into one layer optical-depth object.

    This is an explicit low-memory path for cloud-free, extinction-only
    emission models.  CIA, H-minus, and Rayleigh terms are evaluated one at a
    time and accumulated in one float64 array.  The normal plural helper is
    kept unchanged because callers that need separate diagnostics or cloud
    scattering properties must retain separate contributions.

    The returned source name is the same ``+``-joined sequence used by the
    emission solver for the uncombined path.  Per-source names, kinds, and
    metadata are also retained under namespaced metadata keys.
    """

    shape = (
        gas_optical_depth.pressure_grid.n_layers,
        gas_optical_depth.spectral_grid.size,
    )
    combined = np.zeros(shape, dtype=np.float64)
    source_names: list[str] = []
    source_kinds: list[str] = []
    source_metadata: dict[str, str] = {}

    def add_contribution(contribution: LayerOpticalDepth) -> None:
        if contribution.tau.shape != shape:
            raise RobertValidationError(
                "additional optical depth has an incompatible layer/spectral shape"
            )
        np.add(combined, contribution.tau, out=combined)
        index = len(source_names)
        source_names.append(str(contribution.name))
        source_kinds.append(str(contribution.kind))
        for key, value in contribution.metadata.items():
            source_metadata[f"source_{index}_{key}"] = str(value)

    for table in cia_tables:
        contribution = cia_optical_depth(
            gas_optical_depth,
            table,
            normal_hydrogen=cia_normal_hydrogen,
            temperature_extrapolation=cia_temperature_extrapolation,
            spectral_extrapolation=cia_spectral_extrapolation,
        )
        add_contribution(contribution)
        del contribution

    if hminus_continuum is not None:
        contribution = hminus_optical_depth(gas_optical_depth, hminus_continuum)
        add_contribution(contribution)
        del contribution

    if include_rayleigh:
        contribution = rayleigh_scattering_optical_depth(gas_optical_depth)
        add_contribution(contribution)
        del contribution

    if not source_names:
        return None
    if not np.all(np.isfinite(combined)) or np.any(combined < 0.0):
        raise RobertValidationError(
            "combined additional optical depth must be finite and non-negative"
        )
    metadata = {
        "aggregation": "streamed_float64_sum",
        "source_names": "+".join(source_names),
        "source_kinds": "+".join(source_kinds),
        "source_count": str(len(source_names)),
        **source_metadata,
    }
    return LayerOpticalDepth(
        name="+".join(source_names),
        tau=combined,
        spectral_grid=gas_optical_depth.spectral_grid,
        pressure_grid=gas_optical_depth.pressure_grid,
        kind="extinction",
        metadata=metadata,
    )


__all__ = [
    "evaluate_additional_optical_depths",
    "evaluate_combined_additional_optical_depth",
    "evaluate_gas_optical_depth",
]
