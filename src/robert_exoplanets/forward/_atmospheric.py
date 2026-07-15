"""Shared atmosphere-to-optical-depth evaluation for forward models."""

from __future__ import annotations

from typing import Mapping, Sequence

from numpy.typing import ArrayLike

from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.opacity import (
    OpacityProvider,
    OpacitySamplingProvider,
    PreparedOpacity,
    PreparedOpacitySampling,
)
from robert_exoplanets.rt import (
    CiaTable,
    GasOpticalDepth,
    assemble_gas_optical_depth,
    assemble_opacity_sampling_gas_optical_depth,
    cia_optical_depth,
    rayleigh_scattering_optical_depth,
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
    if include_rayleigh:
        contributions.append(rayleigh_scattering_optical_depth(gas_optical_depth))
    if cloud_model is not None:
        if parameters is None:
            raise RobertValidationError(
                "parameterized cloud evaluation requires model parameters"
            )
        contributions.extend(cloud_model.evaluate(gas_optical_depth, parameters))
    return tuple(contributions)


__all__ = [
    "evaluate_additional_optical_depths",
    "evaluate_gas_optical_depth",
]
