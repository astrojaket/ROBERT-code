"""Shared atmosphere-to-optical-depth evaluation for forward models."""

from __future__ import annotations

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
    GasOpticalDepth,
    assemble_gas_optical_depth,
    assemble_opacity_sampling_gas_optical_depth,
)


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


__all__ = ["evaluate_gas_optical_depth"]
