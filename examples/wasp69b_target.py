"""Shared target parameters for the WASP-69b example scenarios.

Copy this module for a new system and change ``PLANET`` and ``STAR``. The
physics-scenario scripts deliberately import these same objects so target
parameters are not duplicated across clear and cloudy retrievals.

Observation data, instrument modes, opacity coverage, and retrieval priors are
separate science inputs and must also be reviewed for a new target.
"""

from __future__ import annotations

from pathlib import Path

from robert_exoplanets import Planet, Star, load_schlawin2024_wasp69b


GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.67430e-11
JUPITER_RADIUS_M = 7.1492e7
JUPITER_MASS_KG = 1.89813e27
SOLAR_RADIUS_M = 6.957e8
ROOT = Path(__file__).resolve().parents[1]
TARGET_SLUG = "wasp69b"
DATA_DIRECTORY = ROOT / "data" / "wasp69b_schlawin2024"
CACHE_DIRECTORY = ROOT / "opacity_data" / "wasp69b_nircam_observation_bins"

PLANET = Planet(
    name="WASP-69b",
    radius_m=1.06 * JUPITER_RADIUS_M,
    mass_kg=0.26 * JUPITER_MASS_KG,
)
STAR = Star(
    name="WASP-69",
    radius_m=0.813 * SOLAR_RADIUS_M,
    effective_temperature_k=4750.0,
)
PLANET_GRAVITY_M_S2 = (
    GRAVITATIONAL_CONSTANT_M3_KG_S2 * PLANET.mass_kg / PLANET.radius_m**2
)


def load_observations(*, miri_offset_parameter: str | None = "miri_offset"):
    """Load the configured published spectrum without hiding its local path."""

    return load_schlawin2024_wasp69b(
        DATA_DIRECTORY,
        miri_offset_parameter=miri_offset_parameter,
    )


__all__ = [
    "CACHE_DIRECTORY",
    "DATA_DIRECTORY",
    "PLANET",
    "PLANET_GRAVITY_M_S2",
    "STAR",
    "TARGET_SLUG",
    "load_observations",
]
