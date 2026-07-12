"""Target and data configuration for the WASP-80b retrieval examples."""

from __future__ import annotations

from pathlib import Path

from robert_exoplanets import Planet, Star, load_wiser2025_wasp80b


ROOT = Path(__file__).resolve().parents[1]
TARGET_SLUG = "wasp80b"
DATA_DIRECTORY = ROOT / "data" / "wasp80b_wiser2025"
CACHE_DIRECTORY = ROOT / "opacity_data" / "wasp80b_observation_bins"
GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.67430e-11
JUPITER_MASS_KG = 1.89813e27
SOLAR_RADIUS_M = 6.957e8

# The radius uses the fitted Rp/Rs=0.17157299424754727 and Rs=0.586 Rsun
# recorded in the archived Eureka! Stage-5 configuration so the area ratio is
# consistent with the reduced eclipse spectrum. NASA's catalog gives 0.538
# Jupiter masses; the Exoplanet Archive includes Teff=4145 K among the
# published stellar solutions.
PLANET = Planet(
    name="WASP-80b",
    radius_m=0.17157299424754727 * 0.586 * SOLAR_RADIUS_M,
    mass_kg=0.538 * JUPITER_MASS_KG,
    metadata={
        "radius_source": "Zenodo 10.5281/zenodo.13146949 Stage-5 fit",
        "mass_source": "NASA Exoplanet Catalog WASP-80b",
    },
)
STAR = Star(
    name="WASP-80",
    radius_m=0.586 * SOLAR_RADIUS_M,
    effective_temperature_k=4145.0,
    metadata={"source": "Zenodo Stage-5 Rs; NASA Exoplanet Archive Teff solution"},
)
PLANET_GRAVITY_M_S2 = (
    GRAVITATIONAL_CONSTANT_M3_KG_S2 * PLANET.mass_kg / PLANET.radius_m**2
)


def load_observations(*, miri_offset_parameter: str | None = "miri_offset"):
    """Load the configured published spectrum without hiding its local path."""

    return load_wiser2025_wasp80b(
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
