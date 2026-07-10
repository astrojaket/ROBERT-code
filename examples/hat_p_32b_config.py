"""Python configuration for the validated HAT-P-32b clear-sky model.

Copy this file when starting a new target. Paths can be changed here or passed
to :func:`make_hat_p_32b_model_config` by a script or notebook.
"""

from __future__ import annotations

from pathlib import Path

from robert_exoplanets import (
    ClearSkyEmissionFactoryConfig,
    ClearSkyEmissionModelConfig,
    ExoKOpacitySource,
    ExoKTableBinning,
    Planet,
    Star,
    TabulatedTemperatureProfile,
)

DEFAULT_OBSERVATION_NPZ = (
    Path.home()
    / "Dropbox"
    / "PostDoc4"
    / "Emission_Example"
    / "Retrieval_Results"
    / "HAT-P-32b"
    / "quench_study_emission_G395H_spectra_band.npz"
)
DEFAULT_PT_CSV = (
    Path.home()
    / "Dropbox"
    / "PostDoc4"
    / "Emission_Example"
    / "PTprofiles-Teq_1800-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_1.8-TiOVO_false-daysideavg-w_mu_area.csv"
)
DEFAULT_KTA_DIR = (
    Path.home()
    / "Dropbox"
    / "PostDoc4"
    / "Emission_Example"
    / "HAT-P-32b"
    / "kta_temp"
)

R_SUN_M = 6.957e8
R_JUP_M = 7.1492e7
M_JUP_KG = 1.898e27
PLANET_RADIUS_M = 1.98 * R_JUP_M
PLANET_MASS_KG = 0.68 * M_JUP_KG
STAR_RADIUS_M = 1.32 * R_SUN_M
STAR_TEMPERATURE_K = 6001.0
RUNTIME_NONFINITE_FILL_VALUE = 1.0e-300


def make_hat_p_32b_model_config(
    *,
    pt_csv: str | Path = DEFAULT_PT_CSV,
    kta_dir: str | Path = DEFAULT_KTA_DIR,
    opacity_species: tuple[str, ...] = ("H2O",),
    include_rayleigh: bool = True,
    exok_num: int = 300,
) -> ClearSkyEmissionFactoryConfig:
    """Return the complete reusable Python model configuration."""

    species = tuple(str(item).strip().upper() for item in opacity_species)
    pt_path = Path(pt_csv).expanduser()
    return ClearSkyEmissionFactoryConfig(
        planet=Planet(
            name="HAT-P-32b",
            radius_m=PLANET_RADIUS_M,
            mass_kg=PLANET_MASS_KG,
        ),
        star=Star(
            name="HAT-P-32",
            radius_m=STAR_RADIUS_M,
            effective_temperature_k=STAR_TEMPERATURE_K,
        ),
        temperature_profile=TabulatedTemperatureProfile.from_csv(
            pt_path,
            name="HAT-P-32b retrieval PT",
        ),
        opacity_source=ExoKOpacitySource(
            directory=Path(kta_dir).expanduser(),
            species=species,
            filename_pattern="*_emission_R1000.kta",
            interpolation="log_pressure_temperature_log_k",
            nonfinite_policy="floor",
            nonfinite_fill_value=RUNTIME_NONFINITE_FILL_VALUE,
        ),
        opacity_binning=ExoKTableBinning(num=exok_num),
        model=ClearSkyEmissionModelConfig(
            opacity_species=species,
            log_vmr_parameters={item: f"log_{item.lower()}" for item in species},
            include_rayleigh=include_rayleigh,
            gas_combination="random_overlap",
            thermal_integration_backend="auto",
            metadata={
                "target": "HAT-P-32b",
                "configuration": "examples.hat_p_32b_config",
                "temperature_profile": str(pt_path),
                "spectral_preparation": "exo_k_bin_down",
            },
        ),
    )


__all__ = [
    "DEFAULT_KTA_DIR",
    "DEFAULT_OBSERVATION_NPZ",
    "DEFAULT_PT_CSV",
    "make_hat_p_32b_model_config",
]
