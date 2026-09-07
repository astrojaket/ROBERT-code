"""Bounded WASP-77Ab target and real HRS/LRS comparison protocol.

This module contains metadata and acceptance rules only.  It does not import
pRT or a sampler, download data, or start a retrieval.  The values in the
target objects are the nominal values in Smith et al. (2024), Table 1.  The
latest transit ephemeris printed in that table is retained as the default
orbit, while the older Maxted et al. values remain available for provenance.

ROBERT abundance parameters are volume mixing ratios (VMRs).  A pRT
mass-fraction conversion belongs at an external oracle boundary only; no
mass-fraction retrieval parameter is defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping

from robert_exoplanets import Planet, Star


ROOT: Final = Path(__file__).resolve().parents[1]
TARGET_SLUG: Final = "wasp77ab"
DATA_DIRECTORY: Final = Path(
    os.environ.get(
        "ROBERT_WASP77AB_DATA",
        ROOT / "data" / "jwst_emission_spectra",
    )
).expanduser()
CACHE_DIRECTORY: Final = ROOT / "opacity_data" / "wasp77ab_real_joint"

GRAVITATIONAL_CONSTANT_M3_KG_S2: Final = 6.67430e-11
JUPITER_RADIUS_M: Final = 7.1492e7
JUPITER_MASS_KG: Final = 1.89813e27
SOLAR_RADIUS_M: Final = 6.957e8
SOLAR_MASS_KG: Final = 1.98847e30
AU_M: Final = 1.495978707e11

SMITH_PAPER_URL: Final = "https://arxiv.org/abs/2312.13069"
SMITH_PAPER_DOI: Final = "10.3847/1538-3881/ad17bf"
SMITH_ZENODO_URL: Final = "https://zenodo.org/records/10382053"
SMITH_ZENODO_DOI: Final = "10.5281/zenodo.10382053"
SMITH_ZENODO_API_FILES_URL: Final = (
    "https://zenodo.org/api/records/10382053/files"
)
AUGUST_PAPER_URL: Final = "https://arxiv.org/abs/2305.07753"
AUGUST_PAPER_DOI: Final = "10.3847/2041-8213/ace828"
AUGUST_DATA_DOI: Final = "10.17909/3fmp-zj55"
AUGUST_DATA_URL: Final = "https://doi.org/10.17909/3fmp-zj55"
NIRSPEC_ARCHIVE_URL: Final = (
    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/atmospheres/"
    "nph-firefly?atmospheres"
)


@dataclass(frozen=True)
class CitedMeasurement:
    """One nominal value with the source and units needed for provenance."""

    value: float | str
    unit: str
    source: str
    source_url: str
    uncertainty_minus: float | None = None
    uncertainty_plus: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("cited measurement value must be finite")
        for name in ("uncertainty_minus", "uncertainty_plus"):
            uncertainty = getattr(self, name)
            if uncertainty is not None and (
                not math.isfinite(uncertainty) or uncertainty < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class UniformPriorSpec:
    """A literature prior, expressed in the ROBERT parameter convention."""

    lower: float
    upper: float
    unit: str
    source: str = "Smith et al. (2024), Table 3"

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("prior bounds must be finite")
        if self.upper <= self.lower:
            raise ValueError("prior upper bound must be greater than lower bound")


@dataclass(frozen=True)
class NightPlan:
    """One Smith et al. IGRINS observing sequence."""

    label: str
    date: str
    phase_range: tuple[float, float]
    bjd_range: tuple[float, float]
    frame_count: int
    median_snr_h: float
    median_snr_k: float
    role: str
    primary_pc_count: int | None = None
    sensitivity_pc_counts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.phase_range) != 2 or self.phase_range[1] <= self.phase_range[0]:
            raise ValueError("phase_range must be an increasing pair")
        if len(self.bjd_range) != 2 or self.bjd_range[1] <= self.bjd_range[0]:
            raise ValueError("bjd_range must be an increasing pair")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if self.primary_pc_count is not None and self.primary_pc_count <= 0:
            raise ValueError("primary_pc_count must be positive")
        if any(count <= 0 for count in self.sensitivity_pc_counts):
            raise ValueError("sensitivity PC counts must be positive")


@dataclass(frozen=True)
class DataFile:
    """A public data file with hashes available before local acquisition."""

    name: str
    url: str | None
    size_bytes: int | None
    md5: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class SamplerContract:
    """The laptop-safe nested-sampling contract for real target runs."""

    sampler: str = "PyMultiNest"
    backend: str = "MultiNest"
    seeds: tuple[int, int] = (24680, 24681)
    mpi_processes: int = 1
    max_threads: int = 3
    process_rss_limit_bytes: int = 2 * 1024**3
    opacity_rss_limit_bytes: int = 1023 * 1024**2
    default_live_points: int = 64
    max_iter: int = 0
    evidence_repeatability_limit: float = 0.5
    thread_variables: tuple[str, ...] = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "OMP_THREAD_LIMIT",
    )


@dataclass(frozen=True)
class AcceptanceGate:
    """A hard gate that must be recorded in the run manifest."""

    requirement: str
    threshold: str
    failure_action: str


# Smith et al. (2024), Table 1.  All uncertainty fields are positive
# magnitudes.  ``source_url`` is the paper URL because Table 1 cites several
# literature sources; the source labels retain the distinction.
STAR_TABLE_1: Final[Mapping[str, CitedMeasurement]] = MappingProxyType(
    {
        "spectral_type": CitedMeasurement(
            "G8V",
            "spectral type",
            "Smith et al. (2024), Table 1 (Maxted et al. 2013)",
            SMITH_PAPER_URL,
        ),
        "radius": CitedMeasurement(
            0.955,
            "R_sun",
            "Smith et al. (2024), Table 1 (Bonomo et al. 2017)",
            SMITH_PAPER_URL,
        ),
        "mass": CitedMeasurement(
            1.002,
            "M_sun",
            "Smith et al. (2024), Table 1 (Maxted et al. 2013)",
            SMITH_PAPER_URL,
            uncertainty_minus=0.045,
            uncertainty_plus=0.045,
        ),
        "effective_temperature": CitedMeasurement(
            5605.0,
            "K",
            "Smith et al. (2024), Table 1 (Maxted et al. 2013)",
            SMITH_PAPER_URL,
        ),
        "log_g": CitedMeasurement(
            4.33,
            "dex (cgs)",
            "Smith et al. (2024), Table 1 (Maxted et al. 2013)",
            SMITH_PAPER_URL,
            uncertainty_minus=0.08,
            uncertainty_plus=0.08,
        ),
        "metallicity": CitedMeasurement(
            0.0,
            "dex",
            "Smith et al. (2024), Table 1 (Maxted et al. 2013)",
            SMITH_PAPER_URL,
            uncertainty_minus=0.11,
            uncertainty_plus=0.11,
        ),
        "systemic_velocity": CitedMeasurement(
            1.6845,
            "km s^-1",
            "Smith et al. (2024), Table 1 (Maxted et al. 2013)",
            SMITH_PAPER_URL,
            uncertainty_minus=0.0004,
            uncertainty_plus=0.0004,
        ),
    }
)

PLANET_TABLE_1: Final[Mapping[str, CitedMeasurement]] = MappingProxyType(
    {
        "radius": CitedMeasurement(
            1.21,
            "R_J",
            "Smith et al. (2024), Table 1 (Maxted et al. 2013)",
            SMITH_PAPER_URL,
            uncertainty_minus=0.02,
            uncertainty_plus=0.02,
        ),
        "mass": CitedMeasurement(
            1.76,
            "M_J",
            "Smith et al. (2024), Table 1 (Maxted et al. 2013)",
            SMITH_PAPER_URL,
            uncertainty_minus=0.06,
            uncertainty_plus=0.06,
        ),
        "equilibrium_temperature": CitedMeasurement(
            1740.0,
            "K",
            "Smith et al. (2024), Table 1 (Maxted et al. 2013)",
            SMITH_PAPER_URL,
        ),
        "planet_velocity_semi_amplitude": CitedMeasurement(
            192.0,
            "km s^-1",
            "Smith et al. (2024), Table 1",
            SMITH_PAPER_URL,
            uncertainty_minus=4.5,
            uncertainty_plus=4.5,
        ),
        "semi_major_axis": CitedMeasurement(
            0.02405,
            "AU",
            "Smith et al. (2024), Table 1 (Maxted et al. 2013)",
            SMITH_PAPER_URL,
            uncertainty_minus=0.00036,
            uncertainty_plus=0.00036,
        ),
        "eccentricity": CitedMeasurement(
            0.0074,
            "dimensionless",
            "Smith et al. (2024), Table 1 (Cortés-Zuleta et al. update)",
            SMITH_PAPER_URL,
            uncertainty_minus=0.005,
            uncertainty_plus=0.007,
        ),
    }
)

ORBIT_TABLE_1: Final[Mapping[str, CitedMeasurement]] = MappingProxyType(
    {
        "transit_epoch_maxted": CitedMeasurement(
            2455870.44977,
            "BJD",
            "Smith et al. (2024), Table 1 (Maxted et al. 2013)",
            SMITH_PAPER_URL,
            uncertainty_minus=0.00014,
            uncertainty_plus=0.00014,
        ),
        "transit_epoch_cortes": CitedMeasurement(
            2457420.88439,
            "BJD",
            "Smith et al. (2024), Table 1 (Cortés-Zuleta et al. update)",
            SMITH_PAPER_URL,
            uncertainty_minus=0.00085,
            uncertainty_plus=0.00080,
        ),
        "period_maxted": CitedMeasurement(
            1.3600309,
            "day",
            "Smith et al. (2024), Table 1 (Maxted et al. 2013)",
            SMITH_PAPER_URL,
            uncertainty_minus=0.0000020,
            uncertainty_plus=0.0000020,
        ),
        "period_cortes": CitedMeasurement(
            1.36002854,
            "day",
            "Smith et al. (2024), Table 1 (Cortés-Zuleta et al. update)",
            SMITH_PAPER_URL,
            uncertainty_minus=0.00000062,
            uncertainty_plus=0.00000062,
        ),
    }
)

# Direct scalar aliases make the target module easy to use from lightweight
# examples while the mappings above preserve source and unit metadata.
STAR_RADIUS_SOLAR: Final = STAR_TABLE_1["radius"].value
STAR_MASS_SOLAR: Final = STAR_TABLE_1["mass"].value
STAR_EFFECTIVE_TEMPERATURE_K: Final = STAR_TABLE_1["effective_temperature"].value
STAR_LOG_G_CGS: Final = STAR_TABLE_1["log_g"].value
STAR_METALLICITY_DEX: Final = STAR_TABLE_1["metallicity"].value
SYSTEMIC_VELOCITY_KM_S: Final = STAR_TABLE_1["systemic_velocity"].value
PLANET_RADIUS_JUPITER: Final = PLANET_TABLE_1["radius"].value
PLANET_MASS_JUPITER: Final = PLANET_TABLE_1["mass"].value
PLANET_EQUILIBRIUM_TEMPERATURE_K: Final = PLANET_TABLE_1[
    "equilibrium_temperature"
].value
PLANET_KP_KM_S: Final = PLANET_TABLE_1["planet_velocity_semi_amplitude"].value
PLANET_SEMI_MAJOR_AXIS_AU: Final = PLANET_TABLE_1["semi_major_axis"].value

# Use the current Cortés-Zuleta values for the circular default.  Keeping the
# Maxted values explicit prevents accidental mixing of ephemerides.
TRANSIT_EPOCH_BJD: Final = ORBIT_TABLE_1["transit_epoch_cortes"].value
ORBITAL_PERIOD_DAY: Final = ORBIT_TABLE_1["period_cortes"].value
TRANSIT_EPOCH_BJD_MAXTED: Final = ORBIT_TABLE_1["transit_epoch_maxted"].value
ORBITAL_PERIOD_DAY_MAXTED: Final = ORBIT_TABLE_1["period_maxted"].value
ORBITAL_ECCENTRICITY_REFERENCE: Final = PLANET_TABLE_1["eccentricity"].value
ORBIT_DEFAULT: Final = "circular"

PLANET: Final = Planet(
    name="WASP-77Ab",
    radius_m=PLANET_RADIUS_JUPITER * JUPITER_RADIUS_M,
    mass_kg=PLANET_MASS_JUPITER * JUPITER_MASS_KG,
    semi_major_axis_m=PLANET_SEMI_MAJOR_AXIS_AU * AU_M,
    metadata={
        "source": "Smith et al. (2024), Table 1",
        "source_url": SMITH_PAPER_URL,
        "radius_unit": "R_J",
        "mass_unit": "M_J",
        "semi_major_axis_unit": "AU",
    },
)
STAR: Final = Star(
    name="WASP-77A",
    radius_m=STAR_RADIUS_SOLAR * SOLAR_RADIUS_M,
    effective_temperature_k=STAR_EFFECTIVE_TEMPERATURE_K,
    log_g_cgs=STAR_LOG_G_CGS,
    metallicity_dex=STAR_METALLICITY_DEX,
    metadata={
        "source": "Smith et al. (2024), Table 1",
        "source_url": SMITH_PAPER_URL,
        "radius_unit": "R_sun",
        "effective_temperature_unit": "K",
        "log_g_unit": "dex (cgs)",
    },
)
PLANET_GRAVITY_M_S2: Final = (
    GRAVITATIONAL_CONSTANT_M3_KG_S2 * PLANET.mass_kg / PLANET.radius_m**2
)


SMITH_NIGHTS: Final[tuple[NightPlan, ...]] = (
    NightPlan(
        label="igrins_2020-12-06_post_eclipse",
        date="2020-12-06",
        phase_range=(0.535, 0.605),
        bjd_range=(2459189.61326, 2459189.75917),
        frame_count=39,
        median_snr_h=195.0,
        median_snr_k=185.0,
        role="post_eclipse_sensitivity",
        primary_pc_count=3,
        sensitivity_pc_counts=(2, 3, 4, 6, 8),
    ),
    NightPlan(
        label="igrins_2020-12-14_pre_eclipse",
        date="2020-12-14",
        phase_range=(0.325, 0.470),
        bjd_range=(2459197.53017, 2459197.74095),
        frame_count=79,
        median_snr_h=205.0,
        median_snr_k=190.0,
        role="primary_pre_eclipse",
        primary_pc_count=4,
        sensitivity_pc_counts=(2, 3, 4, 6, 8),
    ),
    NightPlan(
        label="igrins_2020-12-21_post_eclipse",
        date="2020-12-21",
        phase_range=(0.535, 0.628),
        bjd_range=(2459204.61064, 2459204.74449),
        frame_count=48,
        median_snr_h=175.0,
        median_snr_k=165.0,
        role="post_eclipse_sensitivity",
        primary_pc_count=3,
        sensitivity_pc_counts=(2, 3, 4, 6, 8),
    ),
)
PRIMARY_HRS_NIGHT: Final = "igrins_2020-12-14_pre_eclipse"
PRIMARY_HRS_PC_COUNT: Final = 4
POST_ECLIPSE_PRIMARY_PC_COUNT: Final = 3
HRS_PC_SENSITIVITY_COUNTS: Final[tuple[int, ...]] = (2, 3, 4, 6, 8)
HRS_DISCARDED_EDGE_PIXELS: Final = 200
HRS_DISCARDED_ORDER_COUNT: Final = 8
HRS_INSTRUMENT_RESOLVING_POWER: Final = 45_000
HRS_MODEL_RESOLVING_POWER: Final = 250_000
HRS_VSIN_I_KM_S: Final = 4.2

NIRSPEC_DETECTORS: Final[tuple[str, str]] = ("NRS1", "NRS2")
NIRSPEC_INSTRUMENT_RANGES_MICRON: Final[Mapping[str, tuple[float, float]]] = (
    MappingProxyType(
        {
            "NRS1": (2.674, 3.716),
            "NRS2": (3.827, 5.173),
        }
    )
)
NIRSPEC_PUBLISHED_TABLE_POINTS: Final = 150
NIRSPEC_SMITH_TEXT_PREDICTIVE_POINTS: Final = 160
NIRSPEC_POINT_COUNT_DISCREPANCY: Final = True
NIRSPEC_POINT_COUNT_RECONCILED: Final = True
NIRSPEC_POINT_COUNT_STATUS: Final = (
    "reconciled: August table has 150 rows and the public Zenodo best-fit "
    "template reproduces Smith's reported NIRSpec chi2/N when binned to "
    "those 150 rows; Smith's N=160 text count is retained as a manuscript "
    "discrepancy"
)


SMITH_PRIORS: Final[Mapping[str, UniformPriorSpec]] = MappingProxyType(
    {
        "log10_h2o_vmr": UniformPriorSpec(-12.0, 0.0, "dex"),
        "log10_co_vmr": UniformPriorSpec(-12.0, 0.0, "dex"),
        "log10_ch4_vmr": UniformPriorSpec(-12.0, 0.0, "dex"),
        "log10_h2s_vmr": UniformPriorSpec(-12.0, 0.0, "dex"),
        "log10_nh3_vmr": UniformPriorSpec(-12.0, 0.0, "dex"),
        "log10_hcn_vmr": UniformPriorSpec(-12.0, 0.0, "dex"),
        "log10_co2_vmr": UniformPriorSpec(-12.0, 0.0, "dex"),
        "iso_13co_12co_earth": UniformPriorSpec(-5.0, 5.0, "dex"),
        "T0_K": UniformPriorSpec(500.0, 2500.0, "K"),
        "log10_P1_bar": UniformPriorSpec(-5.5, 2.5, "dex"),
        "log10_P2_bar": UniformPriorSpec(-5.5, 2.5, "dex"),
        "log10_P3_bar": UniformPriorSpec(-2.0, 2.0, "dex"),
        "alpha1": UniformPriorSpec(0.02, 2.0, "dimensionless"),
        "alpha2": UniformPriorSpec(0.02, 2.0, "dimensionless"),
        "dKp_km_s": UniformPriorSpec(-20.0, 20.0, "km s^-1"),
        "dVsys_km_s": UniformPriorSpec(-20.0, 20.0, "km s^-1"),
        "log10_a": UniformPriorSpec(-2.0, 2.0, "dex"),
    }
)
SMITH_TABLE_3_PRIORS: Final = SMITH_PRIORS
SMITH_LIVE_POINTS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "hrs_only": 500,
        "lrs_only_or_joint": 2500,
    }
)

HMINUS_SPECIES: Final[tuple[str, str, str]] = ("H-", "H", "e-")
HMINUS_POLICY: Final[Mapping[str, object]] = MappingProxyType(
    {
        "retrievable": True,
        "state_abundance_type": "VMR",
        "species": HMINUS_SPECIES,
        "saha_closure": False,
        "mass_fraction_parameters_in_robert": False,
        "prior_policy": (
            "freeze a broad bounded log10-VMR prior before sampling and repeat "
            "with an alternate bound"
        ),
        "lrs_and_hrs": True,
    }
)
ROBERT_ABUNDANCE_POLICY: Final = (
    "ROBERT retrieval composition is VMR-only; pRT mass fractions are allowed "
    "only as a deterministic external-boundary conversion"
)
SAMPLER_POLICY: Final = (
    "PyMultiNest with the MultiNest backend is required; UltraNest is prohibited"
)

SMITH_ZENODO_FILES: Final[Mapping[str, DataFile]] = MappingProxyType(
    {
        "cube06v3.pic": DataFile(
            "cube06v3.pic",
            f"{SMITH_ZENODO_API_FILES_URL}/cube06v3.pic/content",
            55_221_925,
            md5="fa338b902cfa5127fa4707187c2d00f0",
        ),
        "cube14v3.pic": DataFile(
            "cube14v3.pic",
            f"{SMITH_ZENODO_API_FILES_URL}/cube14v3.pic/content",
            77_591_894,
            md5="3fd41560b7dbcc60851ef3a0972c852f",
        ),
        "cube21v3.pic": DataFile(
            "cube21v3.pic",
            f"{SMITH_ZENODO_API_FILES_URL}/cube21v3.pic/content",
            50_346_161,
            md5="53f3c67601475f0f5ffc8d1b9277c304",
        ),
        "20201206_info.csv": DataFile(
            "20201206_info.csv",
            f"{SMITH_ZENODO_API_FILES_URL}/20201206_info.csv/content",
            5_109,
            md5="ec2f3cbc0f0d1dc7a4cec599224cd186",
        ),
        "20201214_info.csv": DataFile(
            "20201214_info.csv",
            f"{SMITH_ZENODO_API_FILES_URL}/20201214_info.csv/content",
            7_261,
            md5="7c861b1baf8ae87801f95d7af8b2e9c6",
        ),
        "20201221_info.csv": DataFile(
            "20201221_info.csv",
            f"{SMITH_ZENODO_API_FILES_URL}/20201221_info.csv/content",
            4_529,
            md5="6bba581a3eaaa5bdd90286264ad202e0",
        ),
        "w77_1DRC_FULL.txt": DataFile(
            "w77_1DRC_FULL.txt",
            f"{SMITH_ZENODO_API_FILES_URL}/w77_1DRC_FULL.txt/content",
            13_899_150,
            md5="dbaa138b7768b727c87ff8f83fcc563f",
        ),
        "w77_1DRC_H2O_ONLY.txt": DataFile(
            "w77_1DRC_H2O_ONLY.txt",
            f"{SMITH_ZENODO_API_FILES_URL}/w77_1DRC_H2O_ONLY.txt/content",
            13_899_150,
            md5="6cf0ec37cbed5db5b18e6d434838fa40",
        ),
        "w77_1DRC_CO_ONLY.txt": DataFile(
            "w77_1DRC_CO_ONLY.txt",
            f"{SMITH_ZENODO_API_FILES_URL}/w77_1DRC_CO_ONLY.txt/content",
            13_899_150,
            md5="d4f0c6dfd4bb50a7ca9ca73ea3318eb3",
        ),
        "w77_1DRC_13CO_ONLY.txt": DataFile(
            "w77_1DRC_13CO_ONLY.txt",
            f"{SMITH_ZENODO_API_FILES_URL}/w77_1DRC_13CO_ONLY.txt/content",
            13_899_150,
            md5="59ec25e625109995f8c702874b771f7c",
        ),
        "w77_1DRC_OTHER_GAS.txt": DataFile(
            "w77_1DRC_OTHER_GAS.txt",
            f"{SMITH_ZENODO_API_FILES_URL}/w77_1DRC_OTHER_GAS.txt/content",
            13_899_150,
            md5="b21937c15985f0fe69abf63ba6403350",
        ),
        "w77_pre_nirspec_best_fit_R500K_scaled.txt": DataFile(
            "w77_pre_nirspec_best_fit_R500K_scaled.txt",
            f"{SMITH_ZENODO_API_FILES_URL}/"
            "w77_pre_nirspec_best_fit_R500K_scaled.txt/content",
            72_672_825,
            md5="93841125cab0a2c74ae4730ebf36b7dc",
        ),
    }
)
NIRSPEC_TABLE: Final = DataFile(
    "WASP_77_A_b_3.11569_5329_1.tbl",
    AUGUST_DATA_URL,
    27_329,
    sha256="96159824870eb7c8132fd9c8bee30caf9ec95f05e128dbe2fea8627909a40d3f",
)

SMITH_COMPARISON_ANCHORS: Final[Mapping[str, Mapping[str, CitedMeasurement]]] = (
    MappingProxyType(
        {
            "hrs_pre": MappingProxyType(
                {
                    "log10_h2o_vmr": CitedMeasurement(
                        -3.97,
                        "dex",
                        "Smith et al. (2024), Table 3, IGRINS pre-eclipse",
                        SMITH_PAPER_URL,
                        uncertainty_minus=0.10,
                        uncertainty_plus=0.11,
                    ),
                    "log10_co_vmr": CitedMeasurement(
                        -3.81,
                        "dex",
                        "Smith et al. (2024), Table 3, IGRINS pre-eclipse",
                        SMITH_PAPER_URL,
                        uncertainty_minus=0.17,
                        uncertainty_plus=0.19,
                    ),
                    "T0_K": CitedMeasurement(
                        1470.0,
                        "K",
                        "Smith et al. (2024), Table 3, IGRINS pre-eclipse",
                        SMITH_PAPER_URL,
                        uncertainty_minus=370.0,
                        uncertainty_plus=210.0,
                    ),
                    "C_to_O": CitedMeasurement(
                        0.59,
                        "ratio",
                        "Smith et al. (2024), Table 3, IGRINS pre-eclipse",
                        SMITH_PAPER_URL,
                        uncertainty_minus=0.07,
                        uncertainty_plus=0.07,
                    ),
                    "metallicity_C_plus_O_H": CitedMeasurement(
                        -0.53,
                        "dex",
                        "Smith et al. (2024), Table 3, IGRINS pre-eclipse",
                        SMITH_PAPER_URL,
                        uncertainty_minus=0.13,
                        uncertainty_plus=0.16,
                    ),
                    "dKp_km_s": CitedMeasurement(
                        -1.26,
                        "km s^-1",
                        "Smith et al. (2024), Table 3, IGRINS pre-eclipse",
                        SMITH_PAPER_URL,
                        uncertainty_minus=0.85,
                        uncertainty_plus=0.85,
                    ),
                    "dVsys_km_s": CitedMeasurement(
                        -5.27,
                        "km s^-1",
                        "Smith et al. (2024), Table 3, IGRINS pre-eclipse",
                        SMITH_PAPER_URL,
                        uncertainty_minus=0.53,
                        uncertainty_plus=0.51,
                    ),
                }
            ),
            "lrs_nirspec": MappingProxyType(
                {
                    "log10_h2o_vmr": CitedMeasurement(
                        -3.80,
                        "dex",
                        "Smith et al. (2024), Table 3, NIRSpec",
                        SMITH_PAPER_URL,
                        uncertainty_minus=0.28,
                        uncertainty_plus=0.34,
                    ),
                    "log10_co_vmr": CitedMeasurement(
                        -3.73,
                        "dex",
                        "Smith et al. (2024), Table 3, NIRSpec",
                        SMITH_PAPER_URL,
                        uncertainty_minus=0.52,
                        uncertainty_plus=0.55,
                    ),
                    "T0_K": CitedMeasurement(
                        1300.0,
                        "K",
                        "Smith et al. (2024), Table 3, NIRSpec",
                        SMITH_PAPER_URL,
                        uncertainty_minus=50.0,
                        uncertainty_plus=80.0,
                    ),
                    "C_to_O": CitedMeasurement(
                        0.54,
                        "ratio",
                        "Smith et al. (2024), Table 3, NIRSpec",
                        SMITH_PAPER_URL,
                        uncertainty_minus=0.16,
                        uncertainty_plus=0.15,
                    ),
                    "metallicity_C_plus_O_H": CitedMeasurement(
                        -0.42,
                        "dex",
                        "Smith et al. (2024), Table 3, NIRSpec",
                        SMITH_PAPER_URL,
                        uncertainty_minus=0.42,
                        uncertainty_plus=0.49,
                    ),
                }
            ),
            "joint_pre_nirspec": MappingProxyType(
                {
                    "log10_h2o_vmr": CitedMeasurement(
                        -4.02,
                        "dex",
                        "Smith et al. (2024), Table 3, IGRINS pre + NIRSpec",
                        SMITH_PAPER_URL,
                        uncertainty_minus=0.06,
                        uncertainty_plus=0.07,
                    ),
                    "log10_co_vmr": CitedMeasurement(
                        -3.91,
                        "dex",
                        "Smith et al. (2024), Table 3, IGRINS pre + NIRSpec",
                        SMITH_PAPER_URL,
                        uncertainty_minus=0.13,
                        uncertainty_plus=0.13,
                    ),
                    "T0_K": CitedMeasurement(
                        1400.0,
                        "K",
                        "Smith et al. (2024), Table 3, IGRINS pre + NIRSpec",
                        SMITH_PAPER_URL,
                        uncertainty_minus=40.0,
                        uncertainty_plus=30.0,
                    ),
                    "C_to_O": CitedMeasurement(
                        0.57,
                        "ratio",
                        "Smith et al. (2024), Table 3, IGRINS pre + NIRSpec",
                        SMITH_PAPER_URL,
                        uncertainty_minus=0.06,
                        uncertainty_plus=0.06,
                    ),
                    "metallicity_C_plus_O_H": CitedMeasurement(
                        -0.61,
                        "dex",
                        "Smith et al. (2024), Table 3, IGRINS pre + NIRSpec",
                        SMITH_PAPER_URL,
                        uncertainty_minus=0.10,
                        uncertainty_plus=0.11,
                    ),
                    "dKp_km_s": CitedMeasurement(
                        -1.29,
                        "km s^-1",
                        "Smith et al. (2024), Table 3, IGRINS pre + NIRSpec",
                        SMITH_PAPER_URL,
                        uncertainty_minus=0.78,
                        uncertainty_plus=0.75,
                    ),
                    "dVsys_km_s": CitedMeasurement(
                        -5.27,
                        "km s^-1",
                        "Smith et al. (2024), Table 3, IGRINS pre + NIRSpec",
                        SMITH_PAPER_URL,
                        uncertainty_minus=0.47,
                        uncertainty_plus=0.48,
                    ),
                }
            ),
        }
    )
)
SMITH_JOINT_ANCHORS: Final = SMITH_COMPARISON_ANCHORS["joint_pre_nirspec"]

SAMPLER_CONTRACT: Final = SamplerContract()
ACCEPTANCE_GATES: Final[Mapping[str, AcceptanceGate]] = MappingProxyType(
    {
        "provenance": AcceptanceGate(
            "Record exact source URLs, versions, file hashes, and data role.",
            "No unknown input may enter a claimed reproduction.",
            "Stop and label the run an independent comparison.",
        ),
        "data_preflight": AcceptanceGate(
            "Validate finite arrays, monotonic grids, masks, edges, phases, and RV inputs.",
            "Zero invalid unmasked values and exact dataset metadata.",
            "Do not start sampling.",
        ),
        "injection_recovery": AcceptanceGate(
            "Recover known H2O, CO, H-minus, temperature, and velocity injections.",
            "Central 90% truth coverage; abundance bias <=0.25 dex; velocity bias <=1 km/s.",
            "Fix the adapter or reject the configuration.",
        ),
        "sampler_repeatability": AcceptanceGate(
            "Run two independent PyMultiNest seeds with the same configuration.",
            "Both converge and abs(delta log Z) <=0.5.",
            "Reject the production posterior.",
        ),
        "resources": AcceptanceGate(
            "Keep one MPI process, <=3 numerical threads, bounded caches, and one order/night at a time.",
            "Process RSS <2 GiB and LBL opacity RSS <1023 MiB.",
            "Stop before or during the run and record the failure.",
        ),
        "comparison": AcceptanceGate(
            "Compare posterior predictions and derived values against Smith anchors.",
            "Use declared model-source caveats; do not require exact posterior equality.",
            "Report an independent comparison, not a reproduction.",
        ),
    }
)

INDEPENDENT_COMPARISON_WORDING: Final = (
    "Until the exact Smith matrices, masks, SVD/model-injection operation, "
    "velocity corrections, likelihood normalisation, and Smith-used NIRSpec "
    "table are confirmed, this is an independent ROBERT comparison, not a "
    "reproduction of Smith et al. (2024)."
)
REPRODUCTION_REQUIREMENTS: Final[tuple[str, ...]] = (
    "exact Zenodo matrix schema, order selection, and masks",
    "exact SVD recomposition and model-injection/scaling operation",
    "exact BJD, coordinates, observatory, ephemeris, and barycentric correction",
    "exact HRS and LRS likelihood normalisation and error treatment",
    "exact Smith-used NIRSpec table, detector split, bin edges, and covariance",
    "opacity and stellar-spectrum versions used by Smith",
)


def load_nirspec_observations(*, verify_checksum: bool = True):
    """Lazily load the checked-in 150-row NIRSpec comparison table.

    The import is intentionally local.  This target module can therefore be
    used for metadata and tests without importing a data loader or sampler.
    The returned table is the August et al. (2023) comparison input, not a
    Smith-produced table; the 150-versus-160 discrepancy is retained in the
    metadata even though the public 150-row table is the reconciled choice.
    """

    from robert_exoplanets import load_august2023_wasp77ab

    return load_august2023_wasp77ab(
        DATA_DIRECTORY,
        verify_checksum=verify_checksum,
    )


__all__ = [
    "ACCEPTANCE_GATES",
    "AU_M",
    "AUGUST_PAPER_DOI",
    "AUGUST_PAPER_URL",
    "AUGUST_DATA_DOI",
    "AUGUST_DATA_URL",
    "CACHE_DIRECTORY",
    "CitedMeasurement",
    "DATA_DIRECTORY",
    "DataFile",
    "HMINUS_POLICY",
    "HMINUS_SPECIES",
    "HRS_DISCARDED_EDGE_PIXELS",
    "HRS_DISCARDED_ORDER_COUNT",
    "HRS_INSTRUMENT_RESOLVING_POWER",
    "HRS_MODEL_RESOLVING_POWER",
    "HRS_PC_SENSITIVITY_COUNTS",
    "HRS_VSIN_I_KM_S",
    "INDEPENDENT_COMPARISON_WORDING",
    "NIRSPEC_ARCHIVE_URL",
    "NIRSPEC_DETECTORS",
    "NIRSPEC_INSTRUMENT_RANGES_MICRON",
    "NIRSPEC_POINT_COUNT_STATUS",
    "NIRSPEC_POINT_COUNT_DISCREPANCY",
    "NIRSPEC_POINT_COUNT_RECONCILED",
    "NIRSPEC_PUBLISHED_TABLE_POINTS",
    "NIRSPEC_SMITH_TEXT_PREDICTIVE_POINTS",
    "NIRSPEC_TABLE",
    "ORBITAL_ECCENTRICITY_REFERENCE",
    "ORBITAL_PERIOD_DAY",
    "ORBITAL_PERIOD_DAY_MAXTED",
    "ORBIT_DEFAULT",
    "ORBIT_TABLE_1",
    "PLANET",
    "PLANET_EQUILIBRIUM_TEMPERATURE_K",
    "PLANET_GRAVITY_M_S2",
    "PLANET_KP_KM_S",
    "PLANET_MASS_JUPITER",
    "PLANET_RADIUS_JUPITER",
    "PLANET_SEMI_MAJOR_AXIS_AU",
    "PLANET_TABLE_1",
    "POST_ECLIPSE_PRIMARY_PC_COUNT",
    "PRIMARY_HRS_NIGHT",
    "PRIMARY_HRS_PC_COUNT",
    "REPRODUCTION_REQUIREMENTS",
    "ROBERT_ABUNDANCE_POLICY",
    "SAMPLER_CONTRACT",
    "SAMPLER_POLICY",
    "SMITH_COMPARISON_ANCHORS",
    "SMITH_JOINT_ANCHORS",
    "SMITH_LIVE_POINTS",
    "SMITH_NIGHTS",
    "SMITH_PAPER_DOI",
    "SMITH_PAPER_URL",
    "SMITH_PRIORS",
    "SMITH_TABLE_3_PRIORS",
    "SMITH_ZENODO_DOI",
    "SMITH_ZENODO_API_FILES_URL",
    "SMITH_ZENODO_FILES",
    "SMITH_ZENODO_URL",
    "STAR",
    "STAR_EFFECTIVE_TEMPERATURE_K",
    "STAR_LOG_G_CGS",
    "STAR_MASS_SOLAR",
    "STAR_METALLICITY_DEX",
    "STAR_RADIUS_SOLAR",
    "STAR_TABLE_1",
    "SYSTEMIC_VELOCITY_KM_S",
    "TARGET_SLUG",
    "TRANSIT_EPOCH_BJD",
    "TRANSIT_EPOCH_BJD_MAXTED",
    "UniformPriorSpec",
    "load_nirspec_observations",
]
