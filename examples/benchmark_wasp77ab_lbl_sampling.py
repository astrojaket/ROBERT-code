"""Benchmark LBL wavelength sampling on the real WASP-77Ab grids.

This is a convergence check for the planned Smith et al. (2024) joint
HRS/LRS comparison.  It uses the real 2020-12-14 IGRINS wavelength pixels
and the August et al. (2023) NIRSpec bins, but it does not fit the measured
fluxes.  The Smith cube is used for its order and pixel grids only.

The reference is evaluated with the finest available LBL table sampling one
order at a time.  Candidate HRS strides are compared after Gray rotational
broadening, the IGRINS Gaussian LSF, and pixel integration.  Candidate LRS
strides are compared after exact NRS1/NRS2 top-hat integration.  The report
contains metrics and provenance only; it never stores spectra.

ROBERT abundances in this example are volume mixing ratios (VMRs).  The
explicit H-minus, H, and electron VMRs are fixed for this convergence test.
No mass-fraction parameter or conversion is used.
"""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import tempfile
from time import perf_counter
from typing import Mapping, Sequence


# Set numerical libraries before NumPy or HDF5 is imported.  The command used
# for the recorded run sets all of these to one; the script clamps an inherited
# value to the repository-wide maximum of three.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)


def _configure_thread_limits() -> None:
    """Clamp all configured numerical thread variables to one through three."""

    for name in THREAD_VARIABLES:
        raw_value = os.environ.get(name, "3")
        try:
            value = int(raw_value)
        except ValueError:
            value = 3
        os.environ[name] = str(min(3, max(1, value)))


_configure_thread_limits()
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib")
)

import numpy as np  # noqa: E402

from examples import wasp77ab_target as target  # noqa: E402
from robert_exoplanets import (  # noqa: E402
    AtmosphereState,
    HMinusContinuumConfig,
    Observation,
    PressureGrid,
    SpectralGrid,
    Spectrum,
    TopHatObservationResponse,
    HighResolutionOrder,
    LineByLineOpacityProvider,
    gauss_legendre_disk_geometry,
    load_august2023_wasp77ab,
    solve_emission_spectrum,
)
from robert_exoplanets.instruments import infer_wavelength_bin_edges  # noqa: E402
from robert_exoplanets.forward._atmospheric import (  # noqa: E402
    evaluate_gas_optical_depth,
)
from robert_exoplanets.io.smith2024_wasp77ab import (  # noqa: E402
    SMITH2024_WASP77AB_CUBE14_MD5,
    SMITH2024_WASP77AB_CUBE14_SHA256,
    SMITH2024_WASP77AB_CUBE14_SIZE,
    SMITH2024_WASP77AB_INFO_MD5,
    SMITH2024_WASP77AB_INFO_SHA256,
    SMITH2024_WASP77AB_INFO_SIZE,
    load_smith2024_wasp77ab_hrs,
)
from robert_exoplanets.rt import hminus_optical_depth  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "docs" / "data" / "wasp77ab_lbl_sampling_20260831.json"
DEFAULT_HRS_DIRECTORY = ROOT / "external_data" / "wasp77ab_smith2024"
DEFAULT_HRS_CUBE = DEFAULT_HRS_DIRECTORY / "cube14v3.pic"
DEFAULT_HRS_INFO = DEFAULT_HRS_DIRECTORY / "20201214_info.csv"
DEFAULT_LRS_ROOT = ROOT / "data" / "jwst_emission_spectra"
H2O_TABLE = (
    ROOT
    / "external_data"
    / "petitRADTRANS"
    / "input_data"
    / "opacities/lines/line_by_line/H2O/1H2-16O"
    / "1H2-16O__POKAZATEL.R1e6_0.3-28mu.xsec.petitRADTRANS.h5"
)
CO_TABLE = (
    ROOT
    / "external_data"
    / "petitRADTRANS"
    / "input_data"
    / "opacities/lines/line_by_line/CO/12C-16O"
    / "12C-16O__HITEMP.R1e6_0.3-28mu.xsec.petitRADTRANS.h5"
)

HRS_K_BAND_RANGE_MICRON = (1.90, 2.45)
HRS_SOURCE_MARGIN_MICRON = 0.002
HRS_RESPONSE_STAGES = (
    "rotational-broadening",
    "gaussian-high-resolution",
    "pixel-bin-integration",
)
HRS_LBL_STRIDES = (1, 2, 4, 6, 8)
HRS_REFERENCE_STRIDE = 1
HRS_CANDIDATE_STRIDES = HRS_LBL_STRIDES[1:]
HRS_LSF_RESOLVING_POWER = float(target.HRS_INSTRUMENT_RESOLVING_POWER)
HRS_ROTATION_KM_S = float(target.HRS_VSIN_I_KM_S)
HRS_LIMB_DARKENING = 0.6
HRS_LINE_SHAPE_RMS_LIMIT = 5.0e-3
HRS_LINE_SHAPE_MAX_LIMIT = 5.0e-2

LRS_BOUNDS_MICRON = (2.808, 5.168)
LRS_SOURCE_MARGIN_MICRON = 0.018
LRS_LBL_STRIDES = (10, 25, 50, 100, 250)
LRS_REFERENCE_STRIDE = 10
LRS_CANDIDATE_STRIDES = LRS_LBL_STRIDES[1:]
LRS_SIGMA_RMS_LIMIT = 0.25
LRS_SIGMA_MAX_LIMIT = 1.0

PRESSURE_FIRST_CENTER_BAR = 1.0e-5
PRESSURE_LAST_CENTER_BAR = 100.0
ATMOSPHERE_LAYERS = 80
TEMPERATURE_TOP_K = 2600.0
TEMPERATURE_BOTTOM_K = 2900.0
GRAVITY_M_S2 = float(target.PLANET_GRAVITY_M_S2)
H2O_LOG10_VMR = -4.02
CO_LOG10_VMR = -3.91
H2O_VMR = 10.0**H2O_LOG10_VMR
CO_VMR = 10.0**CO_LOG10_VMR
HMINUS_VMR = 1.0e-6
HYDROGEN_VMR = 1.0e-2
ELECTRON_VMR = 1.0e-3
BACKGROUND_H2_FRACTION = 0.85
HMINUS_CONFIG = HMinusContinuumConfig(
    temperature_extrapolation="raise",
    spectral_extrapolation="raise",
    metadata={
        "benchmark_state": "fixed_explicit_vmr",
        "convergence_scope": "included_in_every_lbl_stride",
    },
)
MOLECULAR_MASSES_AMU = {
    "H2": 2.01588,
    "He": 4.002602,
    "H2O": 18.010565,
    "CO": 28.0101,
    "H-": 1.00054858,
    "H": 1.00794,
    "e-": 5.485799096e-4,
}

OPACITY_MEMORY_LIMIT_BYTES = 1023 * 1024**2
PROCESS_MEMORY_LIMIT_BYTES = int(1.9 * 1024**3)
MAX_THREADS = 3
RUN_CPU_COUNT = 1
PLANCK_H_J_S = 6.62607015e-34
PLANCK_C_M_S = 299_792_458.0
PLANCK_K_J_K = 1.380649e-23
TABLE_SHA256 = {
    "H2O__POKAZATEL": "9a79513fb92dfa369f85abb273257ae2b2100a4a386169b9bfff8a5b74249b73",
    "CO__HITEMP": "5f77209da3ea67d5a697b06379b1de2b8fe106adf7ea0abfb451c675084087ad",
}


def _thread_values() -> dict[str, int]:
    """Return the clamped thread settings for the report."""

    return {name: int(os.environ[name]) for name in THREAD_VARIABLES}


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _reproducible_command(argv: Sequence[str] | None = None) -> str:
    """Return the explicit conda command used for this invocation."""

    script = str(Path(__file__).resolve().relative_to(ROOT))
    return shlex.join(
        (
            "conda",
            "run",
            "-n",
            "robert-exoplanets",
            "python",
            script,
            *(sys.argv[1:] if argv is None else argv),
        )
    )


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def external_inputs_available(
    *,
    hrs_cube: Path = DEFAULT_HRS_CUBE,
    hrs_info: Path = DEFAULT_HRS_INFO,
    h2o_table: Path = H2O_TABLE,
    co_table: Path = CO_TABLE,
    lrs_root: Path = DEFAULT_LRS_ROOT,
) -> bool:
    """Return whether all external files required by the real run exist."""

    table_path = lrs_root / "spectra/76/89/90/56/WASP_77_A_b_3.11569_5329_1.tbl"
    return all(
        path.is_file()
        for path in (hrs_cube, hrs_info, h2o_table, co_table, table_path)
    )


def _select_k_band_orders(order_wavelengths: np.ndarray) -> tuple[int, ...]:
    """Select real orders wholly inside the declared K-band window."""

    values = np.asarray(order_wavelengths, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("order_wavelengths must be a two-dimensional grid array")
    lower, upper = HRS_K_BAND_RANGE_MICRON
    selected = tuple(
        int(index)
        for index, order in enumerate(values)
        if float(np.min(order)) >= lower and float(np.max(order)) <= upper
    )
    if not selected:
        raise ValueError("no real IGRINS order is wholly inside the K-band window")
    return selected


def _pressure_grid() -> PressureGrid:
    return PressureGrid.from_log_centers(
        PRESSURE_FIRST_CENTER_BAR,
        PRESSURE_LAST_CENTER_BAR,
        ATMOSPHERE_LAYERS,
        unit="bar",
        name="WASP-77Ab fixed LBL convergence pressure grid",
    )


def _profile_temperatures(pressure_grid: PressureGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed non-inverted temperature profile at centers and edges."""

    edge_fraction = (
        np.log10(pressure_grid.edges) - np.log10(pressure_grid.edges[0])
    ) / np.log10(pressure_grid.edges[-1] / pressure_grid.edges[0])
    center_fraction = (
        np.log10(pressure_grid.centers) - np.log10(pressure_grid.edges[0])
    ) / np.log10(pressure_grid.edges[-1] / pressure_grid.edges[0])
    edge_temperature = TEMPERATURE_TOP_K + (
        TEMPERATURE_BOTTOM_K - TEMPERATURE_TOP_K
    ) * edge_fraction
    center_temperature = TEMPERATURE_TOP_K + (
        TEMPERATURE_BOTTOM_K - TEMPERATURE_TOP_K
    ) * center_fraction
    return center_temperature, edge_temperature


def _vmr_state() -> dict[str, float]:
    """Build the complete fixed VMR state used by every spectrum."""

    trace = H2O_VMR + CO_VMR + HMINUS_VMR + HYDROGEN_VMR + ELECTRON_VMR
    if trace >= 1.0:
        raise ValueError("fixed trace VMR state leaves no H2/He background")
    background = 1.0 - trace
    return {
        "H2O": H2O_VMR,
        "CO": CO_VMR,
        "H-": HMINUS_VMR,
        "H": HYDROGEN_VMR,
        "e-": ELECTRON_VMR,
        "H2": background * BACKGROUND_H2_FRACTION,
        "He": background * (1.0 - BACKGROUND_H2_FRACTION),
    }


def _atmosphere() -> AtmosphereState:
    pressure_grid = _pressure_grid()
    temperature, temperature_edges = _profile_temperatures(pressure_grid)
    vmr = _vmr_state()
    mean_molecular_weight = sum(
        vmr[name] * MOLECULAR_MASSES_AMU[name] for name in vmr
    )
    return AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature,
        temperature_edges=temperature_edges,
        composition={
            name: np.full(pressure_grid.n_layers, value, dtype=float)
            for name, value in vmr.items()
        },
        mean_molecular_weight=mean_molecular_weight,
        composition_convention="volume_mixing_ratio",
        metadata={
            "composition_source": "fixed_explicit_vmr_state",
            "mass_fraction_parameters": "none",
            "temperature_profile": "linear_in_log10_pressure_non_inverted",
        },
    )


def _provider(
    *,
    h2o_table: Path = H2O_TABLE,
    co_table: Path = CO_TABLE,
    wavelength_bounds: tuple[float, float],
) -> LineByLineOpacityProvider:
    """Build a bounded two-species pRT-table LBL provider."""

    return LineByLineOpacityProvider.from_hdf_paths(
        {"H2O": h2o_table, "CO": co_table},
        checksum=False,
        max_memory_bytes=OPACITY_MEMORY_LIMIT_BYTES,
        wavelength_bounds_micron=wavelength_bounds,
        native_sampling_stride=1,
        max_cached_slices=0,
    )


def _end_to_end_memory_estimate(
    preparation_bytes: int,
    *,
    n_layers: int,
    n_wavelength: int,
) -> int:
    """Conservatively account for preparation, RT, and continuum arrays."""

    rt_arrays = 24 * n_layers * n_wavelength * np.dtype(float).itemsize
    return int(preparation_bytes + rt_arrays + 256 * 1024**2)


def _planck_radiance(wavelength_micron: np.ndarray, temperature_K: float) -> np.ndarray:
    """Return blackbody spectral radiance in W m^-3 sr^-1."""

    wavelength_m = np.asarray(wavelength_micron, dtype=float) * 1.0e-6
    exponent = PLANCK_H_J_S * PLANCK_C_M_S / (
        wavelength_m * PLANCK_K_J_K * float(temperature_K)
    )
    return 2.0 * PLANCK_H_J_S * PLANCK_C_M_S**2 / (
        wavelength_m**5 * np.expm1(exponent)
    )


def _radiance_to_eclipse_depth(
    spectral_grid: SpectralGrid,
    radiance: np.ndarray,
) -> np.ndarray:
    """Convert ROBERT planet radiance to a physical eclipse-depth spectrum."""

    planet = np.asarray(radiance, dtype=float)
    wavelength = np.asarray(spectral_grid.values, dtype=float)
    stellar = _planck_radiance(wavelength, float(target.STAR_EFFECTIVE_TEMPERATURE_K))
    radius_ratio_squared = (target.PLANET.radius_m / target.STAR.radius_m) ** 2
    eclipse_depth = radius_ratio_squared * planet / stellar
    if not np.all(np.isfinite(eclipse_depth)) or np.any(eclipse_depth <= 0.0):
        raise ValueError("planet/star radiance conversion produced invalid eclipse depth")
    return eclipse_depth


def _model_native_spectrum(
    provider: LineByLineOpacityProvider,
    atmosphere: AtmosphereState,
    native_grid: SpectralGrid,
    *,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> tuple[np.ndarray, dict[str, object]]:
    """Evaluate one bounded LBL spectrum and release its prepared arrays."""

    preparation_estimate = provider.estimate_memory_bytes(
        native_grid,
        species=("H2O", "CO"),
    )
    if preparation_estimate >= OPACITY_MEMORY_LIMIT_BYTES:
        raise MemoryError("LBL opacity estimate reaches the strict 1023 MiB limit")
    end_to_end_estimate = _end_to_end_memory_estimate(
        preparation_estimate,
        n_layers=atmosphere.n_layers,
        n_wavelength=native_grid.size,
    )
    if end_to_end_estimate >= PROCESS_MEMORY_LIMIT_BYTES:
        raise MemoryError("LBL end-to-end estimate reaches the strict 1.9 GiB limit")

    started = perf_counter()
    prepared = provider.prepare(
        native_grid,
        atmosphere.pressure_grid,
        species=("H2O", "CO"),
    )
    gas = evaluate_gas_optical_depth(
        provider,
        prepared,
        atmosphere,
        gravity_m_s2=gravity_m_s2,
        gas_combination="random_overlap",
        retain_species_tau=False,
    )
    continuum = hminus_optical_depth(gas, HMINUS_CONFIG)
    radiance = solve_emission_spectrum(
        gas,
        geometry=gauss_legendre_disk_geometry(n_mu=4),
        bottom_boundary="blackbody",
        additional_optical_depths=(continuum,),
        thermal_integration_backend="numpy",
    )
    radiance_values = np.array(radiance.values, dtype=float, copy=True)
    if radiance_values.shape != (native_grid.size,) or not np.all(
        np.isfinite(radiance_values)
    ):
        raise ValueError("ROBERT LBL model produced invalid native radiance values")
    values = _radiance_to_eclipse_depth(native_grid, radiance_values)
    diagnostics = {
        "n_wavelength": native_grid.size,
        "preparation_estimate_bytes": int(preparation_estimate),
        "end_to_end_estimate_bytes": int(end_to_end_estimate),
        "duration_seconds": perf_counter() - started,
        "native_sampling_stride": int(
            prepared.metadata.get("native_sampling_stride", "1")
        ),
        "assembly_backend": str(gas.metadata.get("assembly_backend", "")),
        "hminus_continuum": True,
        "output_observable": "eclipse_depth",
    }
    del continuum, radiance, radiance_values, gas, prepared
    provider.clear_prepared_cache()
    gc.collect()
    return values, diagnostics


def _hrs_pixel_observation(wavelength: np.ndarray, order_index: int) -> Observation:
    edges = infer_wavelength_bin_edges(wavelength)
    return Observation.from_arrays(
        wavelength,
        np.zeros(wavelength.size, dtype=float),
        np.ones(wavelength.size, dtype=float),
        wavelength_unit="micron",
        flux_unit="eclipse_depth",
        observable="eclipse_depth",
        instrument=f"Smith-IGRINS-order-{order_index}",
        wavelength_bin_edges=edges,
    )


def _hrs_order_flux(
    provider: LineByLineOpacityProvider,
    atmosphere: AtmosphereState,
    wavelength: np.ndarray,
    *,
    order_index: int,
    stride: int,
) -> tuple[np.ndarray, dict[str, object]]:
    observation = _hrs_pixel_observation(wavelength, order_index)
    order = HighResolutionOrder(
        name=f"smith_igrins_order_{order_index}",
        observation=observation,
        order_number=order_index,
        lsf_resolving_power=HRS_LSF_RESOLVING_POWER,
        lsf_kernel_support_sigma=4.0,
        projected_rotation_km_s=HRS_ROTATION_KM_S,
        limb_darkening=HRS_LIMB_DARKENING,
        apply_velocity_shift=False,
        pixel_integration=True,
        infer_pixel_edges=False,
    )
    lower = float(np.min(wavelength)) - HRS_SOURCE_MARGIN_MICRON
    upper = float(np.max(wavelength)) + HRS_SOURCE_MARGIN_MICRON
    native_grid = provider.native_spectral_grid(
        sampling=stride,
        wavelength_bounds_micron=(lower, upper),
        reference_species="H2O",
        name=f"Smith-IGRINS-order-{order_index}-stride-{stride}",
    )
    native_values, diagnostics = _model_native_spectrum(
        provider,
        atmosphere,
        native_grid,
    )
    native_spectrum = Spectrum(
        spectral_grid=native_grid,
        values=native_values,
        unit="eclipse_depth",
        observable="eclipse_depth",
    )
    prepared_order = order.prepare(native_grid)
    observed = prepared_order.observe(native_spectrum)
    output = np.array(observed.values, dtype=float, copy=True)
    if output.shape != wavelength.shape or not np.all(np.isfinite(output)):
        raise ValueError("HRS response produced invalid pixel values")
    diagnostics.update(
        {
            "native_wavelength_count": int(native_grid.size),
            "response_stages": list(HRS_RESPONSE_STAGES),
            "response_grid_orientation": (
                "descending" if native_grid.values[0] > native_grid.values[-1] else "ascending"
            ),
        }
    )
    del observed, native_spectrum, native_values, prepared_order, native_grid
    gc.collect()
    return output, diagnostics


def _line_shape_error(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference_values = np.asarray(reference, dtype=float)
    candidate_values = np.asarray(candidate, dtype=float)
    if reference_values.shape != candidate_values.shape:
        raise ValueError("reference and candidate HRS arrays must match")
    reference_scale = float(np.median(reference_values))
    candidate_scale = float(np.median(candidate_values))
    if reference_scale <= 0.0 or candidate_scale <= 0.0:
        raise ValueError("HRS line-shape normalisation requires positive flux")
    reference_normalised = reference_values / reference_scale
    candidate_normalised = candidate_values / candidate_scale
    return candidate_normalised / reference_normalised - 1.0


def _line_shape_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    error = _line_shape_error(reference, candidate)
    return {
        "rms_fractional": float(np.sqrt(np.mean(np.square(error)))),
        "max_absolute_fractional": float(np.max(np.abs(error))),
        "median_signed_fractional": float(np.median(error)),
    }


def _aggregate_hrs_metrics(
    errors: Mapping[int, np.ndarray],
) -> dict[str, float | int]:
    if not errors:
        raise ValueError("at least one HRS order is required")
    flattened = np.concatenate(tuple(np.asarray(value, dtype=float) for value in errors.values()))
    return {
        "n_orders": len(errors),
        "n_points": int(flattened.size),
        "rms_fractional": float(np.sqrt(np.mean(np.square(flattened)))),
        "max_absolute_fractional": float(np.max(np.abs(flattened))),
        "p95_absolute_fractional": float(np.percentile(np.abs(flattened), 95.0)),
    }


def _lrs_model_observation(observation: Observation) -> Observation:
    return Observation.from_arrays(
        observation.wavelength,
        np.zeros(observation.n_points, dtype=float),
        np.ones(observation.n_points, dtype=float),
        wavelength_unit=observation.wavelength_unit,
        flux_unit="eclipse_depth",
        observable="eclipse_depth",
        instrument=f"{observation.instrument}-model",
        wavelength_bin_edges=observation.wavelength_bin_edges,
    )


def _lrs_bin_flux(
    source_grid: SpectralGrid,
    source_values: np.ndarray,
    observations: Sequence[Observation],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    source_spectrum = Spectrum(
        spectral_grid=source_grid,
        values=source_values,
        unit="eclipse_depth",
        observable="eclipse_depth",
    )
    values_by_detector: dict[str, np.ndarray] = {}
    uncertainties: list[np.ndarray] = []
    for observation in observations:
        model_observation = _lrs_model_observation(observation)
        binned = TopHatObservationResponse().prepare(model_observation).observe(
            source_spectrum
        )
        values_by_detector[str(observation.instrument)] = np.array(
            binned.values,
            dtype=float,
            copy=True,
        )
        uncertainties.append(np.asarray(observation.uncertainty, dtype=float))
    return values_by_detector, np.concatenate(uncertainties)


def _lrs_sigma_metrics(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    observations: Sequence[Observation],
) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
    errors: list[np.ndarray] = []
    per_detector: dict[str, dict[str, float | int]] = {}
    for observation in observations:
        name = str(observation.instrument)
        difference = np.asarray(candidate[name]) - np.asarray(reference[name])
        sigma = np.asarray(observation.uncertainty, dtype=float)
        scaled = difference / sigma
        errors.append(scaled)
        per_detector[name] = {
            "n_points": int(scaled.size),
            "rms_sigma": float(np.sqrt(np.mean(np.square(scaled)))),
            "max_absolute_sigma": float(np.max(np.abs(scaled))),
            "p95_absolute_sigma": float(np.percentile(np.abs(scaled), 95.0)),
        }
    flattened = np.concatenate(errors)
    return (
        {
            "n_detectors": len(per_detector),
            "n_points": int(flattened.size),
            "rms_sigma": float(np.sqrt(np.mean(np.square(flattened)))),
            "max_absolute_sigma": float(np.max(np.abs(flattened))),
            "p95_absolute_sigma": float(np.percentile(np.abs(flattened), 95.0)),
        },
        per_detector,
    )


def _table_records(h2o_table: Path, co_table: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for name, path in (
        ("H2O__POKAZATEL", h2o_table),
        ("CO__HITEMP", co_table),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        records[name] = {
            "path": str(path.relative_to(ROOT)),
            "bytes": int(path.stat().st_size),
            "sha256": TABLE_SHA256[name],
            "checksum_status": "pinned_external_manifest",
        }
    return records


def _source_records(
    hrs_cube: Path,
    hrs_info: Path,
    lrs_root: Path,
) -> dict[str, dict[str, object]]:
    table_path = lrs_root / "spectra/76/89/90/56/WASP_77_A_b_3.11569_5329_1.tbl"
    records = {
        "smith_cube14v3": {
            "path": str(hrs_cube.relative_to(ROOT)),
            "bytes": SMITH2024_WASP77AB_CUBE14_SIZE,
            "md5": SMITH2024_WASP77AB_CUBE14_MD5,
            "sha256": SMITH2024_WASP77AB_CUBE14_SHA256,
            "role": "Smith 2024 IGRINS order and pixel grids",
        },
        "smith_20201214_info": {
            "path": str(hrs_info.relative_to(ROOT)),
            "bytes": SMITH2024_WASP77AB_INFO_SIZE,
            "md5": SMITH2024_WASP77AB_INFO_MD5,
            "sha256": SMITH2024_WASP77AB_INFO_SHA256,
            "role": "Smith 2020-12-14 frame metadata",
        },
        "august_nirspec_table": {
            "path": str(table_path.relative_to(ROOT)),
            "bytes": int(table_path.stat().st_size),
            "sha256": "96159824870eb7c8132fd9c8bee30caf9ec95f05e128dbe2fea8627909a40d3f",
            "role": "August et al. 2023 NRS1/NRS2 bin edges and uncertainties",
            "data_producer": "August et al. 2023; Smith et al. 2024 comparison input",
        },
    }
    return records


def _build_report(
    *,
    command: str,
    source_records: Mapping[str, Mapping[str, object]],
    table_records: Mapping[str, Mapping[str, object]],
    atmosphere: AtmosphereState,
    hrs_record: Mapping[str, object],
    lrs_record: Mapping[str, object],
    acceptance: Mapping[str, bool],
    timings: Mapping[str, float],
    peak_rss_bytes: int,
) -> dict[str, object]:
    thread_values = _thread_values()
    vmr = _vmr_state()
    return {
        "schema_version": 1,
        "record_type": "wasp77ab_real_grid_lbl_sampling_convergence",
        "status": "pass" if all(acceptance.values()) else "fail",
        "command": command,
        "paper": {
            "smith2024_arxiv": "2312.13069",
            "smith2024_doi": "10.3847/1538-3881/ad17bf",
            "zenodo_doi": "10.5281/zenodo.10382053",
            "august2023_doi": "10.3847/2041-8213/ace828",
        },
        "target": {
            "name": "WASP-77Ab",
            "gravity_m_s2": GRAVITY_M_S2,
            "star_radius_r_sun": float(target.STAR_RADIUS_SOLAR),
            "planet_radius_r_j": float(target.PLANET_RADIUS_JUPITER),
            "smith_joint_anchor_log10_h2o_vmr": H2O_LOG10_VMR,
            "smith_joint_anchor_log10_co_vmr": CO_LOG10_VMR,
        },
        "source_inputs": {
            "data": dict(source_records),
            "opacity_tables": dict(table_records),
        },
        "atmosphere": {
            "composition_convention": "volume_mixing_ratio",
            "mass_fraction_parameters": False,
            "vmr": vmr,
            "mean_molecular_weight_amu": float(
                np.mean(np.asarray(atmosphere.mean_molecular_weight, dtype=float))
            ),
            "pressure_centers_bar": [
                PRESSURE_FIRST_CENTER_BAR,
                PRESSURE_LAST_CENTER_BAR,
            ],
            "n_layers": ATMOSPHERE_LAYERS,
            "temperature_profile": {
                "form": "linear_in_log10_pressure",
                "inverted": False,
                "top_K": TEMPERATURE_TOP_K,
                "bottom_K": TEMPERATURE_BOTTOM_K,
            },
            "hminus_continuum": {
                "config_species": list(HMINUS_CONFIG.species),
                "temperature_extrapolation": HMINUS_CONFIG.temperature_extrapolation,
                "spectral_extrapolation": HMINUS_CONFIG.spectral_extrapolation,
                "hminus_vmr": HMINUS_VMR,
                "hydrogen_vmr": HYDROGEN_VMR,
                "electron_vmr": ELECTRON_VMR,
                "included_in_all_stride_evaluations": True,
            },
        },
        "hrs": dict(hrs_record),
        "lrs": dict(lrs_record),
        "acceptance": dict(acceptance),
        "resources": {
            "run_cpu_count": RUN_CPU_COUNT,
            "max_threads": MAX_THREADS,
            "thread_values": thread_values,
            "process_rss_limit_bytes": PROCESS_MEMORY_LIMIT_BYTES,
            "process_rss_comparison": "strictly below",
            "opacity_estimate_limit_bytes": OPACITY_MEMORY_LIMIT_BYTES,
            "opacity_estimate_comparison": "strictly below",
            "measured_peak_rss_bytes": int(peak_rss_bytes),
        },
        "timings_seconds": dict(timings),
        "notes": [
            "The Smith cube flux values are not fitted; its 2020-12-14 order and pixel wavelength grids are used.",
            "The HRS reference and candidates are evaluated one order at a time with no spectra retained in the report.",
            "The LRS reference uses stride 10 as the R~100,000 carrier; broad stride 1 was not attempted.",
            "All ROBERT composition inputs are VMRs. No ROBERT mass-fraction parameter is present.",
        ],
    }


def _write_json(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_benchmark(
    *,
    hrs_cube: Path = DEFAULT_HRS_CUBE,
    hrs_info: Path = DEFAULT_HRS_INFO,
    lrs_root: Path = DEFAULT_LRS_ROOT,
    h2o_table: Path = H2O_TABLE,
    co_table: Path = CO_TABLE,
    report_path: Path = DEFAULT_REPORT,
    argv: Sequence[str] | None = None,
) -> dict[str, object]:
    """Run the real-grid HRS/LRS sampling convergence benchmark."""

    started = perf_counter()
    for path in (hrs_cube, hrs_info, h2o_table, co_table):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_started = perf_counter()
    hrs = load_smith2024_wasp77ab_hrs(
        hrs_cube,
        info_path=hrs_info,
        verify_checksum=True,
        verify_sha256=True,
    )
    order_wavelengths = np.array(hrs.order_wavelengths, dtype=float, copy=True)
    selected_orders = _select_k_band_orders(order_wavelengths)
    pixels_per_order = int(order_wavelengths.shape[1])
    del hrs
    gc.collect()
    collection = load_august2023_wasp77ab(lrs_root, verify_checksum=True)
    if collection.n_points != 150 or collection.names != (
        "nirspec_g395h_nrs1",
        "nirspec_g395h_nrs2",
    ):
        raise ValueError("unexpected August NRS1/NRS2 collection")
    lrs_observations = tuple(dataset.observation for dataset in collection.datasets)
    source_seconds = perf_counter() - source_started
    atmosphere = _atmosphere()

    table_records = _table_records(h2o_table, co_table)
    source_records = _source_records(hrs_cube, hrs_info, lrs_root)

    hrs_provider = _provider(
        h2o_table=h2o_table,
        co_table=co_table,
        wavelength_bounds=(
            HRS_K_BAND_RANGE_MICRON[0] - 0.02,
            HRS_K_BAND_RANGE_MICRON[1] + 0.02,
        ),
    )
    hrs_started = perf_counter()
    hrs_errors: dict[int, dict[int, np.ndarray]] = {
        stride: {} for stride in HRS_CANDIDATE_STRIDES
    }
    hrs_max_estimates: dict[str, int] = {}
    hrs_order_metrics: dict[str, dict[str, dict[str, float]]] = {
        str(stride): {} for stride in HRS_CANDIDATE_STRIDES
    }
    for order_index in selected_orders:
        wavelength = order_wavelengths[order_index]
        reference, reference_diagnostics = _hrs_order_flux(
            hrs_provider,
            atmosphere,
            wavelength,
            order_index=order_index,
            stride=HRS_REFERENCE_STRIDE,
        )
        hrs_max_estimates["1"] = max(
            hrs_max_estimates.get("1", 0),
            int(reference_diagnostics["preparation_estimate_bytes"]),
        )
        for stride in HRS_CANDIDATE_STRIDES:
            candidate, diagnostics = _hrs_order_flux(
                hrs_provider,
                atmosphere,
                wavelength,
                order_index=order_index,
                stride=stride,
            )
            hrs_max_estimates[str(stride)] = max(
                hrs_max_estimates.get(str(stride), 0),
                int(diagnostics["preparation_estimate_bytes"]),
            )
            error = _line_shape_error(reference, candidate)
            hrs_errors[stride][order_index] = error
            hrs_order_metrics[str(stride)][str(order_index)] = {
                "rms_fractional": float(np.sqrt(np.mean(np.square(error)))),
                "max_absolute_fractional": float(np.max(np.abs(error))),
            }
            del candidate, diagnostics
        del reference, reference_diagnostics
        gc.collect()
    hrs_provider.clear_prepared_cache()
    hrs_metrics = {
        str(stride): {
            **_aggregate_hrs_metrics(errors),
            "gate_rms_limit": HRS_LINE_SHAPE_RMS_LIMIT,
            "gate_max_limit": HRS_LINE_SHAPE_MAX_LIMIT,
            "gate_pass": bool(
                _aggregate_hrs_metrics(errors)["rms_fractional"]
                <= HRS_LINE_SHAPE_RMS_LIMIT
                and _aggregate_hrs_metrics(errors)["max_absolute_fractional"]
                <= HRS_LINE_SHAPE_MAX_LIMIT
            ),
            "per_order": hrs_order_metrics[str(stride)],
        }
        for stride, errors in hrs_errors.items()
    }
    hrs_seconds = perf_counter() - hrs_started
    hrs_passing = [
        int(stride)
        for stride, metrics in hrs_metrics.items()
        if bool(metrics["gate_pass"])
    ]
    hrs_selected = min(hrs_passing) if hrs_passing else None

    lrs_provider = _provider(
        h2o_table=h2o_table,
        co_table=co_table,
        wavelength_bounds=(
            LRS_BOUNDS_MICRON[0] - LRS_SOURCE_MARGIN_MICRON,
            LRS_BOUNDS_MICRON[1] + LRS_SOURCE_MARGIN_MICRON,
        ),
    )
    lrs_started = perf_counter()
    lrs_values: dict[int, dict[str, np.ndarray]] = {}
    lrs_estimates: dict[str, int] = {}
    for stride in LRS_LBL_STRIDES:
        native_grid = lrs_provider.native_spectral_grid(
            sampling=stride,
            wavelength_bounds_micron=(
                LRS_BOUNDS_MICRON[0] - LRS_SOURCE_MARGIN_MICRON,
                LRS_BOUNDS_MICRON[1] + LRS_SOURCE_MARGIN_MICRON,
            ),
            reference_species="H2O",
            name=f"Smith-NIRSpec-stride-{stride}",
        )
        native_values, diagnostics = _model_native_spectrum(
            lrs_provider,
            atmosphere,
            native_grid,
        )
        lrs_values[stride], _ = _lrs_bin_flux(
            native_grid,
            native_values,
            lrs_observations,
        )
        lrs_estimates[str(stride)] = int(diagnostics["preparation_estimate_bytes"])
        del native_grid, native_values, diagnostics
        gc.collect()
    lrs_provider.clear_prepared_cache()
    lrs_reference = lrs_values[LRS_REFERENCE_STRIDE]
    lrs_metrics: dict[str, dict[str, object]] = {}
    for stride in LRS_CANDIDATE_STRIDES:
        aggregate, per_detector = _lrs_sigma_metrics(
            lrs_reference,
            lrs_values[stride],
            lrs_observations,
        )
        aggregate.update(
            {
                "gate_rms_limit_sigma": LRS_SIGMA_RMS_LIMIT,
                "gate_max_limit_sigma": LRS_SIGMA_MAX_LIMIT,
                "gate_pass": bool(
                    aggregate["rms_sigma"] <= LRS_SIGMA_RMS_LIMIT
                    and aggregate["max_absolute_sigma"] <= LRS_SIGMA_MAX_LIMIT
                ),
                "per_detector": per_detector,
            }
        )
        lrs_metrics[str(stride)] = aggregate
    lrs_seconds = perf_counter() - lrs_started
    lrs_passing = [
        int(stride)
        for stride, metrics in lrs_metrics.items()
        if bool(metrics["gate_pass"])
    ]
    lrs_selected = min(lrs_passing) if lrs_passing else None

    del lrs_values, lrs_reference, order_wavelengths, collection
    gc.collect()
    peak_rss = _peak_rss_bytes()
    thread_values = _thread_values()
    acceptance = {
        "hrs_reference_and_candidates_complete": bool(
            len(hrs_metrics) == len(HRS_CANDIDATE_STRIDES)
        ),
        "hrs_finest_passing_stride_selected": hrs_selected is not None,
        "lrs_reference_and_candidates_complete": bool(
            len(lrs_metrics) == len(LRS_CANDIDATE_STRIDES)
        ),
        "lrs_finest_passing_stride_selected": lrs_selected is not None,
        "opacity_estimates_below_1023_MiB": bool(
            max((*hrs_max_estimates.values(), *lrs_estimates.values()))
            < OPACITY_MEMORY_LIMIT_BYTES
        ),
        "process_rss_below_1_9_GiB": bool(
            0 < peak_rss < PROCESS_MEMORY_LIMIT_BYTES
        ),
        "thread_values_between_1_and_3": all(
            1 <= value <= MAX_THREADS for value in thread_values.values()
        ),
        "hrs_response_contract": HRS_RESPONSE_STAGES
        == (
            "rotational-broadening",
            "gaussian-high-resolution",
            "pixel-bin-integration",
        ),
    }
    report = _build_report(
        command=_reproducible_command(argv),
        source_records=source_records,
        table_records=table_records,
        atmosphere=atmosphere,
        hrs_record={
            "scope": "real Smith 2020-12-14 IGRINS K-band orders",
            "order_selection_range_micron": list(HRS_K_BAND_RANGE_MICRON),
            "order_indices": list(selected_orders),
            "n_orders": len(selected_orders),
            "pixels_per_order": pixels_per_order,
            "reference_stride": HRS_REFERENCE_STRIDE,
            "candidate_strides": list(HRS_CANDIDATE_STRIDES),
            "native_evaluation": "one buffered order at a time",
            "response_stages": list(HRS_RESPONSE_STAGES),
            "rotation_km_s": HRS_ROTATION_KM_S,
            "gaussian_lsf_resolving_power": HRS_LSF_RESOLVING_POWER,
            "pixel_grid": "real Smith order pixels with midpoint-inferred edges",
            "doppler_stage": "not applied in sampling convergence; rest-grid comparison",
            "max_preparation_estimate_bytes": hrs_max_estimates,
            "metrics": hrs_metrics,
            "selected_finest_passing_stride": hrs_selected,
        },
        lrs_record={
            "scope": "August NRS1/NRS2 bins",
            "bin_edges_micron": list(LRS_BOUNDS_MICRON),
            "detectors": [str(observation.instrument) for observation in lrs_observations],
            "n_points": int(sum(observation.n_points for observation in lrs_observations)),
            "reference_stride": LRS_REFERENCE_STRIDE,
            "candidate_strides": list(LRS_CANDIDATE_STRIDES),
            "reference_resolution_note": "stride 10 approximates R~100,000",
            "broad_stride_1_attempted": False,
            "response": "exact piecewise-linear top-hat integration over published bin edges",
            "published_uncertainty_definition": "loader symmetric mean of absolute asymmetric errors",
            "max_preparation_estimate_bytes": lrs_estimates,
            "metrics": lrs_metrics,
            "selected_finest_passing_stride": lrs_selected,
        },
        acceptance=acceptance,
        timings={
            "source_load_seconds": source_seconds,
            "hrs_seconds": hrs_seconds,
            "lrs_seconds": lrs_seconds,
            "total_seconds": perf_counter() - started,
        },
        peak_rss_bytes=peak_rss,
    )
    _write_json(report_path, report)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hrs-cube", type=Path, default=DEFAULT_HRS_CUBE)
    parser.add_argument("--hrs-info", type=Path, default=DEFAULT_HRS_INFO)
    parser.add_argument("--lrs-root", type=Path, default=DEFAULT_LRS_ROOT)
    parser.add_argument("--h2o-table", type=Path, default=H2O_TABLE)
    parser.add_argument("--co-table", type=Path, default=CO_TABLE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parse_args(argv)
    return run_benchmark(
        hrs_cube=args.hrs_cube.expanduser().resolve(),
        hrs_info=args.hrs_info.expanduser().resolve(),
        lrs_root=args.lrs_root.expanduser().resolve(),
        h2o_table=args.h2o_table.expanduser().resolve(),
        co_table=args.co_table.expanduser().resolve(),
        report_path=args.report.expanduser().resolve(),
        argv=argv,
    )


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))


__all__ = [
    "ATMOSPHERE_LAYERS",
    "CO_LOG10_VMR",
    "CO_TABLE",
    "CO_VMR",
    "DEFAULT_HRS_CUBE",
    "DEFAULT_HRS_INFO",
    "DEFAULT_LRS_ROOT",
    "DEFAULT_REPORT",
    "ELECTRON_VMR",
    "H2O_LOG10_VMR",
    "H2O_TABLE",
    "H2O_VMR",
    "HMINUS_CONFIG",
    "HMINUS_VMR",
    "HRS_CANDIDATE_STRIDES",
    "HRS_K_BAND_RANGE_MICRON",
    "HRS_LBL_STRIDES",
    "HRS_LINE_SHAPE_MAX_LIMIT",
    "HRS_LINE_SHAPE_RMS_LIMIT",
    "HRS_LSF_RESOLVING_POWER",
    "HRS_REFERENCE_STRIDE",
    "HRS_RESPONSE_STAGES",
    "HRS_ROTATION_KM_S",
    "HYDROGEN_VMR",
    "LRS_BOUNDS_MICRON",
    "LRS_CANDIDATE_STRIDES",
    "LRS_LBL_STRIDES",
    "LRS_REFERENCE_STRIDE",
    "LRS_SIGMA_MAX_LIMIT",
    "LRS_SIGMA_RMS_LIMIT",
    "OPACITY_MEMORY_LIMIT_BYTES",
    "PROCESS_MEMORY_LIMIT_BYTES",
    "RUN_CPU_COUNT",
    "THREAD_VARIABLES",
    "_aggregate_hrs_metrics",
    "_atmosphere",
    "_line_shape_metrics",
    "_lrs_sigma_metrics",
    "_profile_temperatures",
    "_select_k_band_orders",
    "_vmr_state",
    "external_inputs_available",
    "main",
    "run_benchmark",
]
