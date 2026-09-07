"""Validate full-table ROBERT LBL emission over a K-band science grid."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import tempfile
from time import perf_counter

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
    """Keep every numerical thread setting in the safe range 1 to 3."""

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


def _thread_values() -> dict[str, int]:
    """Return the numerical thread settings recorded in the report."""

    return {name: int(os.environ[name]) for name in THREAD_VARIABLES}


def _reproducible_command() -> str:
    """Return a runnable project command with the original arguments."""

    script = str(Path(__file__).resolve().relative_to(ROOT))
    return shlex.join(
        (
            "conda",
            "run",
            "-n",
            "robert-exoplanets",
            "python",
            script,
            *sys.argv[1:],
        )
    )

import numpy as np  # noqa: E402

from robert_exoplanets import (  # noqa: E402
    AtmosphereState,
    GaussianHighResolutionResponse,
    Observation,
    PressureGrid,
    SpectralGrid,
    Spectrum,
    gauss_legendre_disk_geometry,
    solve_emission_spectrum,
)
from robert_exoplanets.forward._atmospheric import (  # noqa: E402
    evaluate_gas_optical_depth,
)
from robert_exoplanets.opacity import LineByLineOpacityProvider  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
INPUT_DATA = ROOT / "external_data" / "petitRADTRANS" / "input_data"
OUTPUT_DIR = ROOT / "examples" / "outputs" / "lbl_kband"
REPORT_PATH = ROOT / "docs" / "data" / "lbl_kband_science_grid_20260830.json"
H2O_TABLE = (
    INPUT_DATA
    / "opacities/lines/line_by_line/H2O/1H2-16O/"
    "1H2-16O__POKAZATEL.R1e6_0.3-28mu.xsec.petitRADTRANS.h5"
)
CO_TABLE = (
    INPUT_DATA
    / "opacities/lines/line_by_line/CO/12C-16O/"
    "12C-16O__HITEMP.R1e6_0.3-28mu.xsec.petitRADTRANS.h5"
)
ORACLES = {
    "baseline": OUTPUT_DIR / "petitradtrans3_lbl_kband_grid_baseline.npz",
    "isothermal": OUTPUT_DIR / "petitradtrans3_lbl_kband_grid_isothermal.npz",
    "inversion": OUTPUT_DIR / "petitradtrans3_lbl_kband_grid_inversion.npz",
    "low_abundance": OUTPUT_DIR / "petitradtrans3_lbl_kband_grid_low_abundance.npz",
    "high_gravity": OUTPUT_DIR / "petitradtrans3_lbl_kband_grid_high_gravity.npz",
    "h2o_only": OUTPUT_DIR / "petitradtrans3_lbl_kband_grid_h2o_only.npz",
    "co_only": OUTPUT_DIR / "petitradtrans3_lbl_kband_grid_co_only.npz",
    "background_only": OUTPUT_DIR / "petitradtrans3_lbl_kband_grid_background_only.npz",
}
MEDIUM_80 = OUTPUT_DIR / "petitradtrans3_lbl_kband_medium_reference_80.npz"
MEDIUM_160 = OUTPUT_DIR / "petitradtrans3_lbl_kband_medium_reference_160.npz"
WIDE = OUTPUT_DIR / "petitradtrans3_lbl_kband_wide_reference.npz"
LSF_RESOLVING_POWER = 100_000.0
LSF_SUPPORT_SIGMA = 4.0
GRAVITY_DEFAULT_M_S2 = 15.0
# Keep the process ceiling below the repository-wide two-GiB hard limit.  Use
# the same strict policy for the opacity preparation guard.
MAX_MEMORY_BYTES = int(1.9 * 1024**3)
OPACITY_MEMORY_BYTES = 1023 * 1024**2
EXPECTED_TABLE_SHA256 = {
    "H2O": "9a79513fb92dfa369f85abb273257ae2b2100a4a386169b9bfff8a5b74249b73",
    "CO": "5f77209da3ea67d5a697b06379b1de2b8fe106adf7ea0abfb451c675084087ad",
}


def _load(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing pRT oracle: {path}")
    with np.load(path, allow_pickle=False) as archive:
        reference = {
            "metadata": json.loads(str(archive["metadata_json"])),
            "pressure_bar": np.asarray(archive["pressure_bar"], dtype=float),
            "temperature_K": np.asarray(archive["temperature_K"], dtype=float),
            "wavelength_micron": np.asarray(
                archive["wavelength_micron"], dtype=float
            ),
            "flux_w_m2_m": np.asarray(archive["flux_w_m2_m"], dtype=float),
        }
    metadata = dict(reference["metadata"])
    pressure = np.asarray(reference["pressure_bar"], dtype=float)
    temperature = np.asarray(reference["temperature_K"], dtype=float)
    wavelength = np.asarray(reference["wavelength_micron"], dtype=float)
    flux = np.asarray(reference["flux_w_m2_m"], dtype=float)
    if (
        pressure.ndim != 1
        or temperature.shape != pressure.shape
        or wavelength.ndim != 1
        or flux.shape != wavelength.shape
        or pressure.size < 2
        or wavelength.size < 2
        or not np.all(np.isfinite(pressure))
        or not np.all(np.isfinite(temperature))
        or not np.all(np.isfinite(wavelength))
        or not np.all(np.isfinite(flux))
        or np.any(pressure <= 0.0)
        or np.any(temperature <= 0.0)
        or np.any(wavelength <= 0.0)
        or np.any(flux <= 0.0)
        or not np.all(np.diff(pressure) > 0.0)
        or not np.all(np.diff(wavelength) > 0.0)
    ):
        raise ValueError(f"invalid pRT oracle arrays: {path}")
    if (
        metadata.get("petitradtrans_version") != "3.3.3"
        or metadata.get("line_opacity_mode") != "lbl"
        or int(metadata.get("line_by_line_opacity_sampling", 0)) != 1
        or int(metadata.get("native_resolving_power", 0)) != 1_000_000
        or metadata.get("clear_atmosphere") is not True
        or list(metadata.get("included_continua", []))
    ):
        raise ValueError(f"unexpected pRT oracle contract: {path}")
    return reference


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


def _pressure_grid(reference: dict[str, object]) -> PressureGrid:
    pressure = np.asarray(reference["pressure_bar"], dtype=float)
    return PressureGrid.from_log_centers(
        float(pressure[0]),
        float(pressure[-1]),
        pressure.size,
        unit="bar",
        name="pRT pressure nodes as ROBERT centers",
    )


def _edge_temperatures(
    reference: dict[str, object],
    pressure_grid: PressureGrid,
) -> np.ndarray:
    metadata = dict(reference["metadata"])
    profile = dict(metadata["temperature_profile"])
    top = float(profile["top_K"])
    bottom = float(profile["bottom_K"])
    centers = np.asarray(reference["pressure_bar"], dtype=float)
    fraction = (
        np.log10(pressure_grid.edges) - np.log10(centers[0])
    ) / np.log10(centers[-1] / centers[0])
    return top + (bottom - top) * fraction


def _atmosphere(reference: dict[str, object]) -> AtmosphereState:
    metadata = dict(reference["metadata"])
    volume = dict(metadata["volume_mixing_ratios"])
    pressure_grid = _pressure_grid(reference)
    return AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.asarray(reference["temperature_K"], dtype=float),
        temperature_edges=_edge_temperatures(reference, pressure_grid),
        composition={
            "H2O": np.full(
                pressure_grid.n_layers,
                float(volume["H2O__POKAZATEL"]),
            ),
            "CO": np.full(
                pressure_grid.n_layers,
                float(volume["CO__HITEMP"]),
            ),
        },
        mean_molecular_weight=float(metadata["mean_molar_mass_amu"]),
    )


def _provider(
    *,
    stride: int = 1,
    bounds: tuple[float, float] | None = None,
) -> LineByLineOpacityProvider:
    return LineByLineOpacityProvider.from_hdf_paths(
        {"H2O": H2O_TABLE, "CO": CO_TABLE},
        checksum=False,
        max_memory_bytes=OPACITY_MEMORY_BYTES,
        wavelength_bounds_micron=bounds,
        native_sampling_stride=stride,
        max_cached_slices=1,
    )


def _robert_flux(
    reference: dict[str, object],
    provider: LineByLineOpacityProvider,
    *,
    spectral_grid: SpectralGrid | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    atmosphere = _atmosphere(reference)
    if spectral_grid is None:
        spectral_grid = SpectralGrid.from_array(
            np.asarray(reference["wavelength_micron"], dtype=float),
            unit="micron",
            role="opacity",
            name="pRT R=1e6 physical grid",
        )
    estimated_preparation = provider.estimate_memory_bytes(
        spectral_grid,
        species=("H2O", "CO"),
    )
    conservative_rt = int(
        24
        * atmosphere.n_layers
        * spectral_grid.size
        * np.dtype(float).itemsize
        + 256 * 1024**2
    )
    if estimated_preparation >= OPACITY_MEMORY_BYTES:
        raise MemoryError("LBL opacity preparation estimate reaches the 1 GiB guard")
    if estimated_preparation + conservative_rt >= MAX_MEMORY_BYTES:
        raise MemoryError(
            "LBL end-to-end estimate reaches the configured process-memory guard"
        )
    start = perf_counter()
    prepared = provider.prepare(
        spectral_grid,
        atmosphere.pressure_grid,
        species=("H2O", "CO"),
    )
    gas = evaluate_gas_optical_depth(
        provider,
        prepared,
        atmosphere,
        gravity_m_s2=float(dict(reference["metadata"])["gravity_m_s2"]),
        gas_combination="random_overlap",
        retain_species_tau=False,
    )
    radiance = solve_emission_spectrum(
        gas,
        geometry=gauss_legendre_disk_geometry(n_mu=8),
        bottom_boundary="blackbody",
        thermal_integration_backend="numpy",
    )
    return np.pi * np.asarray(radiance.values, dtype=float), {
        "n_layers": atmosphere.n_layers,
        "n_wavelength": spectral_grid.size,
        "preparation_estimate_bytes": estimated_preparation,
        "end_to_end_estimate_bytes": estimated_preparation + conservative_rt,
        "duration_seconds": perf_counter() - start,
        "assembly_backend": gas.metadata["assembly_backend"],
        "species_tau_diagnostics": gas.metadata["species_tau_diagnostics"],
        "native_sampling_stride": prepared.metadata["native_sampling_stride"],
    }


def _target_grid(wavelength: np.ndarray) -> np.ndarray:
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    lower_native = float(np.min(wavelength))
    upper_native = float(np.max(wavelength))
    margin = (
        LSF_SUPPORT_SIGMA
        * upper_native
        * fwhm_to_sigma
        / LSF_RESOLVING_POWER
    )
    lower = lower_native + margin
    upper = upper_native - margin
    count = int(np.floor(LSF_RESOLVING_POWER * np.log(upper / lower))) + 1
    target = lower * np.exp(np.arange(count, dtype=float) / LSF_RESOLVING_POWER)
    return target[target <= upper]


def _convolve(
    wavelength: np.ndarray,
    flux: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    native_grid = SpectralGrid.from_array(
        wavelength,
        unit="micron",
        role="rt_native",
    )
    observation = Observation.from_arrays(
        target,
        np.zeros(target.size),
        np.ones(target.size),
        flux_unit="W m^-3",
        observable="spectral_flux",
        instrument="Gaussian-R100000",
    )
    prepared = GaussianHighResolutionResponse(
        resolving_power=LSF_RESOLVING_POWER,
        kernel_support=LSF_SUPPORT_SIGMA,
    ).prepare(observation, native_grid)
    return np.asarray(
        prepared.observe(
            Spectrum(
                spectral_grid=native_grid,
                values=flux,
                unit="W m^-3",
                observable="spectral_flux",
            )
        ).values,
        dtype=float,
    )


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    relative = candidate / reference - 1.0
    return {
        "median_signed_relative_error": float(np.median(relative)),
        "rms_relative_error": float(np.sqrt(np.mean(np.square(relative)))),
        "p95_absolute_relative_error": float(
            np.percentile(np.abs(relative), 95.0)
        ),
        "max_absolute_relative_error": float(np.max(np.abs(relative))),
        "integrated_flux_relative_error": float(
            np.sum(candidate) / np.sum(reference) - 1.0
        ),
    }


def _compare_case(
    reference: dict[str, object],
    provider: LineByLineOpacityProvider,
) -> tuple[dict[str, float], dict[str, object], np.ndarray, np.ndarray]:
    wavelength = np.asarray(reference["wavelength_micron"], dtype=float)
    p_rt = np.asarray(reference["flux_w_m2_m"], dtype=float)
    robert, resources = _robert_flux(reference, provider)
    target = _target_grid(wavelength)
    p_rt_lsf = _convolve(wavelength, p_rt, target)
    robert_lsf = _convolve(wavelength, robert, target)
    return _metrics(p_rt_lsf, robert_lsf), resources, target, robert_lsf


def main() -> dict[str, object]:
    for path in (H2O_TABLE, CO_TABLE, WIDE, MEDIUM_80, MEDIUM_160, *ORACLES.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    table_inputs: dict[str, dict[str, object]] = {}
    for species, path in {"H2O": H2O_TABLE, "CO": CO_TABLE}.items():
        checksum = _file_sha256(path)
        if checksum != EXPECTED_TABLE_SHA256[species]:
            raise ValueError(f"unexpected {species} line-by-line table checksum")
        table_inputs[species] = {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": checksum,
        }

    oracle_paths = {
        **ORACLES,
        "medium_80": MEDIUM_80,
        "medium_160": MEDIUM_160,
        "wide": WIDE,
    }
    oracle_inputs = {
        name: {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for name, path in oracle_paths.items()
    }
    cases = {name: _load(path) for name, path in ORACLES.items()}
    narrow_provider = _provider(bounds=(2.2987, 2.3039))
    case_metrics: dict[str, dict[str, float]] = {}
    case_resources: dict[str, dict[str, object]] = {}
    convolved_cases: dict[str, np.ndarray] = {}
    target_narrow: np.ndarray | None = None
    for name, reference in cases.items():
        metrics, resources, target, convolved = _compare_case(
            reference,
            narrow_provider,
        )
        case_metrics[name] = metrics
        case_resources[name] = resources
        convolved_cases[name] = convolved
        target_narrow = target

    medium_80 = _load(MEDIUM_80)
    medium_160 = _load(MEDIUM_160)
    medium_wavelength = np.asarray(medium_160["wavelength_micron"], dtype=float)
    if not np.array_equal(
        medium_wavelength,
        np.asarray(medium_80["wavelength_micron"], dtype=float),
    ):
        raise ValueError("medium pressure oracles use different wavelength grids")
    medium_target = _target_grid(medium_wavelength)
    pressure_metrics = _metrics(
        _convolve(
            medium_wavelength,
            np.asarray(medium_160["flux_w_m2_m"], dtype=float),
            medium_target,
        ),
        _convolve(
            medium_wavelength,
            np.asarray(medium_80["flux_w_m2_m"], dtype=float),
            medium_target,
        ),
    )

    wide = _load(WIDE)
    wide_provider = _provider(bounds=(2.2899, 2.3501))
    wide_metrics, wide_resources, _, _ = _compare_case(wide, wide_provider)
    wide_provider.clear_prepared_cache()

    baseline = cases["baseline"]
    full_provider = _provider(bounds=(2.2987, 2.3039))
    full_wavelength = np.asarray(baseline["wavelength_micron"], dtype=float)
    full_flux, full_resources = _robert_flux(baseline, full_provider)
    full_provider.clear_prepared_cache()
    cache_atmosphere = _atmosphere(baseline)
    cache_grid = SpectralGrid.from_array(
        full_wavelength,
        unit="micron",
        role="opacity",
        name="cache-validation-grid",
    )
    cache_started = perf_counter()
    cold_prepared = full_provider.prepare(
        cache_grid,
        cache_atmosphere.pressure_grid,
        species=("H2O", "CO"),
    )
    cold_seconds = perf_counter() - cache_started
    cache_started = perf_counter()
    warm_prepared = full_provider.prepare(
        cache_grid,
        cache_atmosphere.pressure_grid,
        species=("H2O", "CO"),
    )
    warm_seconds = perf_counter() - cache_started
    cache_record = {
        "same_prepared_object": cold_prepared is warm_prepared,
        "cold_prepare_seconds": cold_seconds,
        "warm_prepare_seconds": warm_seconds,
        "cached_slice_count": full_provider.cached_slice_count,
        "cached_slice_bytes": full_provider.cached_slice_bytes,
    }
    stride_metrics: dict[str, dict[str, float]] = {}
    stride_resources: dict[str, dict[str, object]] = {"1": full_resources}
    for stride in (2, 4, 8):
        provider = _provider(stride=stride, bounds=(2.2987, 2.3039))
        grid = provider.native_spectral_grid(
            sampling=stride,
            wavelength_bounds_micron=(
                float(full_wavelength[0]),
                float(full_wavelength[-1]),
            ),
            reference_species="H2O",
        )
        flux, resources = _robert_flux(baseline, provider, spectral_grid=grid)
        sampled_wavelength = np.asarray(grid.values, dtype=float)
        common_target = _target_grid(sampled_wavelength)
        reference_lsf = _convolve(
            full_wavelength,
            full_flux,
            common_target,
        )
        sampled_lsf = _convolve(
            sampled_wavelength,
            flux,
            common_target,
        )
        stride_metrics[str(stride)] = _metrics(reference_lsf, sampled_lsf)
        stride_resources[str(stride)] = resources

    if target_narrow is None:
        raise RuntimeError("science grid is empty")
    line_imprints = {
        "h2o_only_rms_fraction_of_background": float(
            np.sqrt(
                np.mean(
                    np.square(
                        convolved_cases["h2o_only"]
                        / convolved_cases["background_only"]
                        - 1.0
                    )
                )
            )
        ),
        "co_only_rms_fraction_of_background": float(
            np.sqrt(
                np.mean(
                    np.square(
                        convolved_cases["co_only"]
                        / convolved_cases["background_only"]
                        - 1.0
                    )
                )
            )
        ),
        "combined_rms_fraction_of_background": float(
            np.sqrt(
                np.mean(
                    np.square(
                        convolved_cases["baseline"]
                        / convolved_cases["background_only"]
                        - 1.0
                    )
                )
            )
        ),
    }

    thread_values = _thread_values()
    peak_rss = _peak_rss_bytes()
    acceptance = {
        "all_narrow_cases_R100000": all(
            metrics["rms_relative_error"] <= 0.02
            and metrics["max_absolute_relative_error"] <= 0.06
            for metrics in case_metrics.values()
        ),
        "wide_band_R100000": (
            wide_metrics["rms_relative_error"] <= 0.02
            and wide_metrics["max_absolute_relative_error"] <= 0.06
        ),
        "pressure_80_vs_160_R100000": (
            pressure_metrics["rms_relative_error"] <= 0.0025
            and pressure_metrics["max_absolute_relative_error"] <= 0.01
        ),
        "stride_2_R100000": (
            stride_metrics["2"]["rms_relative_error"] <= 0.002
            and stride_metrics["2"]["max_absolute_relative_error"] <= 0.01
        ),
        "fused_low_memory_path": all(
            values["assembly_backend"] == "fused_lbl_direct_sum"
            and values["species_tau_diagnostics"] == "disabled"
            for values in (*case_resources.values(), wide_resources)
        ),
        "bounded_prepared_slice_cache": (
            bool(cache_record["same_prepared_object"])
            and int(cache_record["cached_slice_count"]) == 1
            and int(cache_record["cached_slice_bytes"]) < OPACITY_MEMORY_BYTES
        ),
        "memory_below_2_GiB": 0 < peak_rss < MAX_MEMORY_BYTES,
        "cpu_threads_at_most_3": all(
            1 <= value <= 3 for value in thread_values.values()
        ),
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "comparison": "ROBERT_full_table_LBL_science_grid_vs_petitRADTRANS3",
        "status": "pass" if all(acceptance.values()) else "fail",
        "command": _reproducible_command(),
        "thread_values": thread_values,
        "table_inputs": table_inputs,
        "oracle_contract": {
            "petitradtrans_version": "3.3.3",
            "line_opacity_mode": "lbl",
            "line_by_line_opacity_sampling": 1,
            "native_resolving_power": 1_000_000,
            "clear_atmosphere": True,
            "included_continua": [],
        },
        "oracle_inputs": oracle_inputs,
        "cases": case_metrics,
        "wide_band": {
            "wavelength_micron": [2.29, 2.35],
            "metrics": wide_metrics,
            "resources": wide_resources,
        },
        "pressure_convergence": {
            "layers": [80, 160],
            "wavelength_micron": [2.29, 2.32],
            "metrics": pressure_metrics,
        },
        "native_sampling_stride_convergence": {
            "reference_stride": 1,
            "metrics": stride_metrics,
            "resources": stride_resources,
            "accuracy_statement": "validated only after Gaussian R=100000 response",
        },
        "prepared_slice_cache": cache_record,
        "isolated_species_line_imprints": line_imprints,
        "acceptance": acceptance,
        "resources": {
            "cpu_thread_limit": max(thread_values.values()),
            "thread_values": thread_values,
            "memory_limit_bytes": MAX_MEMORY_BYTES,
            "opacity_memory_limit_bytes": OPACITY_MEMORY_BYTES,
            "measured_peak_rss_bytes": peak_rss,
            "case_resources": case_resources,
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit("full-table LBL science-grid validation failed")
    return report


if __name__ == "__main__":
    main()
