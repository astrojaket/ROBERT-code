"""Validate ROBERT line-by-line K-band emission against stable pRT 3."""

from __future__ import annotations

import json
from hashlib import sha256
import os
from pathlib import Path
import resource
import shlex
import sys
import tempfile
from time import perf_counter

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)

# Apply the laptop resource policy before numerical libraries load.
for _thread_variable in _THREAD_VARIABLES:
    os.environ[_thread_variable] = "3"
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from robert_exoplanets import (  # noqa: E402
    AtmosphereState,
    GaussianHighResolutionResponse,
    Observation,
    PressureGrid,
    SpectralGrid,
    Spectrum,
    assemble_gas_optical_depth,
    gauss_legendre_disk_geometry,
    solve_emission_spectrum,
)
from robert_exoplanets.opacity.line_by_line import (  # noqa: E402
    LineByLineOpacityProvider,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_DATA = ROOT / "external_data" / "petitRADTRANS" / "input_data"
OUTPUT_DIR = ROOT / "examples" / "outputs" / "lbl_kband"
REFERENCE_80 = OUTPUT_DIR / "petitradtrans3_lbl_kband_reference.npz"
REFERENCE_160 = OUTPUT_DIR / "petitradtrans3_lbl_kband_reference_160.npz"
REPORT = ROOT / "docs" / "data" / "lbl_kband_validation_20260829.json"
SPECTRA = OUTPUT_DIR / "lbl_kband_validation_spectra.npz"
FIGURE = OUTPUT_DIR / "lbl_kband_validation.png"
TABLE_PATHS = {
    "H2O": INPUT_DATA
    / "opacities/lines/line_by_line/H2O/1H2-16O/"
    "1H2-16O__POKAZATEL.R1e6_2.3-2.3mu.xsec.petitRADTRANS.h5",
    "CO": INPUT_DATA
    / "opacities/lines/line_by_line/CO/12C-16O/"
    "12C-16O__HITEMP.R1e6_0.3-28mu.xsec.petitRADTRANS.h5",
}
EXPECTED_TABLE_SHA256 = {
    "H2O": "9bdf7fbe170fb31a241a177d22b8498d0ac56080ebcb53ce0240786e904c5ba9",
    "CO": "5f77209da3ea67d5a697b06379b1de2b8fe106adf7ea0abfb451c675084087ad",
}
GRAVITY_M_S2 = 15.0
LSF_RESOLVING_POWER = 100_000.0
LSF_SUPPORT_SIGMA = 4.0
MAX_MEMORY_GIB = 1.9
MAX_MEMORY_BYTES = int(MAX_MEMORY_GIB * 1024**3)


def _load_reference(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing oracle {path}; run run_petitradtrans3_lbl_kband_reference.py"
        )
    with np.load(path, allow_pickle=False) as archive:
        return {
            "metadata": json.loads(str(archive["metadata_json"])),
            "pressure_bar": np.asarray(archive["pressure_bar"], dtype=float),
            "temperature_K": np.asarray(archive["temperature_K"], dtype=float),
            "wavelength_micron": np.asarray(
                archive["wavelength_micron"], dtype=float
            ),
            "flux_w_m2_m": np.asarray(archive["flux_w_m2_m"], dtype=float),
        }


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if os.uname().sysname == "Darwin":
        return value
    return value * 1024


def _thread_values() -> dict[str, int]:
    """Return all numerical thread settings used by this process."""

    values: dict[str, int] = {}
    for name in _THREAD_VARIABLES:
        try:
            values[name] = int(os.environ[name])
        except (KeyError, ValueError):
            values[name] = 0
    return values


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


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _temperature_at_pressure(pressure_bar: np.ndarray) -> np.ndarray:
    return 900.0 + 900.0 * (np.log10(pressure_bar) + 5.0) / 7.0


def _robert_flux(
    reference: dict[str, object],
    provider: LineByLineOpacityProvider,
) -> tuple[np.ndarray, dict[str, object]]:
    pressure_bar = np.asarray(reference["pressure_bar"], dtype=float)
    temperature_K = np.asarray(reference["temperature_K"], dtype=float)
    wavelength_micron = np.asarray(reference["wavelength_micron"], dtype=float)
    metadata = dict(reference["metadata"])
    volume_mixing_ratios = dict(metadata["volume_mixing_ratios"])
    mean_molar_mass_amu = float(metadata["mean_molar_mass_amu"])

    pressure_grid = PressureGrid.from_log_centers(
        float(pressure_bar[0]),
        float(pressure_bar[-1]),
        pressure_bar.size,
        unit="bar",
        name="petitradtrans-pressure-nodes-as-robert-centers",
    )
    spectral_grid = SpectralGrid.from_array(
        wavelength_micron,
        unit="micron",
        role="opacity",
        name="petitradtrans-R1e6-K-band",
    )
    temperature_edges_K = _temperature_at_pressure(pressure_grid.edges)
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature_K,
        temperature_edges=temperature_edges_K,
        composition={
            "H2O": np.full(
                pressure_grid.n_layers,
                float(volume_mixing_ratios["H2O__POKAZATEL"]),
            ),
            "CO": np.full(
                pressure_grid.n_layers,
                float(volume_mixing_ratios["CO__HITEMP"]),
            ),
        },
        mean_molecular_weight=np.full(
            pressure_grid.n_layers, mean_molar_mass_amu
        ),
    )

    preparation_estimate = provider.estimate_memory_bytes(
        spectral_grid, species=("H2O", "CO")
    )
    rt_estimate = int(
        48
        * 2
        * pressure_grid.n_layers
        * spectral_grid.size
        * np.dtype(float).itemsize
    )
    estimated_peak_bytes = preparation_estimate + rt_estimate + 256 * 1024**2
    if estimated_peak_bytes > MAX_MEMORY_BYTES:
        raise MemoryError(
            "refusing ROBERT LBL run: conservative estimate "
            f"{estimated_peak_bytes / 1024**3:.2f} GiB exceeds the "
            f"{MAX_MEMORY_GIB:.2f} GiB process ceiling"
        )

    start = perf_counter()
    prepared = provider.prepare(
        spectral_grid,
        pressure_grid,
        species=("H2O", "CO"),
    )
    evaluated = provider.evaluate(atmosphere, prepared)
    gas_optical_depth = assemble_gas_optical_depth(
        atmosphere,
        evaluated,
        gravity_m_s2=GRAVITY_M_S2,
        # This deliberate request verifies that LBL still forces direct sum.
        gas_combination="random_overlap",
        retain_species_tau=True,
    )
    radiance = solve_emission_spectrum(
        gas_optical_depth,
        geometry=gauss_legendre_disk_geometry(n_mu=8),
        bottom_boundary="blackbody",
        thermal_integration_backend="auto",
    )
    duration_seconds = perf_counter() - start
    if gas_optical_depth.metadata["gas_combination"] != "sum_by_g":
        raise RuntimeError("line-by-line optical depths were not added directly")
    return np.pi * np.asarray(radiance.values), {
        "n_layers": pressure_grid.n_layers,
        "n_wavelength": spectral_grid.size,
        "provider_memory_estimate_bytes": preparation_estimate,
        "benchmark_memory_estimate_bytes": estimated_peak_bytes,
        "provider_metadata": dict(prepared.metadata),
        "gas_metadata": dict(gas_optical_depth.metadata),
        "duration_seconds": duration_seconds,
    }


def _lsf_target(native_wavelength_micron: np.ndarray) -> np.ndarray:
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    margin = (
        LSF_SUPPORT_SIGMA
        * native_wavelength_micron[-1]
        * fwhm_to_sigma
        / LSF_RESOLVING_POWER
    )
    lower = native_wavelength_micron[0] + margin
    upper = native_wavelength_micron[-1] - margin
    count = int(np.floor(LSF_RESOLVING_POWER * np.log(upper / lower))) + 1
    target = lower * np.exp(np.arange(count, dtype=float) / LSF_RESOLVING_POWER)
    return target[target <= upper]


def _convolve(
    native_wavelength_micron: np.ndarray,
    target_wavelength_micron: np.ndarray,
    flux_w_m2_m: np.ndarray,
) -> np.ndarray:
    native_grid = SpectralGrid.from_array(
        native_wavelength_micron,
        unit="micron",
        role="rt_native",
    )
    observation = Observation.from_arrays(
        wavelength=target_wavelength_micron,
        flux=np.zeros(target_wavelength_micron.size),
        uncertainty=np.ones(target_wavelength_micron.size),
        flux_unit="W m^-3",
        observable="spectral_flux",
        instrument="Gaussian-R100000",
    )
    response = GaussianHighResolutionResponse(
        resolving_power=LSF_RESOLVING_POWER,
        kernel_support=LSF_SUPPORT_SIGMA,
    ).prepare(observation, native_grid)
    spectrum = Spectrum(
        spectral_grid=native_grid,
        values=flux_w_m2_m,
        unit="W m^-3",
        observable="spectral_flux",
    )
    return np.asarray(response.observe(spectrum).values)


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    relative = candidate / reference - 1.0
    return {
        "median_signed_relative_error": float(np.median(relative)),
        "rms_relative_error": float(np.sqrt(np.mean(relative**2))),
        "p95_absolute_relative_error": float(
            np.percentile(np.abs(relative), 95.0)
        ),
        "max_absolute_relative_error": float(np.max(np.abs(relative))),
        "integrated_flux_relative_error": float(
            np.sum(candidate) / np.sum(reference) - 1.0
        ),
    }


def _plot(
    wavelength_native: np.ndarray,
    p_rt_native: np.ndarray,
    robert_native: np.ndarray,
    wavelength_lsf: np.ndarray,
    p_rt_lsf: np.ndarray,
    robert_lsf: np.ndarray,
) -> None:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].plot(
        wavelength_native,
        p_rt_native,
        color="0.65",
        linewidth=0.8,
        label="petitRADTRANS R=1e6",
    )
    axes[0].plot(
        wavelength_native,
        robert_native,
        color="mediumpurple",
        linewidth=0.8,
        alpha=0.8,
        label="ROBERT R=1e6",
    )
    axes[0].plot(
        wavelength_lsf,
        p_rt_lsf,
        color="0.15",
        linewidth=1.4,
        label="petitRADTRANS + Gaussian R=100,000",
    )
    axes[0].plot(
        wavelength_lsf,
        robert_lsf,
        color="rebeccapurple",
        linewidth=1.2,
        linestyle="--",
        label="ROBERT + Gaussian R=100,000",
    )
    axes[0].set_ylabel("Top-of-atmosphere flux (W m$^{-3}$)")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    residual_percent = 100.0 * (robert_lsf / p_rt_lsf - 1.0)
    axes[1].axhline(0.0, color="0.3", linewidth=0.8)
    axes[1].plot(wavelength_lsf, residual_percent, color="rebeccapurple")
    axes[1].set_ylabel("ROBERT - pRT (%)")
    axes[1].set_xlabel("Wavelength (micron)")
    figure.suptitle("Clear CO/H$_2$O K-band thermal emission")
    figure.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)


def main() -> dict[str, object]:
    reference_80 = _load_reference(REFERENCE_80)
    reference_160 = _load_reference(REFERENCE_160)
    table_records: dict[str, dict[str, object]] = {}
    for species, path in TABLE_PATHS.items():
        checksum = _file_sha256(path)
        if checksum != EXPECTED_TABLE_SHA256[species]:
            raise ValueError(f"unexpected {species} line-by-line table checksum")
        table_records[species] = {
            "path": str(path.relative_to(ROOT)),
            "file_bytes": path.stat().st_size,
            "sha256": checksum,
        }
    wavelength = np.asarray(reference_160["wavelength_micron"], dtype=float)
    if not np.array_equal(
        wavelength, np.asarray(reference_80["wavelength_micron"], dtype=float)
    ):
        raise ValueError("pRT convergence oracles must use one native wavelength grid")

    provider = LineByLineOpacityProvider.from_hdf_paths(
        TABLE_PATHS,
        checksum=False,
        max_memory_bytes=1 * 1024**3,
    )
    robert_80, robert_metadata_80 = _robert_flux(reference_80, provider)
    robert_160, robert_metadata_160 = _robert_flux(reference_160, provider)
    p_rt_80 = np.asarray(reference_80["flux_w_m2_m"], dtype=float)
    p_rt_160 = np.asarray(reference_160["flux_w_m2_m"], dtype=float)

    target = _lsf_target(wavelength)
    p_rt_80_lsf = _convolve(wavelength, target, p_rt_80)
    p_rt_160_lsf = _convolve(wavelength, target, p_rt_160)
    robert_160_lsf = _convolve(wavelength, target, robert_160)

    metrics = {
        "petitradtrans_80_vs_160_native": _metrics(p_rt_160, p_rt_80),
        "petitradtrans_80_vs_160_R100000": _metrics(
            p_rt_160_lsf, p_rt_80_lsf
        ),
        "robert_vs_petitradtrans_80_native": _metrics(p_rt_80, robert_80),
        "robert_vs_petitradtrans_160_native": _metrics(p_rt_160, robert_160),
        "robert_vs_petitradtrans_160_R100000": _metrics(
            p_rt_160_lsf, robert_160_lsf
        ),
    }
    thread_values = _thread_values()
    acceptance = {
        "petitradtrans_pressure_convergence_R100000": (
            metrics["petitradtrans_80_vs_160_R100000"]["rms_relative_error"]
            <= 0.0025
            and metrics["petitradtrans_80_vs_160_R100000"][
                "max_absolute_relative_error"
            ]
            <= 0.01
        ),
        "robert_end_to_end_R100000": (
            metrics["robert_vs_petitradtrans_160_R100000"]["rms_relative_error"]
            <= 0.02
            and metrics["robert_vs_petitradtrans_160_R100000"][
                "max_absolute_relative_error"
            ]
            <= 0.05
        ),
        "memory_below_2_GiB": _peak_rss_bytes() < MAX_MEMORY_BYTES,
        "cpu_threads_at_most_3": all(
            1 <= value <= 3 for value in thread_values.values()
        ),
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "comparison": "ROBERT_vs_petitRADTRANS3_R1e6_CO_H2O_K_band_emission",
        "status": "pass" if all(acceptance.values()) else "fail",
        "command": _reproducible_command(),
        "clear_atmosphere": True,
        "included_opacity": ["CO__HITEMP", "H2O__POKAZATEL"],
        "input_tables": table_records,
        "excluded_opacity": ["CIA", "Rayleigh", "clouds", "scattering"],
        "native_resolving_power": 1_000_000,
        "lsf": {
            "profile": "Gaussian",
            "resolving_power": int(LSF_RESOLVING_POWER),
            "kernel_support_sigma": LSF_SUPPORT_SIGMA,
        },
        "wavelength_native_micron": [float(wavelength[0]), float(wavelength[-1])],
        "n_wavelength_native": int(wavelength.size),
        "wavelength_R100000_micron": [float(target[0]), float(target[-1])],
        "n_wavelength_R100000": int(target.size),
        "gravity_m_s2": GRAVITY_M_S2,
        "pressure_bar": [1.0e-5, 100.0],
        "pressure_points": [80, 160],
        "temperature_profile_K": "900 + 900 * (log10(P_bar) + 5) / 7",
        "composition": reference_160["metadata"]["volume_mixing_ratios"],
        "mean_molar_mass_amu": reference_160["metadata"][
            "mean_molar_mass_amu"
        ],
        "metrics": metrics,
        "acceptance": acceptance,
        "resources": {
            "cpu_thread_limit": 3,
            "thread_values": thread_values,
            "memory_limit_bytes": MAX_MEMORY_BYTES,
            "measured_peak_rss_bytes": _peak_rss_bytes(),
            "robert_80": robert_metadata_80,
            "robert_160": robert_metadata_160,
            "petitradtrans_160": reference_160["metadata"]["resource_policy"],
        },
        "artifacts": {
            "spectrum_npz": str(SPECTRA.relative_to(ROOT)),
            "figure_png": str(FIGURE.relative_to(ROOT)),
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SPECTRA,
        wavelength_native_micron=wavelength,
        petitradtrans_160_flux_w_m2_m=p_rt_160,
        robert_160_flux_w_m2_m=robert_160,
        wavelength_R100000_micron=target,
        petitradtrans_160_R100000_flux_w_m2_m=p_rt_160_lsf,
        robert_160_R100000_flux_w_m2_m=robert_160_lsf,
    )
    _plot(
        wavelength,
        p_rt_160,
        robert_160,
        target,
        p_rt_160_lsf,
        robert_160_lsf,
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit("LBL K-band validation did not meet all acceptance gates")
    return report


if __name__ == "__main__":
    main()
