"""Run one isolated external Version-2 Stage-1 pure-absorption worker."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import tempfile
from time import perf_counter

import numpy as np


PICASO_PYTHON = Path("/opt/miniconda3/envs/picaso-v4/bin/python")
PRT_PYTHON = Path("/opt/miniconda3/envs/petitradtrans-stable/bin/python")
PICASO_REFERENCE = Path("/Users/jaketaylor/Dropbox/picaso-v4/reference")

_CACHE_ROOT = Path(tempfile.gettempdir()) / "robert-emission-intercomparison-v2"
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE_ROOT / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("OMPI_MCA_btl", "self")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _planck_radiance(wavelength_micron: np.ndarray, temperature_k: float) -> np.ndarray:
    from scipy.constants import c, h, k

    wavelength_m = np.asarray(wavelength_micron, dtype=float) * 1.0e-6
    exponent = h * c / (wavelength_m * k * float(temperature_k))
    return 2.0 * h * c**2 / wavelength_m**5 / np.expm1(exponent)


def _formal_absorption(contract: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Independent exact absorbing formal solution used in the PICASO process.

    PICASO 4.0's low-level scattering solver does not provide a finite result
    at exact single-scattering albedo zero. This worker therefore retains the
    exact absorption-only formal reference separately from a native PICASO
    result and never substitutes a small non-zero albedo.
    """

    wavelength = contract["wavelength_micron"]
    source = _planck_radiance(wavelength, float(contract["temperature_k"]))
    mu = contract["emission_mu"]
    disk_weights = contract["disk_weights"]
    layer_tau = contract["layer_tau"]
    boundaries = contract["bottom_boundary"]
    flux = np.empty((layer_tau.shape[0], wavelength.size))
    vertical = np.empty((layer_tau.shape[0], layer_tau.shape[1] + 1, wavelength.size))
    for case_index, tau in enumerate(layer_tau):
        cumulative_before = np.zeros_like(tau)
        cumulative_before[1:] = np.cumsum(tau[:-1], axis=0)
        case_vertical = np.zeros((tau.shape[0] + 1, wavelength.size))
        for cosine, weight in zip(mu, disk_weights, strict=True):
            transmission_before = np.exp(-cumulative_before / cosine)
            escape = -np.expm1(-tau / cosine)
            case_vertical[:-1] += (
                float(weight) * transmission_before * escape * source[None, :]
            )
            if str(boundaries[case_index]) == "blackbody":
                case_vertical[-1] += (
                    float(weight)
                    * np.exp(-np.sum(tau, axis=0) / cosine)
                    * source
                )
        vertical[case_index] = np.pi * case_vertical
        flux[case_index] = np.sum(vertical[case_index], axis=0)
    return flux, vertical


def _petitradtrans(contract: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    from scipy.constants import c
    from petitRADTRANS.radtrans import fcore

    wavelength = contract["wavelength_micron"]
    frequency = c / (wavelength * 1.0e-6)
    layer_tau = contract["layer_tau"]
    boundary = contract["bottom_boundary"]
    flux = np.full((layer_tau.shape[0], wavelength.size), np.nan)
    runtime = np.zeros(layer_tau.shape[0])
    for case_index, tau in enumerate(layer_tau):
        if str(boundary[case_index]) != "blackbody":
            continue
        cumulative = np.concatenate(
            (np.zeros((1, wavelength.size)), np.cumsum(tau, axis=0)), axis=0
        )
        optical_depth = cumulative.T[None, :, None, :]
        started = perf_counter()
        flux_nu_cgs, _ = fcore.compute_ck_flux(
            frequency,
            contract["temperature_edges_k"][case_index],
            np.array([1.0]),
            contract["emission_mu"],
            contract["legendre_weights"],
            optical_depth,
            0,
        )
        runtime[case_index] = perf_counter() - started
        flux[case_index] = flux_nu_cgs * 1.0e-3 * c / (wavelength * 1.0e-6) ** 2
    return flux, runtime


def _picaso_exact_zero_probe(contract: dict[str, np.ndarray]) -> dict[str, object]:
    """Probe, but do not use, the PICASO exact-zero scattering interface."""

    if Path(os.environ.get("picaso_refdata", "")).resolve() != PICASO_REFERENCE:
        raise RuntimeError(f"picaso_refdata must be exactly {PICASO_REFERENCE}")
    for name in ("NUMBA_CACHE_DIR", "MPLCONFIGDIR"):
        path = Path(os.environ.get(name, ""))
        if not path.is_dir() or not os.access(path, os.W_OK):
            raise RuntimeError(f"{name} must name an existing writable directory")

    from picaso.fluxes import get_thermal_1d

    selected = int(np.flatnonzero(contract["bottom_boundary"] == "blackbody")[0])
    wavelength = contract["wavelength_micron"]
    pressure = contract["pressure_edges_bar"].copy() * 1.0e6
    pressure[0] = 0.0
    tau = contract["layer_tau"][selected]
    try:
        point, _ = get_thermal_1d(
            pressure.size,
            1.0e4 / wavelength,
            wavelength.size,
            contract["emission_mu"].size,
            1,
            contract["temperature_edges_k"][selected],
            tau,
            np.zeros_like(tau),
            np.zeros_like(tau),
            pressure,
            contract["emission_mu"][:, None],
            np.zeros(wavelength.size),
            1,
            np.zeros(wavelength.size),
            0,
        )
        finite = bool(np.all(np.isfinite(point)))
        return {
            "attempted": True,
            "finite": finite,
            "result": "finite" if finite else "non_finite_at_exact_omega0_zero",
        }
    except Exception as error:  # capability evidence is retained, not hidden
        return {
            "attempted": True,
            "finite": False,
            "result": "exception_at_exact_omega0_zero",
            "exception": f"{type(error).__name__}: {error}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("picaso", "petitradtrans"))
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--probe-native-picaso", action="store_true")
    args = parser.parse_args()
    expected_python = PICASO_PYTHON if args.model == "picaso" else PRT_PYTHON
    if os.path.realpath(os.sys.executable) != os.path.realpath(expected_python):
        raise RuntimeError(f"{args.model} must run with {expected_python}")
    contract = _load(args.contract)
    started = perf_counter()
    if args.model == "picaso":
        flux, vertical = _formal_absorption(contract)
        runtime = np.full(flux.shape[0], (perf_counter() - started) / flux.shape[0])
        supported = np.ones(flux.shape[0], dtype=bool)
        implementation = "independent_absorption_formal_reference_in_picaso_process"
        native_probe = (
            _picaso_exact_zero_probe(contract)
            if args.probe_native_picaso
            else {"attempted": False}
        )
        limitation = (
            "PICASO 4.0 native low-level thermal path is not used for the shared exact-omega0=0 "
            "gate; the independently implemented formal reference is retained and labelled."
        )
    else:
        flux, runtime = _petitradtrans(contract)
        vertical = np.full(
            (flux.shape[0], contract["layer_tau"].shape[1] + 1, flux.shape[1]),
            np.nan,
        )
        supported = contract["bottom_boundary"] == "blackbody"
        implementation = "petitRADTRANS_3_fcore_compute_ck_flux"
        native_probe = {"attempted": False}
        limitation = (
            "The stable pRT fcore shared-tau interface fixes a thermal lower boundary; "
            "no-bottom cases are explicitly unsupported and stored as NaN."
        )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    metadata = {
        "model": args.model,
        "mode": "version_2_stage_1_shared_grey_pure_absorption",
        "implementation": implementation,
        "version": importlib.metadata.version(
            "picaso" if args.model == "picaso" else "petitRADTRANS"
        ),
        "python": os.path.realpath(os.sys.executable),
        "scattering_enabled": False,
        "single_scattering_albedo": 0.0,
        "limitation": limitation,
        "native_exact_zero_probe": native_probe,
        "known_warnings": (
            ["Optional Vega spectrum absent; Version 2 uses an explicit blackbody star"]
            if args.model == "picaso"
            else []
        ),
        "peak_rss_raw": usage.ru_maxrss,
        "peak_rss_platform_units": "bytes_on_macos_kibibytes_on_linux",
        "platform": platform.platform(),
        "wall_time_s": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        case_id=contract["case_id"],
        wavelength_micron=contract["wavelength_micron"],
        flux_w_m2_m=flux,
        vertical_flux_contribution_w_m2_m=vertical,
        supported_case_mask=supported,
        runtime_s=runtime,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )


if __name__ == "__main__":
    main()
