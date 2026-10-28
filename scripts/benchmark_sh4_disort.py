"""Compare ROBERT thermal SH4 fluxes with PythonicDISORT stream solutions.

The comparison uses a plane-parallel thermal source and a black lower
boundary. Optical depth increases downward. The top boundary has no incident
radiation. There is no direct beam, surface reflection, or delta-M scaling.

ROBERT supplies phase moments ``chi_l = (2*l + 1) * beta_l`` through degree
three to its P3/SH4 solver. PythonicDISORT takes the unweighted coefficients
``beta_l``. PythonicDISORT applies the thermal emissivity factor ``1 - omega``
to ``s_poly_coeffs`` internally, so this script passes the raw linear Planck
profile to both solvers.

The cases use arbitrary per-band radiance units. They test solver conventions
and angular convergence. They do not validate Mie cloud properties, spectral
integration, or retrieval-level accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from robert_exoplanets.rt.sh4 import solve_thermal_sh4

STREAM_COUNTS = (16, 32, 64)
SOURCE_QUADRATURE_ORDER = 8


@dataclass(frozen=True)
class ThermalScatteringCase:
    """Small plane-parallel thermal scattering case.

    ``layer_optical_depths`` are total-extinction optical depths per layer.
    ``planck_top`` and ``planck_bottom`` are arbitrary radiance units at the
    top and bottom interfaces. The profile is linear in total optical depth.
    ``phase_parameters`` are the HG asymmetry factors or mixture weights and
    asymmetry factors, as selected by ``phase_model``.
    """

    name: str
    layer_optical_depths: tuple[float, ...]
    single_scattering_albedo: float
    phase_model: str
    phase_parameters: tuple[float, ...]
    planck_top: float
    planck_bottom: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("case name must not be empty")
        if not self.layer_optical_depths or any(
            not np.isfinite(value) or value <= 0.0
            for value in self.layer_optical_depths
        ):
            raise ValueError("layer optical depths must be finite and positive")
        if not np.isfinite(self.single_scattering_albedo) or not (
            0.0 <= self.single_scattering_albedo < 1.0
        ):
            raise ValueError("single_scattering_albedo must be in [0, 1)")
        if self.phase_model not in {"rayleigh", "henyey_greenstein", "hg_mixture"}:
            raise ValueError(f"unsupported phase model: {self.phase_model}")
        if (
            not np.isfinite(self.planck_top)
            or not np.isfinite(self.planck_bottom)
            or self.planck_top < 0.0
            or self.planck_bottom < 0.0
        ):
            raise ValueError("Planck radiances must be non-negative")


def default_cases() -> tuple[ThermalScatteringCase, ...]:
    """Return three synthetic cases with distinct phase-function moments."""

    common_tau = (0.10, 0.20, 0.40, 0.80, 1.10)
    return (
        ThermalScatteringCase(
            name="rayleigh",
            layer_optical_depths=common_tau,
            single_scattering_albedo=0.75,
            phase_model="rayleigh",
            phase_parameters=(),
            planck_top=0.25,
            planck_bottom=0.70,
        ),
        ThermalScatteringCase(
            name="henyey_greenstein_g050",
            layer_optical_depths=common_tau,
            single_scattering_albedo=0.80,
            phase_model="henyey_greenstein",
            phase_parameters=(0.50,),
            planck_top=0.25,
            planck_bottom=0.70,
        ),
        ThermalScatteringCase(
            name="henyey_greenstein_mixture",
            layer_optical_depths=(0.12, 0.25, 0.50, 0.90, 1.40),
            single_scattering_albedo=0.88,
            phase_model="hg_mixture",
            phase_parameters=(0.70, 0.50, 0.82),
            planck_top=0.20,
            planck_bottom=0.80,
        ),
        ThermalScatteringCase(
            name="strong_forward_hg_g090_omega095",
            layer_optical_depths=common_tau,
            single_scattering_albedo=0.95,
            phase_model="henyey_greenstein",
            phase_parameters=(0.90,),
            planck_top=0.25,
            planck_bottom=0.70,
        ),
    )


def unweighted_legendre_coefficients(
    case: ThermalScatteringCase,
    count: int,
) -> np.ndarray:
    """Return PythonicDISORT's unweighted phase coefficients ``beta_l``."""

    if count < 1:
        raise ValueError("coefficient count must be positive")
    degree = np.arange(count, dtype=float)
    if case.phase_model == "rayleigh":
        coefficients = np.zeros(count, dtype=float)
        coefficients[0] = 1.0
        if count > 2:
            coefficients[2] = 0.1
        return coefficients
    if case.phase_model == "henyey_greenstein":
        (asymmetry,) = case.phase_parameters
        return asymmetry**degree
    if case.phase_model == "hg_mixture":
        weight, first_asymmetry, second_asymmetry = case.phase_parameters
        return (
            weight * first_asymmetry**degree
            + (1.0 - weight) * second_asymmetry**degree
        )
    raise ValueError(f"unsupported phase model: {case.phase_model}")


def robert_phase_moments(case: ThermalScatteringCase) -> np.ndarray:
    """Convert the first four unweighted coefficients to ROBERT's ``chi_l``."""

    beta = unweighted_legendre_coefficients(case, 4)
    degree = np.arange(4, dtype=float)
    return (2.0 * degree + 1.0) * beta


def thermal_profile(case: ThermalScatteringCase) -> tuple[np.ndarray, np.ndarray, float]:
    """Return optical-depth edges, interface Planck values, and slope."""

    layer_tau = np.asarray(case.layer_optical_depths, dtype=float)
    tau_edges = np.concatenate(([0.0], np.cumsum(layer_tau)))
    slope = (case.planck_bottom - case.planck_top) / tau_edges[-1]
    planck_levels = case.planck_top + slope * tau_edges
    return tau_edges, planck_levels, float(slope)


def _pythonicdisort_modules() -> tuple[Callable[..., Any], Any, str]:
    """Load the optional reference solver and its quadrature helper."""

    try:
        import PythonicDISORT
        from PythonicDISORT import subroutines
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError(
            "PythonicDISORT 1.8 is required; install it outside the ROBERT "
            "environment or provide its package directory on PYTHONPATH"
        ) from exc
    version = importlib.metadata.version("PythonicDISORT")
    if version != "1.8":
        raise RuntimeError(f"expected PythonicDISORT 1.8, found {version}")
    return PythonicDISORT.pydisort, subroutines, version


def _pythonicdisort_top_flux(
    case: ThermalScatteringCase,
    stream_count: int,
    pydisort: Callable[..., Any],
) -> float:
    """Return the upward diffuse thermal flux at the top boundary."""

    tau_edges, _planck_levels, slope = thermal_profile(case)
    nlayer = len(case.layer_optical_depths)
    beta = unweighted_legendre_coefficients(case, stream_count)
    beta_by_layer = np.tile(beta, (nlayer, 1))
    source_coefficients = np.tile(
        np.array([case.planck_top, slope], dtype=float), (nlayer, 1)
    )
    result = pydisort(
        tau_edges[1:],
        np.full(nlayer, case.single_scattering_albedo, dtype=float),
        stream_count,
        beta_by_layer,
        0.5,  # Unused when I0 is zero; must remain a principal angle.
        0.0,  # No direct beam.
        0.0,
        NLeg=stream_count,
        NFourier=1,
        b_pos=case.planck_bottom,
        b_neg=0.0,
        only_flux=True,
        f_arr=0.0,
        NT_cor=False,
        s_poly_coeffs=source_coefficients,
    )
    flux_up = result[1]
    value = float(np.asarray(flux_up(0.0)))
    if not np.isfinite(value):
        raise RuntimeError(
            f"PythonicDISORT returned a non-finite flux for {case.name}, "
            f"NQuad={stream_count}"
        )
    return value


def _robert_sh4_top_flux(
    case: ThermalScatteringCase,
    quadrature: Callable[..., tuple[np.ndarray, np.ndarray]],
    *,
    angular_stream_count: int,
) -> float:
    """Integrate ROBERT's emergent SH4 radiance over the upper hemisphere."""

    tau_edges, planck_levels, _slope = thermal_profile(case)
    layer_tau = np.asarray(case.layer_optical_depths, dtype=float)
    nlayer = layer_tau.size
    mu, weights = quadrature(angular_stream_count // 2)
    beta = unweighted_legendre_coefficients(case, 4)
    chi = (2.0 * np.arange(4, dtype=float) + 1.0) * beta
    phase_moments = np.broadcast_to(
        chi[:, None, None, None], (4, nlayer, 1, 1)
    ).copy()
    tau_array = layer_tau[:, None, None]
    omega_array = np.full_like(tau_array, case.single_scattering_albedo)
    asymmetry_array = np.full_like(tau_array, beta[1])
    result = solve_thermal_sh4(
        tau_array,
        omega_array,
        asymmetry_array,
        planck_levels[:, None],
        mu,
        bottom_planck_radiance=np.array([case.planck_bottom], dtype=float),
        phase_function_moments=phase_moments,
        delta_m=False,
        source_quadrature_order=SOURCE_QUADRATURE_ORDER,
        boundary_backend="scipy",
    )
    intensity = result.point_radiance[:, 0, 0]
    flux = 2.0 * np.pi * float(np.dot(weights * mu, intensity))
    if not np.isfinite(flux):
        raise RuntimeError(f"ROBERT SH4 returned a non-finite flux for {case.name}")
    return flux


def _timed_median(
    function: Callable[[], float],
    *,
    warmups: int,
    samples: int,
) -> tuple[float, float, list[float]]:
    """Return the final value, median wall time, and individual timings."""

    for _ in range(warmups):
        function()
    values: list[float] = []
    elapsed: list[float] = []
    for _ in range(samples):
        start = time.perf_counter()
        value = function()
        elapsed.append(time.perf_counter() - start)
        values.append(value)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(elapsed)):
        raise RuntimeError("benchmark produced non-finite values or timings")
    return values[-1], float(np.median(elapsed)), elapsed


def _case_input(case: ThermalScatteringCase) -> dict[str, Any]:
    values = asdict(case)
    values["layer_optical_depths"] = list(case.layer_optical_depths)
    values["phase_parameters"] = list(case.phase_parameters)
    return values


def benchmark_case(
    case: ThermalScatteringCase,
    *,
    pydisort: Callable[..., Any],
    quadrature: Callable[..., tuple[np.ndarray, np.ndarray]],
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    """Run one synthetic case and report accuracy and convergence."""

    stream_results: dict[str, dict[str, Any]] = {}
    for stream_count in STREAM_COUNTS:
        flux, median_seconds, timings = _timed_median(
            lambda count=stream_count: _pythonicdisort_top_flux(
                case, count, pydisort
            ),
            warmups=warmups,
            samples=samples,
        )
        stream_results[str(stream_count)] = {
            "upward_flux_top": flux,
            "median_seconds": median_seconds,
            "sample_seconds": timings,
        }

    sh4_flux, sh4_median_seconds, sh4_timings = _timed_median(
        lambda: _robert_sh4_top_flux(
            case,
            quadrature,
            angular_stream_count=STREAM_COUNTS[-1],
        ),
        warmups=warmups,
        samples=samples,
    )
    reference_flux = stream_results[str(STREAM_COUNTS[-1])]["upward_flux_top"]
    flux_16 = stream_results[str(STREAM_COUNTS[0])]["upward_flux_top"]
    flux_32 = stream_results[str(STREAM_COUNTS[1])]["upward_flux_top"]
    flux_64 = stream_results[str(STREAM_COUNTS[2])]["upward_flux_top"]
    return {
        "input": _case_input(case),
        "reference_streams": STREAM_COUNTS[-1],
        "reference_flux_is_converged_32_to_64": bool(
            abs(flux_32 - flux_64) / max(abs(flux_64), 1.0e-300) <= 1.0e-4
        ),
        "stream_convergence": {
            "relative_change_16_to_32": abs(flux_32 - flux_16)
            / max(abs(flux_32), 1.0e-300),
            "relative_change_32_to_64": abs(flux_64 - flux_32)
            / max(abs(flux_64), 1.0e-300),
        },
        "pythonicdisort": stream_results,
        "robert_sh4": {
            "upward_flux_top": sh4_flux,
            "relative_difference_from_64_stream": (sh4_flux - reference_flux)
            / max(abs(reference_flux), 1.0e-300),
            "absolute_relative_difference_from_64_stream": abs(
                sh4_flux - reference_flux
            )
            / max(abs(reference_flux), 1.0e-300),
            "median_seconds": sh4_median_seconds,
            "sample_seconds": sh4_timings,
            "angular_quadrature_streams": STREAM_COUNTS[-1],
        },
    }


def _git_commit(source_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def build_report(
    *,
    source_root: Path,
    source_commit: str | None,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    """Run all cases and return a portable JSON-ready report."""

    if source_commit is not None:
        normalized_commit = source_commit.lower()
        if re.fullmatch(r"[0-9a-f]{40}", normalized_commit) is None:
            raise ValueError("source_commit must be a full 40-character Git SHA-1")
        source_commit = normalized_commit

    pydisort, subroutines, disort_version = _pythonicdisort_modules()
    cases = [
        benchmark_case(
            case,
            pydisort=pydisort,
            quadrature=subroutines.Gauss_Legendre_quad,
            warmups=warmups,
            samples=samples,
        )
        for case in default_cases()
    ]
    try:
        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        source_hash = None
    return {
        "schema_version": 1,
        "benchmark": "ROBERT_SH4_vs_PythonicDISORT_thermal_scattering",
        "provenance": {
            "robert_source_root": str(source_root.resolve()),
            "robert_git_commit": source_commit or _git_commit(source_root),
            "benchmark_script_sha256": source_hash,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "pythonicdisort_version": disort_version,
        },
        "method": {
            "optical_depth_direction": "increases_downward",
            "geometry": "plane_parallel",
            "thermal_source": "linear_Planck_radiance_in_total_optical_depth",
            "source_units": "arbitrary_per_band_radiance_units",
            "top_boundary": "no_incident_radiation",
            "bottom_boundary": "black_surface_at_planck_bottom",
            "direct_beam": False,
            "surface_reflection": False,
            "delta_m_scaling": False,
            "nakajima_tanaka_corrections": False,
            "robert_phase_convention": "chi_l=(2*l+1)*beta_l; use l=0..3",
            "pythonicdisort_phase_convention": "unweighted beta_l; NLeg=NQuad",
            "thermal_emissivity": "PythonicDISORT multiplies raw s_poly_coeffs by (1-omega)",
            "reference_flux": "PythonicDISORT upward flux at tau=0; highest stream count",
            "reported_stream_counts": list(STREAM_COUNTS),
            "reference_convergence_check": "relative 32-to-64 flux change <= 1e-4",
            "timing": {
                "warmups": warmups,
                "samples": samples,
                "summary": "median wall time; includes solver call and top-flux evaluation",
            },
            "limitations": [
                "Synthetic single-channel source profiles; not spectral-integrated Planck emission.",
                "No Mie optical properties, cloud retrieval, direct beam, or surface reflection.",
                "The 64-stream result is a finite-stream reference; convergence is reported, not assumed.",
                "The comparison disables delta-M in both solvers to use identical unscaled phase moments.",
            ],
        },
        "cases": cases,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the JSON result. Use a run directory outside the checkout.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="ROBERT source root used to record the Git commit.",
    )
    parser.add_argument(
        "--source-commit",
        help="exact ROBERT commit to record when Git is not available on the worker.",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args(argv)
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.source_commit is not None:
        args.source_commit = args.source_commit.lower()
        if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
            parser.error("--source-commit must be a full 40-character Git SHA-1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report(
        source_root=args.source_root,
        source_commit=args.source_commit,
        warmups=args.warmups,
        samples=args.samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    for case in report["cases"]:
        print(
            f"{case['input']['name']}: "
            f"SH4={case['robert_sh4']['upward_flux_top']:.8g}, "
            f"DISORT64={case['pythonicdisort']['64']['upward_flux_top']:.8g}, "
            f"abs relative difference="
            f"{case['robert_sh4']['absolute_relative_difference_from_64_stream']:.4g}, "
            f"32-to-64 change={case['stream_convergence']['relative_change_32_to_64']:.4g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
