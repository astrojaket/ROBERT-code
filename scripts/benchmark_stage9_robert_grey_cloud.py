#!/usr/bin/env python3
"""Benchmark exact Stage 9 ROBERT grey clouds with the Numba CPU path.

The Stage 9 source and project products are external, read-only inputs. This
script does not modify them. It is intended for guarded one-process checks
before a cluster retrieval rerun.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import resource
import re
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np

_GIB = 1024**3
_STAGE9_G_ORDINATES = 16
_MEASURED_STAGE9_PEAK_RSS_GIB = 7.46
_PEAK_RSS_SAFETY_FACTOR = 1.20
_DEFAULT_MAX_PEAK_RSS_GIB = 10.0
_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES = _GIB


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage9_source", type=Path)
    parser.add_argument("stage9_project_root", type=Path)
    parser.add_argument("--cloud-tau-5um", type=float, default=3.0)
    parser.add_argument("--cloud-top-pressure-bar", type=float, default=3.0e-3)
    parser.add_argument(
        "--warm-repeats",
        type=int,
        default=2,
        help="number of timed warm calls after the first call (default: 2)",
    )
    parser.add_argument(
        "--max-peak-rss-gib",
        type=float,
        default=_DEFAULT_MAX_PEAK_RSS_GIB,
        help=(
            "refuse when the conservative peak-RSS estimate exceeds this limit "
            f"(default: {_DEFAULT_MAX_PEAK_RSS_GIB:g} GiB)"
        ),
    )
    parser.add_argument(
        "--allow-unsafe-resource-use",
        action="store_true",
        help=(
            "bypass the memory preflight; use only in an isolated process after "
            "reviewing the estimate"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser


def _effect(spectrum: np.ndarray, clear: np.ndarray) -> dict[str, float]:
    difference_ppm = (spectrum - clear) * 1.0e6
    return {
        "max_abs_ppm": float(np.max(np.abs(difference_ppm))),
        "median_abs_ppm": float(np.median(np.abs(difference_ppm))),
        "rms_ppm": float(np.sqrt(np.mean(difference_ppm**2))),
        "signed_min_ppm": float(np.min(difference_ppm)),
        "signed_max_ppm": float(np.max(difference_ppm)),
    }


def _peak_rss_bytes() -> int:
    """Return process high-water RSS in bytes on macOS and Linux."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the BSDs report KiB.
    return value if sys.platform == "darwin" else value * 1024


def _physical_memory_bytes() -> int | None:
    """Return physical memory when the host exposes it through sysconf."""

    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size


def _available_memory_bytes() -> int | None:
    """Return a conservative available-memory estimate without new dependencies."""

    if sys.platform == "darwin":
        try:
            output = subprocess.run(
                ["vm_stat"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        page_match = re.search(r"page size of (\d+) bytes", output)
        if page_match is None:
            return None
        page_size = int(page_match.group(1))
        # Do not count compressed or wired pages as immediately available.
        names = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
        pages = 0
        for name in names:
            match = re.search(rf"^{re.escape(name)}:\s+(\d+)", output, re.MULTILINE)
            if match is not None:
                pages += int(match.group(1))
        return pages * page_size if pages > 0 else None

    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError):
            return None
    return None


def _stage9_shape_from_contract(common: dict[str, Any]) -> tuple[int, int, int]:
    """Return a conservative Stage 9 layer, wavelength, and g shape."""

    pressure_grids = common.get("pressure_grids", ())
    selected = next(
        (item for item in pressure_grids if int(item.get("n_cells", -1)) == 80),
        None,
    )
    if selected is None:
        raise ValueError("common contract does not contain the frozen 80-cell grid")
    spectral_contract = common.get("spectral_contract", {})
    native = spectral_contract.get("native_reference_wavelength_micron", ())
    if not native:
        raise ValueError("common contract does not contain the native wavelength grid")
    return int(selected["n_cells"]), len(native), _STAGE9_G_ORDINATES


def _estimate_peak_rss_bytes(layers: int, wavelengths: int, g_ordinates: int) -> int:
    """Estimate the full-grid SH4 high-water RSS before allocating it.

    The estimate counts the large band, RHS, eigenmode, particular-source, and
    flux arrays, then adds a 35 percent allowance for temporary copies and JIT
    work. It is also floored at 1.2 times the archived 7.46 GiB exact-case
    peak. The native wavelength count from the common contract is an upper
    bound, so this estimate is intentionally conservative.
    """

    columns = wavelengths * g_ordinates
    layer_columns = layers * columns
    array_elements = (
        11 * (4 * layers) * columns  # compact band matrix
        + (4 * layers) * columns  # RHS
        + 2 * layer_columns * 16  # top/bottom eigenmodes
        + 2 * layer_columns * 4  # top/bottom particular sources
        + 2 * layer_columns * 16  # top/bottom half-range fluxes
        + 12 * layer_columns  # input and intermediate source arrays
    )
    analytic_estimate = int(array_elements * 8 * 1.35)
    # The archived exact-case measurement reached 7.46 GiB. Keep a 20 percent
    # empirical margin above that observation, even if a future array-count
    # estimate is accidentally too small.
    measured_floor = int(_MEASURED_STAGE9_PEAK_RSS_GIB * _PEAK_RSS_SAFETY_FACTOR * _GIB)
    return max(analytic_estimate, measured_floor)


def _resource_preflight(
    common: dict[str, Any], *, max_peak_rss_gib: float, allow_unsafe: bool
) -> dict[str, Any]:
    """Refuse a full-grid run when memory pressure is predictably unsafe."""

    if not np.isfinite(max_peak_rss_gib) or max_peak_rss_gib <= 0.0:
        raise ValueError("--max-peak-rss-gib must be finite and positive")
    layers, wavelengths, g_ordinates = _stage9_shape_from_contract(common)
    estimated_peak = _estimate_peak_rss_bytes(layers, wavelengths, g_ordinates)
    physical = _physical_memory_bytes()
    available = _available_memory_bytes()
    limit = int(max_peak_rss_gib * _GIB)
    reasons: list[str] = []
    if estimated_peak > limit:
        reasons.append(
            f"estimate {estimated_peak / _GIB:.2f} GiB exceeds "
            f"configured limit {max_peak_rss_gib:.2f} GiB"
        )
    if physical is not None and estimated_peak > physical // 2:
        reasons.append(
            f"estimate {estimated_peak / _GIB:.2f} GiB exceeds half of physical "
            f"memory ({physical / _GIB:.2f} GiB)"
        )
    if available is not None:
        safe_available = max(0, available - _AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES)
        if estimated_peak > safe_available:
            reasons.append(
                f"estimate {estimated_peak / _GIB:.2f} GiB leaves less than "
                f"a 1.00 GiB safety margin in currently available memory "
                f"({available / _GIB:.2f} GiB)"
            )
    if reasons and not allow_unsafe:
        raise RuntimeError(
            "refusing unsafe Stage 9 full-grid benchmark: "
            + "; ".join(reasons)
            + ". Use --allow-unsafe-resource-use only in an isolated process."
        )
    if reasons and allow_unsafe:
        print(
            "WARNING: bypassing Stage 9 resource preflight: " + "; ".join(reasons),
            file=sys.stderr,
        )
    return {
        "allow_unsafe_resource_use": bool(allow_unsafe),
        "available_memory_bytes": available,
        "estimated_peak_rss_bytes": estimated_peak,
        "g_ordinates": g_ordinates,
        "layers": layers,
        "max_peak_rss_bytes": limit,
        "physical_memory_bytes": physical,
        "wavelengths_upper_bound": wavelengths,
    }


def main() -> None:
    args = _parser().parse_args()
    if args.cloud_tau_5um <= 0.0 or args.cloud_top_pressure_bar <= 0.0:
        raise ValueError("cloud optical depth and top pressure must be positive")
    if args.warm_repeats < 1:
        raise ValueError("--warm-repeats must be at least one")

    source = args.stage9_source.expanduser().resolve()
    project = args.stage9_project_root.expanduser().resolve()
    sys.path.insert(0, str(source / "examples"))

    import robert_exoplanets as robert
    import robert_exoplanets.diagnostics as diagnostics
    import robert_exoplanets.rt as rt

    diagnostics_path = source / "src" / "robert_exoplanets" / "diagnostics"
    diagnostics.__path__.append(str(diagnostics_path))
    stage9 = importlib.import_module("emission_intercomparison_v2_stage_9_native")

    common = stage9.load_common_contract(project / "contracts" / "common_contract.json")
    resource_guard = _resource_preflight(
        common,
        max_peak_rss_gib=args.max_peak_rss_gib,
        allow_unsafe=args.allow_unsafe_resource_use,
    )
    forward = stage9.build_native_forward(
        "robert", common, "grey_scattering_non_inverted"
    )
    values = stage9.truth_parameters(common, "grey_scattering_non_inverted")
    clear = np.load(
        project / "injections" / "robert" / "clear_non_inverted" / "native_mean.npz"
    )["eclipse_depth"]

    sh4_reference = rt.solve_thermal_sh4_spectrum
    rt.solve_thermal_sh4_spectrum = lambda *positional, **keywords: sh4_reference(
        *positional,
        **{**keywords, "backend": "numba", "boundary_backend": "numba"},
    )
    assemble_reference = robert.assemble_gas_optical_depth
    robert.assemble_gas_optical_depth = lambda *positional, **keywords: (
        assemble_reference(
            *positional,
            **{**keywords, "retain_species_tau": False},
        )
    )

    cases: list[dict[str, Any]] = []
    for label, tau, cloud_top in (
        ("original", 1.0, 1.0e-2),
        ("stronger", args.cloud_tau_5um, args.cloud_top_pressure_bar),
    ):
        for cloud, omega in (("absorbing", 0.0), ("scattering", 0.9)):
            trial = dict(values)
            trial["log10_cloud_tau_5um"] = float(np.log10(tau))
            trial["log10_cloud_top_pressure_bar"] = float(np.log10(cloud_top))
            trial["cloud_single_scattering_albedo"] = omega
            started = perf_counter()
            spectrum = forward.eclipse_depth(trial)
            first_call_seconds = perf_counter() - started
            warm_samples: list[float] = []
            for _ in range(args.warm_repeats):
                started = perf_counter()
                spectrum = forward.eclipse_depth(trial)
                warm_samples.append(perf_counter() - started)
            cases.append(
                {
                    "case": label,
                    "cloud": cloud,
                    "cloud_tau_5um": tau,
                    "cloud_top_pressure_bar": cloud_top,
                    "single_scattering_albedo": omega,
                    "first_call_seconds": first_call_seconds,
                    "warm_seconds": float(np.median(warm_samples)),
                    "warm_samples_seconds": warm_samples,
                    "effect": _effect(spectrum, clear),
                }
            )

    payload = {
        "schema_version": "1.0",
        "backend": "numba",
        "boundary_backend": "numba",
        "retain_species_tau": False,
        "layers": int(forward.pressure.n_layers),
        "wavelengths": int(forward.wavelength.size),
        "g_ordinates": int(forward.g_weights.size),
        "peak_rss_bytes": _peak_rss_bytes(),
        "resource_guard": resource_guard,
        "warm_repeats": args.warm_repeats,
        "cases": cases,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(encoded, end="")
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
