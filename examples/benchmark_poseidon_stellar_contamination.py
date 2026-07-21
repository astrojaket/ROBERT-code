"""Compare ROBERT's TSLE transform with the official POSEIDON source.

This benchmark deliberately isolates the stellar-contamination transform from
stellar-atmosphere interpolation. It executes the unmodified function bodies
from ``POSEIDON/stellar.py`` in an otherwise clean namespace, then supplies the
same independently generated analytic radiances to POSEIDON and ROBERT. Full
generated arrays belong under ``examples/outputs/`` and are not committed.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Callable

import numpy as np

from robert_exoplanets import (
    SpectralGrid,
    Spectrum,
    StellarContaminationModel,
    StellarHeterogeneity,
)

POSEIDON_REPOSITORY = "https://github.com/MartianColonist/POSEIDON"
EXPECTED_POSEIDON_VERSION = "1.4.0"
EXPECTED_POSEIDON_COMMIT = "d1632214c9f3087e367a8d752454f3668fc30e18"
FACTOR_ABS_TOLERANCE = 2.0e-15
DEPTH_ABS_TOLERANCE = 2.0e-17
DEPTH_RMS_TOLERANCE = 5.0e-18


@dataclass(frozen=True)
class Case:
    name: str
    fractions: tuple[float, ...]
    temperatures_k: tuple[float, ...]
    kinds: tuple[str, ...]


CASES = (
    Case("homogeneous", (), (), ()),
    Case("cool_spot", (0.18,), (4200.0,), ("spot",)),
    Case("hot_facula", (0.12,), (6000.0,), ("facula",)),
    Case("spot_and_facula", (0.18, 0.12), (4200.0, 6000.0), ("spot", "facula")),
)


def _official_poseidon_functions(
    source_file: Path,
) -> tuple[Callable, Callable]:
    """Compile only POSEIDON's two dependency-free TSLE function bodies."""

    source = source_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_file))
    wanted = {
        "stellar_contamination_single_spot",
        "stellar_contamination_general",
    }
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            node.decorator_list = []
            functions.append(node)
    if {node.name for node in functions} != wanted:
        raise RuntimeError("official POSEIDON TSLE functions were not found")
    namespace: dict[str, object] = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_file), "exec"), namespace)
    return (
        namespace["stellar_contamination_single_spot"],
        namespace["stellar_contamination_general"],
    )


def _planck_radiance(wavelength_micron: np.ndarray, temperature_k: float) -> np.ndarray:
    """Independent analytic input spectrum in W m^-3 sr^-1."""

    h = 6.62607015e-34
    c = 299792458.0
    k = 1.380649e-23
    wavelength_m = wavelength_micron * 1.0e-6
    return (2.0 * h * c**2 / wavelength_m**5) / np.expm1(
        h * c / (wavelength_m * k * temperature_k)
    )


def _spectrum(grid: SpectralGrid, values: np.ndarray) -> Spectrum:
    return Spectrum(
        spectral_grid=grid,
        values=values,
        unit="W m^-3 sr^-1",
        observable="stellar_spectral_radiance",
        metadata={"stellar_model": "analytic_planck_benchmark_input"},
    )


def _planet_depth(wavelength_micron: np.ndarray) -> np.ndarray:
    """Nontrivial synthetic planetary spectrum used only to test multiplication."""

    return (
        0.008
        + 2.5e-4 * np.exp(-0.5 * ((wavelength_micron - 1.4) / 0.12) ** 2)
        + 1.7e-4 * np.exp(-0.5 * ((wavelength_micron - 4.3) / 0.25) ** 2)
        + 3.0e-5 * np.sin(2.0 * np.pi * np.log(wavelength_micron / 0.6))
    )


def run_benchmark(poseidon_root: Path) -> dict[str, object]:
    source_file = poseidon_root / "POSEIDON" / "stellar.py"
    setup_file = poseidon_root / "setup.py"
    if not source_file.is_file() or not setup_file.is_file():
        raise FileNotFoundError(
            "--poseidon-source must be an official POSEIDON source checkout"
        )
    setup_text = setup_file.read_text(encoding="utf-8")
    if f'version="{EXPECTED_POSEIDON_VERSION}"' not in setup_text:
        raise RuntimeError(
            f"benchmark requires POSEIDON {EXPECTED_POSEIDON_VERSION}"
        )
    commit = subprocess.check_output(
        ["git", "-C", str(poseidon_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_POSEIDON_COMMIT:
        raise RuntimeError(
            f"benchmark requires POSEIDON commit {EXPECTED_POSEIDON_COMMIT}, got {commit}"
        )
    single_spot, general = _official_poseidon_functions(source_file)

    wavelength = np.geomspace(0.6, 12.0, 257)
    sample_indices = np.array([0, 64, 128, 192, 256])
    grid = SpectralGrid.from_array(wavelength, unit="micron", role="rt_native")
    photosphere_values = _planck_radiance(wavelength, 5200.0)
    photosphere = _spectrum(grid, photosphere_values)
    planet_depth = _planet_depth(wavelength)
    case_summaries = []
    all_factor_residuals = []
    all_depth_residuals = []

    for case in CASES:
        region_values = tuple(
            _planck_radiance(wavelength, temperature)
            for temperature in case.temperatures_k
        )
        poseidon_factor = general(
            np.asarray(case.fractions, dtype=float),
            np.asarray(region_values, dtype=float).reshape(
                len(region_values), wavelength.size
            ),
            photosphere_values,
        )
        if len(region_values) == 1:
            np.testing.assert_array_equal(
                poseidon_factor,
                single_spot(case.fractions[0], region_values[0], photosphere_values),
            )
        robert_model = StellarContaminationModel(
            photosphere,
            tuple(
                StellarHeterogeneity(
                    name=kind,
                    kind=kind,
                    spectrum=_spectrum(grid, values),
                    covering_fraction=fraction,
                )
                for kind, values, fraction in zip(
                    case.kinds, region_values, case.fractions, strict=True
                )
            ),
        )
        robert_factor = robert_model.evaluate().contamination_factor.values
        poseidon_observed_depth = poseidon_factor * planet_depth
        robert_observed_depth = robert_factor * planet_depth
        factor_residual = robert_factor - poseidon_factor
        depth_residual = robert_observed_depth - poseidon_observed_depth
        all_factor_residuals.append(factor_residual)
        all_depth_residuals.append(depth_residual)
        case_summaries.append(
            {
                "name": case.name,
                "fractions": list(case.fractions),
                "temperatures_k": list(case.temperatures_k),
                "sample_wavelength_micron": wavelength[sample_indices].tolist(),
                "poseidon_contamination_factor": poseidon_factor[sample_indices].tolist(),
                "poseidon_observed_depth": poseidon_observed_depth[sample_indices].tolist(),
                "max_abs_factor_residual": float(np.max(np.abs(factor_residual))),
                "max_abs_depth_residual": float(np.max(np.abs(depth_residual))),
                "rms_depth_residual": float(np.sqrt(np.mean(depth_residual**2))),
            }
        )

    factors = np.concatenate(all_factor_residuals)
    depths = np.concatenate(all_depth_residuals)
    metrics = {
        "max_abs_factor_residual": float(np.max(np.abs(factors))),
        "max_abs_depth_residual": float(np.max(np.abs(depths))),
        "rms_depth_residual": float(np.sqrt(np.mean(depths**2))),
    }
    passed = (
        metrics["max_abs_factor_residual"] <= FACTOR_ABS_TOLERANCE
        and metrics["max_abs_depth_residual"] <= DEPTH_ABS_TOLERANCE
        and metrics["rms_depth_residual"] <= DEPTH_RMS_TOLERANCE
    )
    return {
        "schema_version": 1,
        "benchmark": "poseidon_stellar_contamination_transform",
        "validation_scope": (
            "Rackham/POSEIDON multiplicative transform only; analytic Planck inputs "
            "isolate the transform and do not validate stellar atmosphere grids"
        ),
        "poseidon": {
            "repository": POSEIDON_REPOSITORY,
            "version": EXPECTED_POSEIDON_VERSION,
            "commit": commit,
            "source_file": "POSEIDON/stellar.py",
            "source_sha256": sha256(source_file.read_bytes()).hexdigest(),
            "execution": "AST-extracted unmodified official function bodies with decorators removed",
        },
        "input": {
            "wavelength_range_micron": [0.6, 12.0],
            "wavelength_points": wavelength.size,
            "sampling": "geometric",
            "photosphere_temperature_k": 5200.0,
            "stellar_input": "independent analytic Planck radiance",
            "planet_depth": "nonconstant deterministic analytic spectrum",
        },
        "tolerances": {
            "max_abs_factor_residual": FACTOR_ABS_TOLERANCE,
            "max_abs_depth_residual": DEPTH_ABS_TOLERANCE,
            "rms_depth_residual": DEPTH_RMS_TOLERANCE,
            "basis": "float64 operation-order parity with no grid interpolation",
        },
        "metrics": metrics,
        "passed": passed,
        "cases": case_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--poseidon-source",
        type=Path,
        required=True,
        help="Official POSEIDON v1.4.0 source checkout",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/outputs/poseidon_stellar_contamination/summary.json"),
    )
    args = parser.parse_args()
    summary = run_benchmark(args.poseidon_source.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit("POSEIDON stellar-contamination parity failed")


if __name__ == "__main__":
    main()
