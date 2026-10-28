#!/usr/bin/env python3
"""Benchmark ROBERT's Virga cloud adapter and its Mie slab-cloud baseline.

This is a timing tool, not a physics-equivalence test. Both providers use the
same material, pressure layers, and wavelength samples. Virga calculates a
vertical eddy-sed cloud with its precomputed Mie table. The ROBERT baseline
calculates a full-column cloud slab with a fixed condensate fraction and
effective radius. Their cloud physics are not matched.

Run in the ``robert-exoplanets`` environment. Virga is optional. If
``--mie-directory`` is omitted, the script creates a small Virga Mie table
from the bundled MgSiO3 optical constants. Use ``--work-directory`` to keep
the generated table. Otherwise, the script removes its temporary files after
it writes the report.

Example::

    PYTHONPATH=/path/to/virga-exo \
    conda run -n robert-exoplanets python scripts/benchmark_virga_cloud.py \
      --layers 12 --wavelengths 8 --warmups 1 --samples 5 \
      --output /private/tmp/virga-benchmark.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable


_ROOT = Path(__file__).resolve().parents[1]
_OPTICAL_CONSTANTS = (
    _ROOT / "data" / "optical_constants" / "exo_skryer"
)
_MATERIAL = "MgSiO3"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mie-directory",
        type=Path,
        help="Use an existing Virga directory containing MgSiO3.mieff.",
    )
    parser.add_argument(
        "--work-directory",
        type=Path,
        help="Keep a generated Mie table in this directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the JSON report here. Print JSON to stdout when omitted.",
    )
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--wavelengths", type=int, default=8)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--mie-radii",
        type=int,
        default=20,
        help="Virga particle-radius bins (minimum 8).",
    )
    parser.add_argument(
        "--slab-quadrature-points",
        type=int,
        default=12,
        help="Lognormal Mie quadrature points for the slab baseline.",
    )
    parser.add_argument(
        "--skip-mie-slab",
        action="store_true",
        help="Time Virga only; omit ROBERT's Mie slab baseline.",
    )
    parser.add_argument(
        "--include-arrays",
        action="store_true",
        help="Include full output arrays in the JSON report.",
    )
    args = parser.parse_args()
    if args.layers < 1:
        parser.error("--layers must be positive")
    if args.wavelengths < 2:
        parser.error("--wavelengths must be at least two")
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.mie_radii < 8:
        parser.error("--mie-radii must be at least eight for Virga interpolation")
    if args.slab_quadrature_points < 1:
        parser.error("--slab-quadrature-points must be positive")
    if args.mie_directory is not None and args.work_directory is not None:
        parser.error("use only one of --mie-directory and --work-directory")
    return args


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_mib() -> float:
    # macOS reports bytes; Linux and the other supported Unix platforms report
    # KiB. Convert both cases to MiB for the JSON report.
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0**2)
    return value / 1024.0


def _memory_total_gib() -> float | None:
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return float(pages * page_size) / (1024.0**3)
    except (OSError, ValueError):
        pass
    try:
        import psutil

        return float(psutil.virtual_memory().total) / (1024.0**3)
    except ImportError:
        return None


def _hardware_details() -> dict[str, Any]:
    affinity_count = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except OSError:
            pass
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "allowed_cpu_count": affinity_count,
        "memory_total_gib": _memory_total_gib(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def _prepare_mie_table(
    virga: Any,
    refractive_index: Any,
    *,
    supplied_directory: Path | None,
    work_directory: Path | None,
    mie_radii: int,
) -> tuple[Path, dict[str, Any], float]:
    start = time.perf_counter()
    if supplied_directory is not None:
        mie_directory = supplied_directory.expanduser().resolve()
        generated = False
    elif work_directory is not None:
        root = work_directory.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        mie_directory = root / "mie"
        mie_directory.mkdir(parents=True, exist_ok=True)
        generated = not (mie_directory / f"{_MATERIAL}.mieff").is_file()
    else:
        raise ValueError("provide a Mie directory or a temporary work directory")

    mie_file = mie_directory / f"{_MATERIAL}.mieff"
    source_checksum: str | None = refractive_index.metadata.get("source_sha256")
    if generated:
        if mie_file.exists():
            raise FileExistsError(
                f"refusing to replace an existing Mie table: {mie_file}"
            )
        refrind_directory = mie_directory.parent / "refrind"
        refrind_directory.mkdir(parents=True, exist_ok=True)
        refrind_path = refrind_directory / f"{_MATERIAL}.refrind"
        import numpy as np

        refrind = np.column_stack(
            (
                np.arange(1, refractive_index.wavelength_micron.size + 1),
                refractive_index.wavelength_micron,
                refractive_index.real_index,
                refractive_index.imaginary_index,
            )
        )
        np.savetxt(
            refrind_path,
            refrind,
            fmt=("%d", "%.10g", "%.10g", "%.10g"),
        )
        virga.calc_mie_db(
            _MATERIAL,
            virga_dir=str(refrind_directory),
            dir_out=str(mie_directory),
            rmin=1.0e-8,
            rmax=1.0e-2,
            nradii=mie_radii,
            logspace=True,
            aggregates=False,
            fort_calc_mie=False,
        )
    elif not mie_file.is_file():
        raise FileNotFoundError(f"Virga Mie table is missing: {mie_file}")
    if not mie_file.is_file() or mie_file.stat().st_size == 0:
        raise RuntimeError(f"Virga did not create a non-empty Mie table: {mie_file}")
    elapsed = time.perf_counter() - start
    report = {
        "generated_in_this_run": generated,
        "directory": str(mie_directory),
        "table_file": str(mie_file),
        "table_bytes": mie_file.stat().st_size,
        "table_sha256": _sha256(mie_file),
        "refractive_index_file": str(_OPTICAL_CONSTANTS / f"{_MATERIAL}.txt"),
        "refractive_index_source_sha256": source_checksum,
        "radius_grid": {
            "minimum_cm": 1.0e-8,
            "maximum_cm": 1.0e-2,
            "count": mie_radii if generated else None,
            "spacing": "logarithmic",
        },
    }
    return mie_directory, report, elapsed


def _build_fixture(
    np: Any,
    *,
    layers: int,
    wavelengths: int,
    mie_directory: Path,
    refractive_index: Any,
    slab_quadrature_points: int,
    skip_mie_slab: bool,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    from robert_exoplanets.atmosphere import AtmosphereState
    from robert_exoplanets.core import PressureGrid, SpectralGrid
    from robert_exoplanets.forward.clouds import ParameterizedMieCloudModel
    from robert_exoplanets.forward.virga_cloud import VirgaEddySedCloudModel
    from robert_exoplanets.opacity import (
        EvaluatedCorrelatedKOpacity,
        PreparedCorrelatedKOpacity,
    )
    from robert_exoplanets.rt import assemble_gas_optical_depth

    pressure_edges = np.logspace(1.0, -5.0, layers + 1)
    pressure_grid = PressureGrid(
        edges=pressure_edges,
        centers=np.sqrt(pressure_edges[:-1] * pressure_edges[1:]),
        unit="bar",
        name="Virga cloud benchmark pressure grid",
    )
    temperature_edges = np.linspace(1800.0, 850.0, layers + 1)
    temperature_layers = 0.5 * (temperature_edges[:-1] + temperature_edges[1:])
    wavelength = np.geomspace(0.8, 20.0, wavelengths)
    spectral_grid = SpectralGrid(
        values=wavelength,
        unit="micron",
        role="native",
        name="Virga cloud benchmark wavelengths",
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature_layers,
        temperature_edges=temperature_edges,
        composition={"H2O": np.full(layers, 1.0e-3)},
        mean_molecular_weight=2.3,
    )
    prepared = PreparedCorrelatedKOpacity(
        provider_name="virga-cloud-benchmark",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("H2O",),
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        cache_key="virga-cloud-benchmark-opacity",
    )
    evaluated = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=np.full((1, layers, wavelengths, 1), 1.0e-25),
    )
    gas_optical_depth = assemble_gas_optical_depth(
        atmosphere,
        evaluated,
        gravity_m_s2=15.0,
    )
    virga_model = VirgaEddySedCloudModel(
        condensibles=(_MATERIAL,),
        gas_mass_mixing_ratios={_MATERIAL: 2.75e-3},
        mie_directory=mie_directory,
        fsed=0.5,
        kz_cm2_s=1.0e8,
        metallicity=1.0,
        geometric_stddev=2.0,
    )
    slab_parameters = {
        "log10_condensate_mass_fraction": float(np.log10(2.75e-3)),
        "log10_effective_radius_micron": float(np.log10(0.3)),
    }
    mie_slab_model = None
    if not skip_mie_slab:
        mie_slab_model = ParameterizedMieCloudModel(
            refractive_index_wavelength_micron=(),
            real_index_parameter_names=(),
            log10_imaginary_index_parameter_names=(),
            log10_condensate_mass_fraction_parameter=(
                "log10_condensate_mass_fraction"
            ),
            log10_effective_radius_micron_parameter=(
                "log10_effective_radius_micron"
            ),
            particle_density_kg_m3=3200.0,
            geometric_stddev=2.0,
            quadrature_points=slab_quadrature_points,
            refractive_index_extrapolation="raise",
            fixed_refractive_index=refractive_index,
        )
    fixture = {
        "pressure_layers": layers,
        "pressure_edges_bar": pressure_edges.tolist(),
        "wavelength_micron": wavelength.tolist(),
        "temperature_edges_k": temperature_edges.tolist(),
        "mean_molecular_weight_amu": 2.3,
        "gravity_m_s2": 15.0,
        "gas_opacity_fixture": {
            "species": "H2O",
            "kcoeff": 1.0e-25,
            "purpose": "small fixed input grid for cloud provider timing",
        },
        "virga_controls": {
            "condensible": _MATERIAL,
            "bottom_gas_mass_mixing_ratio_g_g": 2.75e-3,
            "fsed": 0.5,
            "kz_cm2_s": 1.0e8,
            "metallicity_linear_solar": 1.0,
            "geometric_stddev": 2.0,
            "supersaturation": 0.0,
            "multiple_scattering": "not computed by the cloud adapter",
        },
        "mie_slab_controls": None
        if skip_mie_slab
        else {
            "vertical_profile": "uniform full-column condensate mass fraction",
            "condensate_mass_fraction": 2.75e-3,
            "effective_radius_micron": 0.3,
            "geometric_stddev": 2.0,
            "particle_density_kg_m3": 3200.0,
            "lognormal_quadrature_points": slab_quadrature_points,
        },
        "physics_comparison": {
            "matched_material": True,
            "matched_pressure_layers": True,
            "matched_wavelength_samples": True,
            "matched_cloud_physics": False,
            "explanation": [
                "Virga calculates condensate and particle sizes from its eddy-sed model.",
                "The ROBERT Mie baseline uses a fixed full-column condensate fraction and effective radius.",
                "The Virga table and ROBERT Mie quadrature are separate optical calculations.",
                "Timing and optical outputs are descriptive; they do not establish equal physics.",
            ],
        },
    }
    return (
        gas_optical_depth,
        virga_model,
        mie_slab_model,
        slab_parameters,
        fixture,
    )


def _timed_runs(
    evaluate: Callable[[], Any],
    *,
    warmups: int,
    samples: int,
) -> tuple[Any, dict[str, Any]]:
    start = time.perf_counter()
    result = evaluate()
    cold_seconds = time.perf_counter() - start
    warmup_seconds = []
    for _ in range(warmups):
        start = time.perf_counter()
        result = evaluate()
        warmup_seconds.append(time.perf_counter() - start)
    sample_seconds = []
    for _ in range(samples):
        start = time.perf_counter()
        result = evaluate()
        sample_seconds.append(time.perf_counter() - start)
    return result, {
        "cold_seconds": cold_seconds,
        "warmup_seconds": warmup_seconds,
        "warm_sample_seconds": sample_seconds,
        "warm_sample_median_seconds": float(median(sample_seconds)),
        "warm_sample_min_seconds": min(sample_seconds),
        "warm_sample_max_seconds": max(sample_seconds),
    }


def _output_summary(cloud: Any, *, include_arrays: bool) -> dict[str, Any]:
    import numpy as np

    arrays = {
        "extinction_tau": np.asarray(cloud.extinction_tau, dtype=np.float64),
        "single_scattering_albedo": np.asarray(
            cloud.single_scattering_albedo, dtype=np.float64
        ),
        "asymmetry_factor": np.asarray(cloud.asymmetry_factor, dtype=np.float64),
    }
    digest = hashlib.sha256()
    summary: dict[str, Any] = {
        "shape": list(arrays["extinction_tau"].shape),
        "finite": all(bool(np.all(np.isfinite(array))) for array in arrays.values()),
        "metrics": {},
    }
    for name, array in arrays.items():
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
        summary["metrics"][name] = {
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
            "mean": float(np.mean(array)),
            "nonzero_count": int(np.count_nonzero(array)),
        }
        if include_arrays:
            summary.setdefault("arrays", {})[name] = array.tolist()
    summary["arrays_sha256"] = digest.hexdigest()
    return summary


def _run(args: argparse.Namespace) -> dict[str, Any]:
    setup_start = time.perf_counter()
    import numpy as np
    import virga.justdoit as virga
    from robert_exoplanets import OpticalConstantsCatalog

    import_seconds = time.perf_counter() - setup_start
    catalog_start = time.perf_counter()
    catalog = OpticalConstantsCatalog(_OPTICAL_CONSTANTS)
    refractive_index = catalog.load(_MATERIAL)
    optical_constants_load_seconds = time.perf_counter() - catalog_start

    mie_directory, mie_report, mie_setup_seconds = _prepare_mie_table(
        virga,
        refractive_index,
        supplied_directory=args.mie_directory,
        work_directory=args.work_directory,
        mie_radii=args.mie_radii,
    )

    fixture_start = time.perf_counter()
    (
        gas_optical_depth,
        virga_model,
        mie_slab_model,
        slab_parameters,
        fixture,
    ) = _build_fixture(
        np,
        layers=args.layers,
        wavelengths=args.wavelengths,
        mie_directory=mie_directory,
        refractive_index=refractive_index,
        slab_quadrature_points=args.slab_quadrature_points,
        skip_mie_slab=args.skip_mie_slab,
    )
    fixture_setup_seconds = time.perf_counter() - fixture_start

    virga_cloud, virga_timing = _timed_runs(
        lambda: virga_model.evaluate(gas_optical_depth, parameters={})[0],
        warmups=args.warmups,
        samples=args.samples,
    )
    results: dict[str, Any] = {
        "virga_eddy_sed": {
            "timing": virga_timing,
            "output": _output_summary(
                virga_cloud,
                include_arrays=args.include_arrays,
            ),
        }
    }
    if mie_slab_model is not None:
        slab_cloud, slab_timing = _timed_runs(
            lambda: mie_slab_model.evaluate(
                gas_optical_depth,
                parameters=slab_parameters,
            )[0],
            warmups=args.warmups,
            samples=args.samples,
        )
        results["robert_mie_full_column_slab"] = {
            "timing": slab_timing,
            "output": _output_summary(
                slab_cloud,
                include_arrays=args.include_arrays,
            ),
        }

    try:
        robert_version = importlib.metadata.version("robert-exoplanets")
    except importlib.metadata.PackageNotFoundError:
        robert_version = "source checkout"
    setup_seconds = {
        "python_imports_seconds": import_seconds,
        "optical_constants_load_seconds": optical_constants_load_seconds,
        "mie_table_setup_seconds": mie_setup_seconds,
        "fixture_setup_seconds": fixture_setup_seconds,
        "total_seconds_before_first_cloud_evaluation": (
            import_seconds
            + optical_constants_load_seconds
            + mie_setup_seconds
            + fixture_setup_seconds
        ),
    }
    return {
        "schema_version": 1,
        "benchmark": "robert_virga_cloud_speed",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": {
            "python": platform.python_version(),
            "robert_exoplanets": robert_version,
            "virga_exo": _version("virga-exo") or _version("virga"),
            "numpy": np.__version__,
            "scipy": _version("scipy"),
            "astropy": _version("astropy"),
        },
        "hardware": _hardware_details(),
        "fixture": {
            "material": _MATERIAL,
            "optical_constants_catalog": str(_OPTICAL_CONSTANTS),
            "optical_constants_source_sha256": refractive_index.metadata.get(
                "source_sha256"
            ),
            "mie_table": mie_report,
            **fixture,
        },
        "setup_seconds": setup_seconds,
        "peak_rss_mib": _peak_rss_mib(),
        "results": results,
    }


def main() -> int:
    args = _parse_args()
    temporary_workspace = None
    if args.mie_directory is None and args.work_directory is None:
        temporary_workspace = tempfile.TemporaryDirectory(
            prefix="robert-virga-benchmark-"
        )
        args.work_directory = Path(temporary_workspace.name)
    try:
        try:
            report = _run(args)
            if temporary_workspace is not None:
                report["fixture"]["mie_table"]["retained_after_run"] = False
                report["fixture"]["mie_table"][
                    "temporary_files_removed_after_report"
                ] = True
            else:
                report["fixture"]["mie_table"]["retained_after_run"] = True
        except (ImportError, ModuleNotFoundError) as exc:
            raise SystemExit(
                "Virga benchmark dependencies are missing. Install virga-exo and run "
                "this script in the robert-exoplanets environment. Details: "
                f"{exc}"
            ) from exc
        except Exception as exc:
            raise SystemExit(
                f"Virga benchmark failed: {type(exc).__name__}: {exc}"
            ) from exc

        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
            sys.stdout.write(f"Wrote benchmark report: {output}\n")
        return 0
    finally:
        if temporary_workspace is not None:
            temporary_workspace.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
