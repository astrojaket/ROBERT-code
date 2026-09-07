"""Validate the Smith et al. WASP-77Ab model against the published NIRSpec data.

This is a deterministic real-data check.  It verifies the public inputs,
converts the Smith planet and star surface-flux spectra to eclipse depth, and
integrates the model over the two published NIRSpec detector grids.  It does
not run a retrieval and it does not claim an independent ROBERT atmosphere
fit.
"""

from __future__ import annotations

import argparse
from hashlib import md5, sha256
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter
from typing import Final, Iterable


THREAD_VARIABLES: Final = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)
for _thread_variable in THREAD_VARIABLES:
    _requested = os.environ.get(_thread_variable, "3")
    try:
        _requested = str(min(3, max(1, int(_requested))))
    except ValueError:
        _requested = "3"
    os.environ[_thread_variable] = _requested
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib")
)

import numpy as np  # noqa: E402
from numpy.typing import ArrayLike, NDArray  # noqa: E402


ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import wasp77ab_target as target  # noqa: E402
from robert_exoplanets import load_august2023_wasp77ab  # noqa: E402


DEFAULT_DATA_ROOT: Final = ROOT / "data" / "jwst_emission_spectra"
DEFAULT_TEMPLATE: Final = (
    ROOT
    / "external_data"
    / "wasp77ab_smith2024"
    / "w77_pre_nirspec_best_fit_R500K_scaled.txt"
)
DEFAULT_REPORT: Final = (
    ROOT
    / "docs"
    / "data"
    / "wasp77ab_smith2024_lrs_validation_20260831.json"
)
DEFAULT_PLOT: Final = (
    ROOT
    / "examples"
    / "outputs"
    / "wasp77ab_smith2024"
    / "nirspec_lrs_validation.png"
)
TEMPLATE_BYTES: Final = 72_672_825
TEMPLATE_MD5: Final = "93841125cab0a2c74ae4730ebf36b7dc"
TEMPLATE_SHA256: Final = (
    "fe1fb7cd37b6dec52173a19ee1fd7ea1529c421946c76d728d0bdcf4285ad34d"
)
EXPECTED_POINT_COUNT: Final = 150
EXPECTED_DETECTOR_NAMES: Final = (
    "nirspec_g395h_nrs1",
    "nirspec_g395h_nrs2",
)
SMITH_REPORTED_BEST_CHI_SQUARE_PER_POINT: Final = 0.75
CHI_SQUARE_PER_POINT_TOLERANCE: Final = 0.05
MAX_MEMORY_BYTES: Final = 2 * 1024**3
DEFAULT_MAX_MEMORY_BYTES: Final = 1900 * 1024**2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--max-memory-gib",
        type=float,
        default=DEFAULT_MAX_MEMORY_BYTES / 1024**3,
        help="Strict process RSS guard. The value must be below 2 GiB.",
    )
    return parser.parse_args()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _thread_values() -> dict[str, int]:
    return {name: int(os.environ[name]) for name in THREAD_VARIABLES}


def _stream_hashes(path: Path) -> tuple[str, str]:
    md5_digest = md5()  # noqa: S324 - required upstream integrity identifier
    sha256_digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5_digest.update(block)
            sha256_digest.update(block)
    return md5_digest.hexdigest(), sha256_digest.hexdigest()


def _load_template(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing Smith et al. best-fit template: {path}. "
            "Run examples/fetch_wasp77ab_smith2024_data.py first."
        )
    if path.stat().st_size != TEMPLATE_BYTES:
        raise ValueError("Smith et al. best-fit template has an unexpected size")
    actual_md5, actual_sha256 = _stream_hashes(path)
    if actual_md5 != TEMPLATE_MD5 or actual_sha256 != TEMPLATE_SHA256:
        raise ValueError("Smith et al. best-fit template checksum mismatch")
    values = np.loadtxt(path, dtype=float)
    if values.ndim != 2 or values.shape != (968_971, 3):
        raise ValueError("Smith et al. best-fit template must have shape (968971, 3)")
    if not np.all(np.isfinite(values)):
        raise ValueError("Smith et al. best-fit template must contain finite values")
    wavelength = values[:, 0]
    if np.any(wavelength <= 0.0) or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("Smith et al. template wavelengths must increase")
    if np.any(values[:, 1:] <= 0.0):
        raise ValueError("Smith et al. planet and star fluxes must be positive")
    radius_ratio_squared = (target.PLANET.radius_m / target.STAR.radius_m) ** 2
    eclipse_depth = radius_ratio_squared * values[:, 1] / values[:, 2]
    return wavelength, eclipse_depth


def top_hat_bin_average(
    source_wavelength: ArrayLike,
    source_values: ArrayLike,
    target_edges: ArrayLike,
) -> NDArray[np.float64]:
    """Integrate a sampled spectrum over explicit contiguous target bins."""

    wavelength = np.asarray(source_wavelength, dtype=float)
    values = np.asarray(source_values, dtype=float)
    edges = np.asarray(target_edges, dtype=float)
    if wavelength.ndim != 1 or values.shape != wavelength.shape:
        raise ValueError("source wavelength and values must be matching 1-D arrays")
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("target edges must contain at least one bin")
    if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(values)):
        raise ValueError("source arrays must be finite")
    if not np.all(np.diff(wavelength) > 0.0) or not np.all(np.diff(edges) > 0.0):
        raise ValueError("source wavelength and target edges must increase")
    if edges[0] < wavelength[0] or edges[-1] > wavelength[-1]:
        raise ValueError("target bins must lie inside the source wavelength coverage")

    output = np.empty(edges.size - 1, dtype=float)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        left = int(np.searchsorted(wavelength, lower, side="right"))
        right = int(np.searchsorted(wavelength, upper, side="left"))
        integration_wavelength = np.concatenate(
            ((lower,), wavelength[left:right], (upper,))
        )
        integration_values = np.concatenate(
            (
                (float(np.interp(lower, wavelength, values)),),
                values[left:right],
                (float(np.interp(upper, wavelength, values)),),
            )
        )
        output[index] = np.trapezoid(
            integration_values, integration_wavelength
        ) / (upper - lower)
    if not np.all(np.isfinite(output)):
        raise ValueError("top-hat integration produced non-finite values")
    output.setflags(write=False)
    return output


def _csv_metadata_array(metadata: object, key: str) -> NDArray[np.float64]:
    mapping = getattr(metadata, "get", None)
    if not callable(mapping):
        raise ValueError("observation metadata must be a mapping")
    values = np.fromstring(str(mapping(key, "")), sep=",")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"observation metadata is missing {key}")
    return values


def _chi_square_metrics(
    model: NDArray[np.float64], observation: object
) -> dict[str, float | int]:
    data = np.asarray(getattr(observation, "flux"), dtype=float)
    uncertainty = np.asarray(getattr(observation, "uncertainty"), dtype=float)
    if model.shape != data.shape or uncertainty.shape != data.shape:
        raise ValueError("model, data, and uncertainty must have matching shapes")
    residual = model - data
    symmetric = float(np.sum(np.square(residual / uncertainty)))

    error_1 = 0.01 * np.abs(
        _csv_metadata_array(getattr(observation, "metadata"), "published_error_1_percent")
    )
    error_2 = 0.01 * np.abs(
        _csv_metadata_array(getattr(observation, "metadata"), "published_error_2_percent")
    )
    if error_1.shape != data.shape or error_2.shape != data.shape:
        raise ValueError("published asymmetric errors must match the observation")
    split_1_upper = np.where(residual >= 0.0, error_1, error_2)
    split_2_upper = np.where(residual >= 0.0, error_2, error_1)
    return {
        "n_points": int(data.size),
        "chi_square_symmetric": symmetric,
        "chi_square_per_point_symmetric": symmetric / data.size,
        "chi_square_split_error1_upper": float(
            np.sum(np.square(residual / split_1_upper))
        ),
        "chi_square_split_error2_upper": float(
            np.sum(np.square(residual / split_2_upper))
        ),
        "mean_residual": float(np.mean(residual)),
        "rms_residual": float(np.sqrt(np.mean(np.square(residual)))),
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
    }


def _write_plot(
    path: Path,
    wavelength: NDArray[np.float64],
    eclipse_depth: NDArray[np.float64],
    detector_records: Iterable[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, (spectrum_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(8.4, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
        constrained_layout=True,
    )
    spectrum_axis.plot(
        wavelength[::100],
        1.0e6 * eclipse_depth[::100],
        color="0.65",
        linewidth=0.7,
        label="Smith joint best fit (R=500,000)",
    )
    colors = ("#2f6f9f", "#d98324")
    for color, record in zip(colors, detector_records, strict=True):
        observed_wavelength = np.asarray(record["wavelength"], dtype=float)
        data = np.asarray(record["data"], dtype=float)
        uncertainty = np.asarray(record["uncertainty"], dtype=float)
        model = np.asarray(record["model"], dtype=float)
        label = str(record["name"]).replace("nirspec_g395h_", "").upper()
        spectrum_axis.errorbar(
            observed_wavelength,
            1.0e6 * data,
            yerr=1.0e6 * uncertainty,
            fmt="o",
            markersize=2.8,
            linewidth=0.7,
            color=color,
            label=label,
        )
        spectrum_axis.plot(observed_wavelength, 1.0e6 * model, color=color, linewidth=1.0)
        residual_axis.errorbar(
            observed_wavelength,
            1.0e6 * (data - model),
            yerr=1.0e6 * uncertainty,
            fmt="o",
            markersize=2.5,
            linewidth=0.7,
            color=color,
        )
    spectrum_axis.set_ylabel("Eclipse depth (ppm)")
    spectrum_axis.legend(frameon=False, ncol=3, fontsize=8)
    spectrum_axis.set_title("WASP-77Ab: Smith et al. model and August et al. NIRSpec data")
    residual_axis.axhline(0.0, color="0.3", linewidth=0.8)
    residual_axis.set_xlabel("Wavelength (micron)")
    residual_axis.set_ylabel("Data - model\n(ppm)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_validation(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    template_path: Path = DEFAULT_TEMPLATE,
    report_path: Path = DEFAULT_REPORT,
    plot_path: Path | None = DEFAULT_PLOT,
    max_memory_bytes: int = DEFAULT_MAX_MEMORY_BYTES,
) -> dict[str, object]:
    """Run the deterministic public-data check and write its compact report."""

    if not 0 < int(max_memory_bytes) < MAX_MEMORY_BYTES:
        raise ValueError("max_memory_bytes must be positive and strictly below 2 GiB")
    started = perf_counter()
    collection = load_august2023_wasp77ab(data_root)
    if collection.names != EXPECTED_DETECTOR_NAMES or collection.n_points != 150:
        raise ValueError("unexpected WASP-77Ab NIRSpec detector collection")
    wavelength, eclipse_depth = _load_template(template_path)

    records: list[dict[str, object]] = []
    total_symmetric = 0.0
    total_split_1 = 0.0
    total_split_2 = 0.0
    for dataset in collection.datasets:
        observation = dataset.observation
        model = top_hat_bin_average(
            wavelength,
            eclipse_depth,
            observation.wavelength_bin_edges,
        )
        metrics = _chi_square_metrics(model, observation)
        total_symmetric += float(metrics["chi_square_symmetric"])
        total_split_1 += float(metrics["chi_square_split_error1_upper"])
        total_split_2 += float(metrics["chi_square_split_error2_upper"])
        records.append(
            {
                "name": dataset.name,
                "wavelength": np.asarray(observation.wavelength, dtype=float),
                "data": np.asarray(observation.flux, dtype=float),
                "uncertainty": np.asarray(observation.uncertainty, dtype=float),
                "model": model,
                "wavelength_min_micron": float(observation.wavelength[0]),
                "wavelength_max_micron": float(observation.wavelength[-1]),
                **metrics,
            }
        )

    chi_square_per_point = total_symmetric / EXPECTED_POINT_COUNT
    peak_rss = _peak_rss_bytes()
    if peak_rss >= max_memory_bytes:
        raise MemoryError("WASP-77Ab LRS validation exceeded its process RSS guard")
    thread_values = _thread_values()
    if any(not 1 <= value <= 3 for value in thread_values.values()):
        raise RuntimeError("WASP-77Ab LRS validation exceeded the thread limit")
    smith_consistency = (
        abs(chi_square_per_point - SMITH_REPORTED_BEST_CHI_SQUARE_PER_POINT)
        <= CHI_SQUARE_PER_POINT_TOLERANCE
    )

    if plot_path is not None:
        _write_plot(plot_path, wavelength, eclipse_depth, records)
        plot_sha256 = _stream_hashes(plot_path)[1]
    else:
        plot_sha256 = None

    json_records = []
    for record in records:
        json_records.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"wavelength", "data", "uncertainty", "model"}
            }
        )
    report: dict[str, object] = {
        "schema_version": "1.0",
        "status": "pass" if smith_consistency else "fail",
        "scope": "deterministic Smith et al. model versus August et al. NIRSpec data",
        "claim": "public-data and bin-integration reproduction; not a ROBERT atmosphere retrieval",
        "data": {
            "producer": "August et al. 2023",
            "comparison_study": "Smith et al. 2024",
            "table_points": EXPECTED_POINT_COUNT,
            "detectors": list(EXPECTED_DETECTOR_NAMES),
            "table_sha256": collection.metadata["checksum_sha256"],
            "published_count": 150,
            "smith_predictive_text_count": 160,
            "count_interpretation": (
                "the public 150-bin table reproduces the reported fit; "
                "N=160 is treated as a manuscript count discrepancy"
            ),
        },
        "template": {
            "path": str(template_path),
            "bytes": TEMPLATE_BYTES,
            "md5": TEMPLATE_MD5,
            "sha256": TEMPLATE_SHA256,
            "shape": [968_971, 3],
            "columns": ["wavelength_micron", "planet_surface_flux", "star_surface_flux"],
            "radius_ratio_squared": (target.PLANET.radius_m / target.STAR.radius_m) ** 2,
            "eclipse_depth_formula": "(Rp/Rs)^2 * planet_surface_flux / star_surface_flux",
        },
        "binning": {
            "method": "top-hat wavelength integral over each published bin",
            "detector_gap_integrated": False,
            "detector_gap_micron": 0.119,
        },
        "metrics": {
            "n_points": EXPECTED_POINT_COUNT,
            "chi_square_symmetric": total_symmetric,
            "chi_square_per_point_symmetric": chi_square_per_point,
            "chi_square_split_error1_upper": total_split_1,
            "chi_square_split_error2_upper": total_split_2,
            "smith_reported_best_chi_square_per_point": (
                SMITH_REPORTED_BEST_CHI_SQUARE_PER_POINT
            ),
            "absolute_difference_from_smith": abs(
                chi_square_per_point - SMITH_REPORTED_BEST_CHI_SQUARE_PER_POINT
            ),
            "acceptance_tolerance": CHI_SQUARE_PER_POINT_TOLERANCE,
            "smith_consistency_pass": smith_consistency,
            "detector_metrics": json_records,
        },
        "uncertainty_policy": {
            "primary": "mean absolute asymmetric errors, matching ROBERT Gaussian input",
            "sensitivity": [
                "published error 1 used above the data and error 2 below",
                "published error 2 used above the data and error 1 below",
            ],
        },
        "resources": {
            "thread_values": thread_values,
            "thread_limit": 3,
            "peak_rss_bytes": peak_rss,
            "process_rss_limit_bytes": int(max_memory_bytes),
            "elapsed_seconds": perf_counter() - started,
        },
        "sampler": "none",
        "abundance_state": "not evaluated; ROBERT remains VMR-only",
        "plot": None
        if plot_path is None
        else {"path": str(plot_path), "sha256": plot_sha256},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = _parse_args()
    max_memory_bytes = int(args.max_memory_gib * 1024**3)
    report = run_validation(
        data_root=args.data_root,
        template_path=args.template,
        report_path=args.report,
        plot_path=None if args.no_plot else args.plot,
        max_memory_bytes=max_memory_bytes,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if report["status"] != "pass":
        raise SystemExit("WASP-77Ab NIRSpec validation failed")


if __name__ == "__main__":
    main()
