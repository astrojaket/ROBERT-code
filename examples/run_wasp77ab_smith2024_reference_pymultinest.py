"""Run bounded PyMultiNest reference fits for the Smith WASP-77Ab data.

The workflow is deliberately a fixed-template comparison.  It builds three
sampler-compatible problems from the strict Smith et al. (2024) IGRINS cube,
the public August et al. (2023) NIRSpec table, and the official Smith
R=500,000 template:

* HRS-only: ``Kp``, ``dVsys``, and the HRS multiplicative scale;
* LRS-only: one explicit multiplicative NIRSpec-template scale;
* joint: the HRS parameters plus the shared LRS scale.

The HRS statistic is the prepared PCA likelihood.  Its observation and
projection remain inside the prepared likelihood.  The two NIRSpec detector
terms use explicit ``Observation`` objects and independent Gaussian errors.
The HRS template is cropped for the full velocity-prior and response-support
range, then convolved with a Gray 4.2 km/s rotation kernel and a Gaussian
R=45,000 LSF before the likelihood interpolates it to detector pixels.  The
LRS path keeps the original template and uses top-hat bin averages.
This module does not convert abundances, run an atmosphere model, or claim a
truth-recovery or Smith-method reproduction result.

No sampler is started unless ``--run-pymultinest`` is supplied.  A normal
invocation is therefore a dry run.  PyMultiNest with the MultiNest backend is
the only supported sampler; UltraNest is not imported or used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import md5, sha256
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter
from types import MappingProxyType
from typing import Final, Mapping, Sequence


ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
THREAD_VARIABLES: Final = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)
MAX_THREADS: Final = 3


def _clamp_thread_environment() -> None:
    """Clamp numerical thread settings before importing NumPy."""

    for name in THREAD_VARIABLES:
        raw = os.environ.get(name, str(MAX_THREADS))
        try:
            value = min(MAX_THREADS, max(1, int(raw)))
        except (TypeError, ValueError):
            value = MAX_THREADS
        os.environ[name] = str(value)


_clamp_thread_environment()
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "robert-matplotlib"),
)

import numpy as np  # noqa: E402
from numpy.typing import ArrayLike, NDArray  # noqa: E402

from examples import wasp77ab_target as target  # noqa: E402
from robert_exoplanets.core import (  # noqa: E402
    RobertCoverageError,
    RobertValidationError,
    SpectralGrid,
    Spectrum,
)
from robert_exoplanets.instruments import (  # noqa: E402
    GaussianHighResolutionResponse,
    HighResolutionEmissionTemplate,
    ObservationCollection,
    RotationalBroadeningResponse,
    TimeResolvedHighResolutionObservation,
)
from robert_exoplanets.io import (  # noqa: E402
    SMITH2024_WASP77AB_CUBE14_MD5,
    SMITH2024_WASP77AB_CUBE14_SHA256,
    SMITH2024_WASP77AB_CUBE14_SIZE,
    SMITH2024_WASP77AB_INFO_MD5,
    SMITH2024_WASP77AB_INFO_SHA256,
    SMITH2024_WASP77AB_INFO_SIZE,
    SMITH2024_WASP77AB_TEMPLATE_R500K_MD5,
    SMITH2024_WASP77AB_TEMPLATE_R500K_SHA256,
    SMITH2024_WASP77AB_TEMPLATE_R500K_SIZE,
    load_august2023_wasp77ab,
    load_smith2024_wasp77ab_hrs,
    load_smith2024_wasp77ab_template,
)
from robert_exoplanets.likelihoods import (  # noqa: E402
    GaussianLikelihood,
    TimeResolvedHighResolutionLikelihood,
)
from robert_exoplanets.retrieval import (  # noqa: E402
    HeterogeneousLikelihood,
    HeterogeneousLikelihoodComponent,
    HeterogeneousRetrievalProblem,
    RetrievalParameter,
    RetrievalParameterSet,
    UniformPrior,
)


DEFAULT_SMITH_DATA_ROOT: Final = ROOT / "external_data" / "wasp77ab_smith2024"
DEFAULT_NIRSPEC_DATA_ROOT: Final = target.DATA_DIRECTORY
DEFAULT_TEMPLATE_PATH: Final = (
    DEFAULT_SMITH_DATA_ROOT / "w77_pre_nirspec_best_fit_R500K_scaled.txt"
)
DEFAULT_OUTPUT_ROOT: Final = ROOT / "examples" / "outputs" / "wasp77ab_smith2024"
DEFAULT_REPORT_PATH: Final = (
    ROOT / "docs" / "data" / "wasp77ab_smith2024_reference_pymultinest.json"
)
DEFAULT_PLOT_PATH: Final = DEFAULT_OUTPUT_ROOT / "reference_lrs.png"

HRS_MODE: Final = "hrs_only"
LRS_MODE: Final = "lrs_only"
JOINT_MODE: Final = "joint"
ALL_MODES: Final = (HRS_MODE, LRS_MODE, JOINT_MODE)
SEEDS: Final[tuple[int, int]] = (24680, 24681)
N_LIVE_POINTS: Final = 64
MAX_ITER: Final = 0
EVIDENCE_TOLERANCE: Final = 0.5
SAMPLING_EFFICIENCY: Final = 0.8
MPI_PROCESSES: Final = 1
PROCESS_RSS_LIMIT_BYTES: Final = 2 * 1024**3
DEFAULT_PROCESS_RSS_LIMIT_BYTES: Final = int(1.9 * 1024**3)
HRS_ROTATION_VSIN_KM_S: Final = 4.2
HRS_ROTATION_LIMB_DARKENING: Final = 0.6
HRS_GAUSSIAN_RESOLVING_POWER: Final = 45_000.0
HRS_GAUSSIAN_KERNEL_SUPPORT_SIGMA: Final = 4.0
SPEED_OF_LIGHT_KM_S: Final = 299_792.458

HRS_KP_PARAMETER: Final = "Kp"
HRS_DVSYS_PARAMETER: Final = "dVsys"
HRS_SCALE_PARAMETER: Final = "log10_a_hrs"
LRS_SCALE_PARAMETER: Final = "lrs_template_scale"
HRS_KP_PRIOR: Final = UniformPrior(172.0, 212.0)
HRS_DVSYS_PRIOR: Final = UniformPrior(-20.0, 20.0)
HRS_SCALE_PRIOR: Final = UniformPrior(-2.0, 2.0)
# A dimensionless multiplicative calibration nuisance.  The interval is
# deliberately bounded around the fixed Smith template normalization.
LRS_SCALE_PRIOR: Final = UniformPrior(0.5, 1.5)
LRS_SCALE_PRIOR_DESCRIPTION: Final = "Uniform(0.5, 1.5), dimensionless template scale"
INVALID_LOGLIKE_FLOOR: Final = -1.0e100
MAX_HASH_CHUNK_BYTES: Final = 8 * 1024**2


@dataclass(frozen=True)
class ReferenceInputs:
    """Strictly loaded fixed inputs shared by all three reference problems."""

    hrs_observation: TimeResolvedHighResolutionObservation
    nirspec_observations: ObservationCollection
    template: HighResolutionEmissionTemplate
    hrs_template: HighResolutionEmissionTemplate
    lrs_template_spectra: Mapping[str, Spectrum]
    source_hashes: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        if not isinstance(self.hrs_observation, TimeResolvedHighResolutionObservation):
            raise RobertValidationError("hrs_observation has an unexpected type")
        if not isinstance(self.nirspec_observations, ObservationCollection):
            raise RobertValidationError("nirspec_observations has an unexpected type")
        if not isinstance(self.template, HighResolutionEmissionTemplate):
            raise RobertValidationError("template has an unexpected type")
        if not isinstance(self.hrs_template, HighResolutionEmissionTemplate):
            raise RobertValidationError("hrs_template has an unexpected type")
        if self.hrs_template.metadata.get("hrs_response_stage_order") != (
            "gray_rotation_then_gaussian_lsf"
        ):
            raise RobertValidationError(
                "hrs_template must be prepared with the Smith rotation and LSF response"
            )
        if not self.lrs_template_spectra:
            raise RobertValidationError("at least one LRS template spectrum is required")
        if not self.source_hashes:
            raise RobertValidationError("source hashes must not be empty")
        object.__setattr__(
            self,
            "lrs_template_spectra",
            MappingProxyType(dict(self.lrs_template_spectra)),
        )
        object.__setattr__(
            self,
            "source_hashes",
            MappingProxyType(
                {
                    str(name): MappingProxyType(dict(record))
                    for name, record in self.source_hashes.items()
                }
            ),
        )


def _peak_rss_bytes() -> int:
    """Return peak resident memory in bytes on the current platform."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _thread_values() -> dict[str, int]:
    """Return the clamped values of all numerical thread controls."""

    return {name: int(os.environ[name]) for name in THREAD_VARIABLES}


def _mpi_world_size() -> int:
    """Return the active MPI world size without requiring mpi4py."""

    try:
        from mpi4py import MPI
    except ImportError:
        return 1
    return int(MPI.COMM_WORLD.Get_size())


def resource_record(
    *,
    max_memory_bytes: int = DEFAULT_PROCESS_RSS_LIMIT_BYTES,
    check_peak: bool = True,
) -> dict[str, object]:
    """Record and optionally enforce the one-MPI/three-thread memory gate."""

    limit = int(max_memory_bytes)
    if limit <= 0 or limit >= PROCESS_RSS_LIMIT_BYTES:
        raise ValueError("max_memory_bytes must be positive and strictly below 2 GiB")
    threads = _thread_values()
    mpi_world_size = _mpi_world_size() if check_peak else MPI_PROCESSES
    peak = _peak_rss_bytes() if check_peak else None
    threads_pass = all(1 <= value <= MAX_THREADS for value in threads.values())
    mpi_pass = mpi_world_size == MPI_PROCESSES
    memory_pass = peak is None or peak < limit
    return {
        "thread_values": threads,
        "thread_limit": MAX_THREADS,
        "mpi_processes_configured": MPI_PROCESSES,
        "mpi_world_size_observed": mpi_world_size,
        "mpi_pass": mpi_pass,
        "peak_rss_bytes": peak,
        "configured_process_rss_limit_bytes": limit,
        "process_rss_hard_limit_bytes": PROCESS_RSS_LIMIT_BYTES,
        "process_rss_pass": memory_pass,
        "threads_pass": threads_pass,
        "gate_passed": bool(mpi_pass and threads_pass and memory_pass),
    }


def _stream_hashes(path: Path) -> tuple[int, str, str]:
    """Compute size, MD5, and SHA-256 with a bounded read buffer."""

    if not path.is_file():
        raise FileNotFoundError(path)
    digest_md5 = md5()
    digest_sha256 = sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for block in iter(
                lambda: stream.read(MAX_HASH_CHUNK_BYTES),
                b"",
            ):
                size += len(block)
                digest_md5.update(block)
                digest_sha256.update(block)
    except OSError as error:
        raise OSError(f"could not hash source file {path}") from error
    return size, digest_md5.hexdigest(), digest_sha256.hexdigest()


def _verified_source_record(
    *,
    name: str,
    path: Path,
    role: str,
    url: str | None,
    expected_size: int | None = None,
    expected_md5: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Hash one external input and fail closed on identity mismatch."""

    size, actual_md5, actual_sha256 = _stream_hashes(path)
    if expected_size is not None and size != int(expected_size):
        raise ValueError(
            f"{name} size mismatch: expected {expected_size}, got {size}"
        )
    if expected_md5 is not None and actual_md5 != str(expected_md5):
        raise ValueError(f"{name} MD5 mismatch")
    if expected_sha256 is not None and actual_sha256 != str(expected_sha256):
        raise ValueError(f"{name} SHA-256 mismatch")
    return {
        "name": name,
        "role": role,
        "url": url,
        "path": str(path),
        "size_bytes": size,
        "md5": actual_md5,
        "sha256": actual_sha256,
        "expected_size_bytes": expected_size,
        "expected_md5": expected_md5,
        "expected_sha256": expected_sha256,
        "verified": True,
    }


def _expected_file(name: str):
    try:
        return target.SMITH_ZENODO_FILES[name]
    except KeyError as error:
        raise RobertValidationError(f"unknown Smith source file: {name}") from error


def _preflight_nirspec(
    nirspec_data_root: Path,
) -> tuple[ObservationCollection, Path]:
    """Load the small NIRSpec table and verify its reconciled 150-row shape."""

    observations = load_august2023_wasp77ab(
        nirspec_data_root,
        verify_checksum=True,
    )
    if observations.n_points != target.NIRSPEC_PUBLISHED_TABLE_POINTS:
        raise ValueError("NIRSpec source must contain the reconciled 150 points")
    if observations.names != (
        "nirspec_g395h_nrs1",
        "nirspec_g395h_nrs2",
    ):
        raise ValueError("NIRSpec source must retain NRS1 and NRS2 as separate terms")
    source_path = Path(
        str(observations.datasets[0].observation.metadata["source_path"])
    )
    return observations, source_path


def preflight_inputs(
    *,
    smith_data_root: Path = DEFAULT_SMITH_DATA_ROOT,
    nirspec_data_root: Path = DEFAULT_NIRSPEC_DATA_ROOT,
    template_path: Path = DEFAULT_TEMPLATE_PATH,
) -> dict[str, Mapping[str, object]]:
    """Verify all external identities without unpickling the HRS cube."""

    smith_root = Path(smith_data_root).expanduser()
    nirspec_root = Path(nirspec_data_root).expanduser()
    _, nirspec_path = _preflight_nirspec(nirspec_root)
    cube_spec = _expected_file("cube14v3.pic")
    info_spec = _expected_file("20201214_info.csv")
    template_spec = _expected_file("w77_pre_nirspec_best_fit_R500K_scaled.txt")
    records = {
        "hrs_cube": _verified_source_record(
            name="cube14v3.pic",
            path=smith_root / "cube14v3.pic",
            role="Smith et al. (2024) primary pre-eclipse IGRINS cube",
            url=cube_spec.url,
            expected_size=SMITH2024_WASP77AB_CUBE14_SIZE,
            expected_md5=SMITH2024_WASP77AB_CUBE14_MD5,
            expected_sha256=SMITH2024_WASP77AB_CUBE14_SHA256,
        ),
        "hrs_metadata": _verified_source_record(
            name="20201214_info.csv",
            path=smith_root / "20201214_info.csv",
            role="Smith et al. (2024) primary pre-eclipse frame metadata",
            url=info_spec.url,
            expected_size=SMITH2024_WASP77AB_INFO_SIZE,
            expected_md5=SMITH2024_WASP77AB_INFO_MD5,
            expected_sha256=SMITH2024_WASP77AB_INFO_SHA256,
        ),
        "official_template": _verified_source_record(
            name=template_spec.name,
            path=Path(template_path).expanduser(),
            role="Smith et al. (2024) official R=500,000 NIRSpec template",
            url=template_spec.url,
            expected_size=SMITH2024_WASP77AB_TEMPLATE_R500K_SIZE,
            expected_md5=SMITH2024_WASP77AB_TEMPLATE_R500K_MD5,
            expected_sha256=SMITH2024_WASP77AB_TEMPLATE_R500K_SHA256,
        ),
    }
    nirspec_spec = target.NIRSPEC_TABLE
    records["nirspec_table"] = _verified_source_record(
        name=nirspec_spec.name,
        path=nirspec_path,
        role="August et al. (2023) / Smith et al. (2024) NIRSpec comparison table",
        url=nirspec_spec.url,
        expected_size=nirspec_spec.size_bytes,
        expected_sha256=nirspec_spec.sha256,
    )
    return records


def top_hat_bin_average(
    source_wavelength: ArrayLike,
    source_values: ArrayLike,
    target_edges: ArrayLike,
) -> NDArray[np.float64]:
    """Integrate a sampled template over one-dimensional target bins."""

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
        raise ValueError("target bins must lie inside source template coverage")

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
            integration_values,
            integration_wavelength,
        ) / (upper - lower)
    if not np.all(np.isfinite(output)):
        raise ValueError("template binning produced non-finite values")
    output.setflags(write=False)
    return output


def _hrs_velocity_bounds(
    observation: TimeResolvedHighResolutionObservation,
) -> tuple[float, float]:
    """Return extrema over the HRS Kp and dVsys prior rectangle."""

    phase_sine = np.sin(2.0 * np.pi * np.asarray(observation.phase, dtype=float))
    kp_lower = float(HRS_KP_PRIOR.lower)
    kp_upper = float(HRS_KP_PRIOR.upper)
    dVsys_lower = float(HRS_DVSYS_PRIOR.lower)
    dVsys_upper = float(HRS_DVSYS_PRIOR.upper)
    fixed = np.asarray(observation.fixed_velocity_km_s, dtype=float)
    kp_for_lower = np.where(phase_sine >= 0.0, kp_lower, kp_upper)
    kp_for_upper = np.where(phase_sine >= 0.0, kp_upper, kp_lower)
    lower = fixed + dVsys_lower + kp_for_lower * phase_sine
    upper = fixed + dVsys_upper + kp_for_upper * phase_sine
    lower_bound = float(np.min(lower))
    upper_bound = float(np.max(upper))
    if (
        not np.isfinite(lower_bound)
        or not np.isfinite(upper_bound)
        or abs(lower_bound) >= SPEED_OF_LIGHT_KM_S
        or abs(upper_bound) >= SPEED_OF_LIGHT_KM_S
    ):
        raise RobertValidationError(
            "HRS velocity prior extrema must be finite and below the speed of light"
        )
    return lower_bound, upper_bound


def _hrs_query_wavelength_bounds(
    observation: TimeResolvedHighResolutionObservation,
) -> tuple[float, float]:
    """Return rest-grid bounds sampled by all allowed HRS velocities."""

    velocity_lower, velocity_upper = _hrs_velocity_bounds(observation)
    wavelengths = np.asarray(observation.order_wavelengths, dtype=float)
    sampled = np.concatenate(
        (
            wavelengths * (1.0 - velocity_lower / SPEED_OF_LIGHT_KM_S),
            wavelengths * (1.0 - velocity_upper / SPEED_OF_LIGHT_KM_S),
        )
    )
    lower = float(np.min(sampled))
    upper = float(np.max(sampled))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0:
        raise RobertValidationError("HRS sampled template bounds must be finite and positive")
    return lower, upper


def _prepare_hrs_template(
    template: HighResolutionEmissionTemplate,
    observation: TimeResolvedHighResolutionObservation,
) -> HighResolutionEmissionTemplate:
    """Apply Smith's rotation and LSF to a bounded template grid.

    The temporal PCA likelihood interpolates the convolved model onto each
    detector wavelength.  Therefore this preparation intentionally does not
    pixel-integrate.  The crop includes the complete prior velocity range and
    the support of both response kernels, so every allowed trial remains
    interpolation-only and non-extrapolating.
    """

    if not isinstance(template, HighResolutionEmissionTemplate):
        raise RobertValidationError("template must be a HighResolutionEmissionTemplate")
    if not isinstance(observation, TimeResolvedHighResolutionObservation):
        raise RobertValidationError(
            "observation must be a TimeResolvedHighResolutionObservation"
        )
    if template.wavelength_unit != observation.wavelength_unit:
        raise RobertValidationError("template and observation wavelength units must match")

    source_wavelength = np.asarray(template.wavelength, dtype=float)
    source_planet = np.asarray(template.planet_flux, dtype=float)
    source_stellar = np.asarray(template.stellar_flux, dtype=float)
    if source_wavelength[0] > source_wavelength[-1]:
        source_wavelength = source_wavelength[::-1]
        source_planet = source_planet[::-1]
        source_stellar = source_stellar[::-1]

    query_lower, query_upper = _hrs_query_wavelength_bounds(observation)
    gaussian_velocity_support = (
        HRS_GAUSSIAN_KERNEL_SUPPORT_SIGMA
        * SPEED_OF_LIGHT_KM_S
        / (2.0 * np.sqrt(2.0 * np.log(2.0)) * HRS_GAUSSIAN_RESOLVING_POWER)
    )
    response_velocity_support = HRS_ROTATION_VSIN_KM_S + gaussian_velocity_support
    response_factor = float(np.exp(response_velocity_support / SPEED_OF_LIGHT_KM_S))
    crop_lower = query_lower / response_factor
    crop_upper = query_upper * response_factor
    if source_wavelength[0] > crop_lower or source_wavelength[-1] < crop_upper:
        raise RobertCoverageError(
            "official HRS template does not cover velocity and response-support margins"
        )

    # Include two source samples outside each requested bound.  This protects
    # the final trimmed response grid from endpoint round-off at R=500,000.
    lower_index = max(
        0,
        int(np.searchsorted(source_wavelength, crop_lower, side="left")) - 2,
    )
    upper_index = min(
        source_wavelength.size,
        int(np.searchsorted(source_wavelength, crop_upper, side="right")) + 2,
    )
    if upper_index - lower_index < 2:
        raise RobertCoverageError("HRS template crop must contain at least two samples")
    cropped_wavelength = source_wavelength[lower_index:upper_index]
    cropped_planet = source_planet[lower_index:upper_index]
    cropped_stellar = source_stellar[lower_index:upper_index]
    source_grid = SpectralGrid(
        values=cropped_wavelength,
        unit=template.wavelength_unit,
        name=f"{template.name}-hrs-source-crop",
        role="high_resolution_native",
        metadata={
            "crop_lower": f"{float(cropped_wavelength[0]):.17g}",
            "crop_upper": f"{float(cropped_wavelength[-1]):.17g}",
        },
    )
    planet_spectrum = Spectrum(
        spectral_grid=source_grid,
        values=cropped_planet,
        unit="surface_flux",
        observable="surface_flux",
    )
    stellar_spectrum = Spectrum(
        spectral_grid=source_grid,
        values=cropped_stellar,
        unit="surface_flux",
        observable="surface_flux",
    )

    rotation = RotationalBroadeningResponse(
        projected_velocity_km_s=HRS_ROTATION_VSIN_KM_S,
        limb_darkening=HRS_ROTATION_LIMB_DARKENING,
    ).prepare_on_grid(source_grid)
    rotated_planet = rotation.observe(planet_spectrum)
    rotated_stellar = rotation.observe(stellar_spectrum)
    gaussian = GaussianHighResolutionResponse(
        resolving_power=HRS_GAUSSIAN_RESOLVING_POWER,
        kernel_support=HRS_GAUSSIAN_KERNEL_SUPPORT_SIGMA,
    ).prepare_on_grid(rotated_planet.spectral_grid)
    convolved_planet = gaussian.observe(rotated_planet)
    convolved_stellar = gaussian.observe(rotated_stellar)
    output_wavelength = np.asarray(convolved_planet.spectral_grid.values, dtype=float)
    output_lower = float(output_wavelength[0])
    output_upper = float(output_wavelength[-1])
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, query_upper)
    if output_lower > query_lower + tolerance or output_upper < query_upper - tolerance:
        raise RobertCoverageError(
            "convolved HRS template does not cover all allowed Doppler samples"
        )
    metadata = dict(template.metadata)
    metadata.update(
        {
            "hrs_response_stage_order": "gray_rotation_then_gaussian_lsf",
            "hrs_rotation_vsin_km_s": f"{HRS_ROTATION_VSIN_KM_S:g}",
            "hrs_rotation_limb_darkening": f"{HRS_ROTATION_LIMB_DARKENING:g}",
            "hrs_gaussian_resolving_power": f"{HRS_GAUSSIAN_RESOLVING_POWER:g}",
            "hrs_gaussian_kernel_support_sigma": f"{HRS_GAUSSIAN_KERNEL_SUPPORT_SIGMA:g}",
            "hrs_input_resolution": "R=500,000",
            "hrs_smith_model_resolution": (
                "Smith used R=250,000; this public template carrier is R=500,000"
            ),
            "hrs_response_output_resolution": "approximately R=45,000 plus rotation",
            "hrs_final_mapping": "linear interpolation by TimeResolvedHighResolutionLikelihood",
            "hrs_pixel_integration": "not applied; Smith interpolates the convolved model",
            "hrs_velocity_bounds_km_s": (
                f"{_hrs_velocity_bounds(observation)[0]:.17g},"
                f"{_hrs_velocity_bounds(observation)[1]:.17g}"
            ),
            "hrs_query_bounds_micron": f"{query_lower:.17g},{query_upper:.17g}",
            "hrs_crop_bounds_micron": (
                f"{float(cropped_wavelength[0]):.17g},"
                f"{float(cropped_wavelength[-1]):.17g}"
            ),
            "hrs_response_output_bounds_micron": f"{output_lower:.17g},{output_upper:.17g}",
            "hrs_source_points_after_crop": str(int(cropped_wavelength.size)),
        }
    )
    return HighResolutionEmissionTemplate(
        wavelength=output_wavelength,
        planet_flux=convolved_planet.values,
        stellar_flux=convolved_stellar.values,
        wavelength_unit=template.wavelength_unit,
        name=f"{template.name}-hrs-rot4p2-lsfR45000",
        flux_ratio_scale=template.flux_ratio_scale,
        metadata=metadata,
    )


def _lrs_template_spectra(
    template: HighResolutionEmissionTemplate,
    observations: ObservationCollection,
) -> dict[str, Spectrum]:
    """Build fixed eclipse-depth spectra on the two NIRSpec detector grids."""

    wavelength = np.asarray(getattr(template, "wavelength"), dtype=float)
    values = np.asarray(getattr(template, "scaled_flux_ratio"), dtype=float)
    output: dict[str, Spectrum] = {}
    for dataset in observations.datasets:
        observation = dataset.observation
        binned = top_hat_bin_average(
            wavelength,
            values,
            observation.wavelength_bin_edges,
        )
        output[dataset.name] = Spectrum.from_arrays(
            observation.wavelength,
            binned,
            unit=observation.flux_unit,
            observable=observation.observable,
            wavelength_unit=observation.wavelength_unit,
        )
    return output


def _loaded_source_hashes(
    *,
    smith_data_root: Path,
    nirspec_observations: ObservationCollection,
    template_path: Path,
    hrs_observation: TimeResolvedHighResolutionObservation,
    template: HighResolutionEmissionTemplate,
) -> dict[str, Mapping[str, object]]:
    """Return verified source identities reported by the strict loaders."""

    cube_spec = _expected_file("cube14v3.pic")
    info_spec = _expected_file("20201214_info.csv")
    template_spec = _expected_file("w77_pre_nirspec_best_fit_R500K_scaled.txt")
    hrs_metadata = hrs_observation.metadata
    nirspec_metadata = nirspec_observations.datasets[0].observation.metadata
    template_metadata = getattr(template, "metadata", {})

    def loader_record(
        *,
        name: str,
        path: Path,
        role: str,
        url: str | None,
        size: object,
        actual_md5: object,
        actual_sha256: object,
        expected_size: int,
        expected_md5: str,
        expected_sha256: str,
    ) -> dict[str, object]:
        record = {
            "name": name,
            "role": role,
            "url": url,
            "path": str(path),
            "size_bytes": int(size),
            "md5": str(actual_md5),
            "sha256": str(actual_sha256),
            "expected_size_bytes": expected_size,
            "expected_md5": expected_md5,
            "expected_sha256": expected_sha256,
            "verified": True,
        }
        if (
            record["size_bytes"] != expected_size
            or record["md5"] != expected_md5
            or record["sha256"] != expected_sha256
        ):
            raise ValueError(f"strict loader metadata mismatch for {name}")
        return record

    records: dict[str, Mapping[str, object]] = {
        "hrs_cube": loader_record(
            name="cube14v3.pic",
            path=smith_data_root / "cube14v3.pic",
            role="Smith et al. (2024) primary pre-eclipse IGRINS cube",
            url=cube_spec.url,
            size=hrs_metadata["cube_size_bytes"],
            actual_md5=hrs_metadata["cube_md5"],
            actual_sha256=hrs_metadata["cube_sha256"],
            expected_size=SMITH2024_WASP77AB_CUBE14_SIZE,
            expected_md5=SMITH2024_WASP77AB_CUBE14_MD5,
            expected_sha256=SMITH2024_WASP77AB_CUBE14_SHA256,
        ),
        "hrs_metadata": loader_record(
            name="20201214_info.csv",
            path=smith_data_root / "20201214_info.csv",
            role="Smith et al. (2024) primary pre-eclipse frame metadata",
            url=info_spec.url,
            size=hrs_metadata["info_size_bytes"],
            actual_md5=hrs_metadata["info_md5"],
            actual_sha256=hrs_metadata["info_sha256"],
            expected_size=SMITH2024_WASP77AB_INFO_SIZE,
            expected_md5=SMITH2024_WASP77AB_INFO_MD5,
            expected_sha256=SMITH2024_WASP77AB_INFO_SHA256,
        ),
        "official_template": loader_record(
            name=template_spec.name,
            path=Path(template_path),
            role="Smith et al. (2024) official R=500,000 NIRSpec template",
            url=template_spec.url,
            size=template_metadata.get("template_size_bytes", 0),
            actual_md5=template_metadata.get("template_md5", ""),
            actual_sha256=template_metadata.get("template_sha256", ""),
            expected_size=SMITH2024_WASP77AB_TEMPLATE_R500K_SIZE,
            expected_md5=SMITH2024_WASP77AB_TEMPLATE_R500K_MD5,
            expected_sha256=SMITH2024_WASP77AB_TEMPLATE_R500K_SHA256,
        ),
    }
    source_path = Path(str(nirspec_metadata["source_path"]))
    nirspec_spec = target.NIRSPEC_TABLE
    nirspec_size, nirspec_md5, nirspec_sha256 = _stream_hashes(source_path)
    if nirspec_size != nirspec_spec.size_bytes or nirspec_sha256 != nirspec_spec.sha256:
        raise ValueError("strict NIRSpec loader metadata mismatch")
    records["nirspec_table"] = {
        "name": nirspec_spec.name,
        "role": "August et al. (2023) / Smith et al. (2024) NIRSpec comparison table",
        "url": nirspec_spec.url,
        "path": str(source_path),
        "size_bytes": nirspec_size,
        "md5": nirspec_md5,
        "sha256": nirspec_sha256,
        "expected_size_bytes": nirspec_spec.size_bytes,
        "expected_md5": None,
        "expected_sha256": nirspec_spec.sha256,
        "verified": True,
    }
    return records


def load_reference_inputs(
    *,
    smith_data_root: Path = DEFAULT_SMITH_DATA_ROOT,
    nirspec_data_root: Path = DEFAULT_NIRSPEC_DATA_ROOT,
    template_path: Path = DEFAULT_TEMPLATE_PATH,
) -> ReferenceInputs:
    """Load and verify the fixed Smith HRS/LRS reference inputs."""

    smith_root = Path(smith_data_root).expanduser()
    nirspec_root = Path(nirspec_data_root).expanduser()
    template = load_smith2024_wasp77ab_template(
        Path(template_path).expanduser(),
        verify_checksum=True,
        verify_sha256=True,
        flux_ratio_scale=(target.PLANET.radius_m / target.STAR.radius_m) ** 2,
        name="Smith et al. 2024 official R=500,000 template",
    )
    hrs_observation = load_smith2024_wasp77ab_hrs(
        smith_root / "cube14v3.pic",
        info_path=smith_root / "20201214_info.csv",
        verify_checksum=True,
        verify_sha256=True,
    )
    nirspec_observations = load_august2023_wasp77ab(
        nirspec_root,
        verify_checksum=True,
    )
    if nirspec_observations.n_points != target.NIRSPEC_PUBLISHED_TABLE_POINTS:
        raise ValueError("NIRSpec source must contain the reconciled 150 points")
    spectra = _lrs_template_spectra(template, nirspec_observations)
    source_hashes = _loaded_source_hashes(
        smith_data_root=smith_root,
        nirspec_observations=nirspec_observations,
        template_path=Path(template_path).expanduser(),
        hrs_observation=hrs_observation,
        template=template,
    )
    return ReferenceInputs(
        hrs_observation=hrs_observation,
        nirspec_observations=nirspec_observations,
        template=template,
        hrs_template=_prepare_hrs_template(template, hrs_observation),
        lrs_template_spectra=spectra,
        source_hashes=source_hashes,
    )


def _parameter_set(
    specifications: Sequence[tuple[str, UniformPrior, str]],
) -> RetrievalParameterSet:
    """Construct an ordered ROBERT parameter set from named prior specs."""

    return RetrievalParameterSet(
        tuple(
            RetrievalParameter(name=name, prior=prior, unit=unit)
            for name, prior, unit in specifications
        )
    )


def _prepared_hrs_likelihood(
    observation: TimeResolvedHighResolutionObservation,
) -> object:
    """Prepare the Smith PCA statistic once for repeated sampler calls."""

    return TimeResolvedHighResolutionLikelihood(
        n_components=target.PRIMARY_HRS_PC_COUNT,
        kp_parameter=HRS_KP_PARAMETER,
        dVsys_parameter=HRS_DVSYS_PARAMETER,
        dphi_parameter="dphi_fixed_zero",
        scale_parameter=HRS_SCALE_PARAMETER,
        scale_is_log10=True,
        doppler_mode="smith_nonrelativistic",
        flux_ratio_scale=1.0,
        name="smith2024-primary-pre-eclipse-pca",
    ).prepare(observation)


def _hrs_component(prepared: object) -> HeterogeneousLikelihoodComponent:
    return HeterogeneousLikelihoodComponent(
        name="hrs",
        prediction_key="hrs",
        likelihood=prepared,
        metadata={
            "dataset": "Smith et al. 2024 IGRINS 2020-12-14 pre-eclipse",
            "statistic": (
                "ROBERT prepared Smith/Brogi-Line comparison statistic; "
                "not a claim of exact Smith likelihood reproduction"
            ),
            "observation_embedded": "true",
            "mask_policy": (
                "Zenodo cube is already the Smith-cleaned 44-order/1848-pixel "
                "product; no additional edge/order mask is inferred"
            ),
            "pca_components": str(target.PRIMARY_HRS_PC_COUNT),
            "pca_sigma_clip": (
                "3-sigma residual zero-fill used by this ROBERT comparison; "
                "not stated as a separate data-cleaning step in Smith et al."
            ),
            "template_response": (
                "official R=500,000 template convolved with Gray rotation and "
                "Gaussian LSF before detector-grid interpolation; no pixel "
                "integration is applied"
            ),
            "velocity_terms": (
                "fixed RV column (gamma plus barycentric correction) + "
                "Kp*sin(2*pi*phase) + dVsys"
            ),
        },
    )


def _lrs_components(
    observations: ObservationCollection,
) -> tuple[HeterogeneousLikelihoodComponent, ...]:
    """Create two explicit-observation detector likelihood components."""

    gaussian = GaussianLikelihood(
        include_normalization=False,
        offset_parameter=None,
        jitter_parameter=None,
        uncertainty_scale_parameter=None,
    )
    return tuple(
        HeterogeneousLikelihoodComponent(
            name=dataset.name,
            prediction_key=dataset.name,
            likelihood=gaussian,
            observation=dataset.observation,
            metadata={
                "dataset": dataset.name,
                "detector": str(dataset.metadata.get("detector", "unknown")),
                "likelihood": "independent Gaussian with symmetrised published uncertainty",
                "covariance": "not published; detector terms treated independently",
            },
        )
        for dataset in observations.datasets
    )


def _lrs_prediction(
    inputs: ReferenceInputs,
    parameters: Mapping[str, float],
) -> dict[str, Spectrum]:
    """Apply the one bounded multiplicative scale to both detector spectra."""

    scale = float(parameters[LRS_SCALE_PARAMETER])
    if not np.isfinite(scale) or scale <= 0.0:
        raise RobertValidationError("LRS template scale must be finite and positive")
    return {
        name: Spectrum.from_arrays(
            spectrum.spectral_grid.values,
            scale * spectrum.values,
            unit=spectrum.unit,
            observable=spectrum.observable,
            wavelength_unit=spectrum.spectral_grid.unit,
        )
        for name, spectrum in inputs.lrs_template_spectra.items()
    }


def build_hrs_problem(
    inputs: ReferenceInputs,
    *,
    prepared_likelihood: object | None = None,
) -> HeterogeneousRetrievalProblem:
    """Build the three-parameter fixed-template HRS-only problem."""

    prepared = (
        _prepared_hrs_likelihood(inputs.hrs_observation)
        if prepared_likelihood is None
        else prepared_likelihood
    )
    parameters = _parameter_set(
        (
            (HRS_KP_PARAMETER, HRS_KP_PRIOR, "km s^-1"),
            (HRS_DVSYS_PARAMETER, HRS_DVSYS_PRIOR, "km s^-1"),
            (HRS_SCALE_PARAMETER, HRS_SCALE_PRIOR, "log10"),
        )
    )

    def forward(parameter_values: Mapping[str, float]) -> Mapping[str, object]:
        del parameter_values
        return {"hrs": inputs.hrs_template}

    return HeterogeneousRetrievalProblem(
        name="wasp77ab-smith2024-hrs-only",
        parameters=parameters,
        forward_model=forward,
        likelihood=HeterogeneousLikelihood(
            (_hrs_component(prepared),),
            name="wasp77ab-smith2024-hrs-likelihood",
        ),
        invalid_loglike=INVALID_LOGLIKE_FLOOR,
        metadata={
            "target": "WASP-77Ab",
            "data_date": "2020-12-14",
            "data_role": "Smith et al. pre-eclipse primary IGRINS HRS",
            "template_role": "official Smith R=500,000 template",
            "phase_offset": "dphi fixed to zero; not a retrieval parameter",
            "velocity_parameterization": (
                "absolute Kp in [172, 212] km s^-1; equivalent to Smith "
                "dKp=Kp-192 km s^-1"
            ),
            "velocity_fixed_terms": (
                "Zenodo RV [km/s] column is used as the fixed gamma plus "
                "barycentric term"
            ),
            "model_response": (
                "R=500,000 template convolved with 4.2 km s^-1 Gray rotation "
                "and Gaussian R=45,000 LSF before PCA likelihood interpolation"
            ),
            "smith_model_resolution_note": (
                "Smith used R=250,000; the public fixed-template carrier is "
                "R=500,000 before the same response stages"
            ),
            "abundance_state": "fixed official template; no VMR retrieval",
            "comparison_claim": "independent fixed-template comparison, not truth recovery",
        },
        opacity_identifiers={"template": "smith2024-w77-pre-nirspec-best-fit-R500K"},
    )


def build_lrs_problem(inputs: ReferenceInputs) -> HeterogeneousRetrievalProblem:
    """Build the LRS-only problem with separate NRS1/NRS2 terms."""

    parameters = _parameter_set(
        ((LRS_SCALE_PARAMETER, LRS_SCALE_PRIOR, "dimensionless"),)
    )

    def forward(parameter_values: Mapping[str, float]) -> Mapping[str, object]:
        return _lrs_prediction(inputs, parameter_values)

    return HeterogeneousRetrievalProblem(
        name="wasp77ab-smith2024-lrs-only",
        parameters=parameters,
        forward_model=forward,
        likelihood=HeterogeneousLikelihood(
            _lrs_components(inputs.nirspec_observations),
            name="wasp77ab-smith2024-lrs-likelihood",
        ),
        invalid_loglike=INVALID_LOGLIKE_FLOOR,
        metadata={
            "target": "WASP-77Ab",
            "data_role": "August 2023 NIRSpec G395H comparison input",
            "detector_terms": "NRS1 and NRS2 retained separately",
            "template_role": "official Smith R=500,000 template",
            "template_scale_prior": LRS_SCALE_PRIOR_DESCRIPTION,
            "lrs_model_binning": (
                "top-hat average of the official R=500,000 template over the "
                "150 published NIRSpec bins"
            ),
            "smith_lrs_method_note": (
                "Smith reports R=100,000 model spectra and approximately R=250 "
                "data bins; this fixed-template comparison uses the public "
                "R=500,000 product and is not a reproduction"
            ),
            "abundance_state": "fixed official template; no VMR retrieval",
            "comparison_claim": "independent fixed-template comparison, not truth recovery",
        },
        opacity_identifiers={"template": "smith2024-w77-pre-nirspec-best-fit-R500K"},
    )


def build_joint_problem(
    inputs: ReferenceInputs,
    *,
    prepared_likelihood: object | None = None,
) -> HeterogeneousRetrievalProblem:
    """Build the joint HRS plus two-detector LRS problem."""

    prepared = (
        _prepared_hrs_likelihood(inputs.hrs_observation)
        if prepared_likelihood is None
        else prepared_likelihood
    )
    parameters = _parameter_set(
        (
            (HRS_KP_PARAMETER, HRS_KP_PRIOR, "km s^-1"),
            (HRS_DVSYS_PARAMETER, HRS_DVSYS_PRIOR, "km s^-1"),
            (HRS_SCALE_PARAMETER, HRS_SCALE_PRIOR, "log10"),
            (LRS_SCALE_PARAMETER, LRS_SCALE_PRIOR, "dimensionless"),
        )
    )

    def forward(parameter_values: Mapping[str, float]) -> Mapping[str, object]:
        return {
            "hrs": inputs.hrs_template,
            **_lrs_prediction(inputs, parameter_values),
        }

    return HeterogeneousRetrievalProblem(
        name="wasp77ab-smith2024-joint",
        parameters=parameters,
        forward_model=forward,
        likelihood=HeterogeneousLikelihood(
            (_hrs_component(prepared), *_lrs_components(inputs.nirspec_observations)),
            name="wasp77ab-smith2024-joint-likelihood",
        ),
        invalid_loglike=INVALID_LOGLIKE_FLOOR,
        metadata={
            "target": "WASP-77Ab",
            "data_role": "Smith pre-eclipse IGRINS plus August NIRSpec",
            "shared_template": "true",
            "detector_terms": "NRS1 and NRS2 retained separately",
            "template_scale_prior": LRS_SCALE_PRIOR_DESCRIPTION,
            "lrs_model_binning": (
                "top-hat average of the official R=500,000 template over the "
                "150 published NIRSpec bins"
            ),
            "smith_lrs_method_note": (
                "Smith reports R=100,000 model spectra and approximately R=250 "
                "data bins; this fixed-template comparison uses the public "
                "R=500,000 product and is not a reproduction"
            ),
            "velocity_parameterization": (
                "absolute Kp in [172, 212] km s^-1; equivalent to Smith "
                "dKp=Kp-192 km s^-1"
            ),
            "model_response": (
                "R=500,000 template convolved with 4.2 km s^-1 Gray rotation "
                "and Gaussian R=45,000 LSF before PCA likelihood interpolation"
            ),
            "smith_model_resolution_note": (
                "Smith used R=250,000; the public fixed-template carrier is "
                "R=500,000 before the same response stages"
            ),
            "abundance_state": "fixed official template; no VMR retrieval",
            "comparison_claim": "independent fixed-template comparison, not truth recovery",
        },
        opacity_identifiers={"template": "smith2024-w77-pre-nirspec-best-fit-R500K"},
    )


def build_reference_problems(
    inputs: ReferenceInputs,
) -> dict[str, HeterogeneousRetrievalProblem]:
    """Build HRS-only, LRS-only, and joint problems in stable order."""

    prepared = _prepared_hrs_likelihood(inputs.hrs_observation)
    return {
        HRS_MODE: build_hrs_problem(inputs, prepared_likelihood=prepared),
        LRS_MODE: build_lrs_problem(inputs),
        JOINT_MODE: build_joint_problem(inputs, prepared_likelihood=prepared),
    }


def _weighted_quantile(
    values: ArrayLike,
    quantiles: ArrayLike,
    weights: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Return weighted quantiles without constructing a sample matrix copy."""

    numbers = np.asarray(values, dtype=float)
    requested = np.asarray(quantiles, dtype=float)
    if numbers.ndim != 1 or numbers.size == 0 or not np.all(np.isfinite(numbers)):
        raise ValueError("posterior values must be a non-empty finite vector")
    if requested.ndim != 1 or np.any(requested < 0.0) or np.any(requested > 1.0):
        raise ValueError("posterior quantiles must lie in [0, 1]")
    if weights is None:
        sample_weights = np.ones(numbers.size, dtype=float)
    else:
        sample_weights = np.asarray(weights, dtype=float)
        if sample_weights.shape != numbers.shape:
            raise ValueError("posterior weights must match sample values")
        if not np.all(np.isfinite(sample_weights)) or np.any(sample_weights < 0.0):
            raise ValueError("posterior weights must be finite and non-negative")
    total_weight = float(np.sum(sample_weights))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("posterior weights must have positive total weight")
    order = np.argsort(numbers)
    sorted_values = numbers[order]
    sorted_weights = sample_weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative /= total_weight
    output = np.interp(requested, cumulative, sorted_values)
    output.setflags(write=False)
    return output


def posterior_summary(result: object) -> dict[str, Mapping[str, float]]:
    """Summarise the median and central 90% interval of a sampler result."""

    samples = np.asarray(getattr(result, "samples"), dtype=float)
    names = tuple(str(name) for name in getattr(result, "parameter_names"))
    if samples.ndim != 2 or samples.shape[1] != len(names) or samples.shape[0] == 0:
        raise ValueError("nested-sampler samples do not match parameter names")
    weights = getattr(result, "weights", None)
    output: dict[str, Mapping[str, float]] = {}
    for index, name in enumerate(names):
        lower, median, upper = _weighted_quantile(
            samples[:, index],
            [0.05, 0.5, 0.95],
            weights,
        )
        output[name] = {
            "median": float(median),
            "lower_90": float(lower),
            "upper_90": float(upper),
        }
    return output


def _component_rank(component: HeterogeneousLikelihoodComponent) -> int | None:
    """Read an exact rank from a prepared or explicit likelihood when exposed."""

    value = getattr(component.likelihood, "effective_residual_rank", None)
    if value is None:
        return None
    if callable(value):
        if component.observation is None:
            return None
        value = value(component.observation)
    try:
        rank = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return rank if rank > 0 else None


def component_statistics(
    problem: HeterogeneousRetrievalProblem,
    result: object,
) -> dict[str, Mapping[str, object]]:
    """Evaluate exact component terms at a sampler best-fit vector."""

    best_fit = getattr(result, "best_fit_parameters", {})
    vector = np.asarray(
        [float(best_fit[name]) for name in problem.parameter_names],
        dtype=float,
    )
    terms = problem.log_likelihood_by_component_from_vector(vector)
    predictions: Mapping[str, object] | None = None
    parameters: Mapping[str, float] | None = None
    try:
        parameters = problem.parameter_mapping(vector)
        candidate = problem.predict(parameters)
        if isinstance(candidate, Mapping):
            predictions = candidate
    except (RobertValidationError, ValueError, TypeError, FloatingPointError, OverflowError):
        # The sampler result remains reportable when a best-fit vector is no
        # longer evaluable (for example, after an external file changes).
        predictions = None
    components: dict[str, Mapping[str, object]] = {}
    for component in problem.likelihood.components:
        chi_square: float | None = None
        reduced_chi_square: float | None = None
        chi_square_method = getattr(component.likelihood, "chi_square", None)
        if (
            predictions is not None
            and parameters is not None
            and component.observation is not None
            and callable(chi_square_method)
        ):
            try:
                value = float(
                    chi_square_method(
                        predictions[component.prediction_key],
                        component.observation,
                        parameters,
                    )
                )
                if np.isfinite(value) and value >= 0.0:
                    chi_square = value
                    rank = _component_rank(component)
                    if rank is not None:
                        reduced_chi_square = value / rank
            except (
                RobertValidationError,
                KeyError,
                ValueError,
                TypeError,
                FloatingPointError,
                OverflowError,
            ):
                chi_square = None
        components[component.name] = {
            "loglike": float(terms[component.name]),
            "effective_residual_rank": _component_rank(component),
            "chi_square": chi_square,
            "reduced_chi_square": reduced_chi_square,
            "prediction_key": component.prediction_key,
            "observation_embedded": component.observation is None,
        }
    return components


def _likelihood_calls(result: object) -> int | None:
    metadata = getattr(result, "metadata", {})
    raw = metadata.get("likelihood_evaluations") if hasattr(metadata, "get") else None
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value >= 0 else None


def _result_record(
    *,
    mode: str,
    seed: int,
    problem: HeterogeneousRetrievalProblem,
    result: object,
    elapsed_seconds: float,
    resources: Mapping[str, object],
    output_dir: Path,
) -> dict[str, object]:
    log_likelihood = np.asarray(getattr(result, "log_likelihood"), dtype=float)
    best_fit_loglike = None
    if log_likelihood.size and np.all(np.isfinite(log_likelihood)):
        best_fit_loglike = float(np.max(log_likelihood))
    evidence = getattr(result, "log_evidence", None)
    evidence_error = getattr(result, "log_evidence_error", None)
    sampler_metadata = getattr(result, "metadata", {})
    effective_seed = None
    if hasattr(sampler_metadata, "get"):
        raw_effective_seed = sampler_metadata.get("effective_multinest_seed")
        if raw_effective_seed is not None:
            try:
                effective_seed = int(raw_effective_seed)
            except (TypeError, ValueError, OverflowError):
                effective_seed = None
    return {
        "mode": mode,
        "seed": int(seed),
        "sampler": "PyMultiNest",
        "backend": "MultiNest",
        "effective_multinest_seed": effective_seed,
        "output_dir": str(output_dir),
        "status": "pass" if bool(getattr(result, "converged", False)) else "fail",
        "converged": bool(getattr(result, "converged", False)),
        "message": str(getattr(result, "message", "")),
        "calls": _likelihood_calls(result),
        "wall_time_seconds": float(elapsed_seconds),
        "peak_rss_bytes": resources.get("peak_rss_bytes"),
        "resource_gate_passed": bool(resources.get("gate_passed", False)),
        "log_evidence": None if evidence is None else float(evidence),
        "log_evidence_error": None if evidence_error is None else float(evidence_error),
        "best_fit_log_likelihood": best_fit_loglike,
        "best_fit_parameters": {
            str(name): float(value)
            for name, value in getattr(result, "best_fit_parameters", {}).items()
        },
        "per_component": component_statistics(problem, result),
        "posterior": posterior_summary(result),
    }


def _evidence_comparison(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Compare the two fixed-seed evidences with their combined uncertainty."""

    if len(records) != 2:
        return {
            "seed_pair": [int(record["seed"]) for record in records],
            "delta_log_evidence": None,
            "combined_error": None,
            "repeatability_limit": EVIDENCE_TOLERANCE,
            "gate_passed": False,
        }
    first, second = records
    evidence_1 = first.get("log_evidence")
    evidence_2 = second.get("log_evidence")
    error_1 = first.get("log_evidence_error")
    error_2 = second.get("log_evidence_error")
    finite_evidence = False
    delta: float | None = None
    if evidence_1 is not None and evidence_2 is not None:
        try:
            values = np.asarray((evidence_1, evidence_2), dtype=float)
        except (TypeError, ValueError, OverflowError):
            values = np.asarray((np.nan, np.nan), dtype=float)
        finite_evidence = bool(np.all(np.isfinite(values)))
        if finite_evidence:
            delta = abs(float(values[0]) - float(values[1]))

    finite_errors = False
    combined_error: float | None = None
    if error_1 is not None and error_2 is not None:
        try:
            errors = np.asarray((error_1, error_2), dtype=float)
        except (TypeError, ValueError, OverflowError):
            errors = np.asarray((np.nan, np.nan), dtype=float)
        finite_errors = bool(np.all(np.isfinite(errors)) and np.all(errors >= 0.0))
        if finite_errors:
            combined_error = float(np.hypot(float(errors[0]), float(errors[1])))
    gate_passed = bool(
        all(bool(record.get("converged")) for record in records)
        and all(bool(record.get("resource_gate_passed")) for record in records)
        and finite_evidence
        and finite_errors
        and delta is not None
        and delta <= EVIDENCE_TOLERANCE
    )
    return {
        "seed_pair": [int(first["seed"]), int(second["seed"])],
        "delta_log_evidence": delta,
        "combined_error": combined_error,
        "finite_evidence": finite_evidence,
        "finite_reported_errors": finite_errors,
        "within_combined_error": (
            bool(delta <= combined_error)
            if delta is not None and combined_error is not None
            else False
        ),
        "repeatability_limit": EVIDENCE_TOLERANCE,
        "gate_passed": gate_passed,
    }


def run_sampler_pair(
    problem: HeterogeneousRetrievalProblem,
    *,
    mode: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_memory_bytes: int = DEFAULT_PROCESS_RSS_LIMIT_BYTES,
    sampler_runner: object | None = None,
) -> dict[str, object]:
    """Run the two bounded PyMultiNest seeds for one problem."""

    if mode not in ALL_MODES:
        raise ValueError(f"unsupported reference mode: {mode}")
    resources_before = resource_record(max_memory_bytes=max_memory_bytes)
    if not bool(resources_before["gate_passed"]):
        raise RuntimeError("resource contract failed before starting PyMultiNest")
    if sampler_runner is None:
        from robert_exoplanets.retrieval.samplers import run_multinest

        sampler_runner = run_multinest
    if not callable(sampler_runner):
        raise TypeError("sampler_runner must be callable")

    records: list[Mapping[str, object]] = []
    for seed in SEEDS:
        seed_output = Path(output_root) / mode / f"seed-{seed}"
        resources_start = resource_record(max_memory_bytes=max_memory_bytes)
        if not bool(resources_start["gate_passed"]):
            records.append(
                {
                    "mode": mode,
                    "seed": int(seed),
                    "output_dir": str(seed_output),
                    "status": "fail",
                    "converged": False,
                    "message": "resource contract failed before starting PyMultiNest",
                    "calls": None,
                    "wall_time_seconds": 0.0,
                    "peak_rss_bytes": resources_start.get("peak_rss_bytes"),
                    "resource_gate_passed": False,
                    "log_evidence": None,
                    "log_evidence_error": None,
                    "best_fit_log_likelihood": None,
                    "best_fit_parameters": {},
                    "per_component": {},
                    "posterior": {},
                }
            )
            break
        started = perf_counter()
        try:
            result = sampler_runner(
                problem,
                output_dir=seed_output,
                n_live_points=N_LIVE_POINTS,
                max_iter=MAX_ITER,
                evidence_tolerance=EVIDENCE_TOLERANCE,
                sampling_efficiency=SAMPLING_EFFICIENCY,
                mpi_nprocs=MPI_PROCESSES,
                seed=seed,
                resume=False,
                verbose=False,
                invalid_loglike_floor=INVALID_LOGLIKE_FLOOR,
            )
            if str(getattr(result, "method", "")).strip().lower() != "multinest":
                raise RobertValidationError(
                    "reference workflow accepts only a MultiNest/PyMultiNest result"
                )
        except Exception as error:  # preserve an honest optional-run report
            resources = resource_record(max_memory_bytes=max_memory_bytes)
            records.append(
                {
                    "mode": mode,
                    "seed": int(seed),
                    "output_dir": str(seed_output),
                    "status": "fail",
                    "converged": False,
                    "message": f"{type(error).__name__}: {error}",
                    "calls": None,
                    "wall_time_seconds": max(perf_counter() - started, 0.0),
                    "peak_rss_bytes": resources.get("peak_rss_bytes"),
                    "resource_gate_passed": bool(resources.get("gate_passed", False)),
                    "log_evidence": None,
                    "log_evidence_error": None,
                    "best_fit_log_likelihood": None,
                    "best_fit_parameters": {},
                    "per_component": {},
                    "posterior": {},
                }
            )
            if not bool(resources.get("gate_passed", False)):
                break
            continue
        elapsed = max(perf_counter() - started, 0.0)
        resources = resource_record(max_memory_bytes=max_memory_bytes)
        records.append(
            _result_record(
                mode=mode,
                seed=seed,
                problem=problem,
                result=result,
                elapsed_seconds=elapsed,
                resources=resources,
                output_dir=seed_output,
            )
        )
        # Do not start the next seed after a process-resource violation.  The
        # hard gate is intentionally checked after every independent run.
        if not bool(resources.get("gate_passed", False)):
            break
    comparison = _evidence_comparison(records)
    return {
        "mode": mode,
        "runs": records,
        "evidence_comparison": comparison,
        "gate_passed": bool(comparison["gate_passed"]),
    }


def _problem_record(problem: HeterogeneousRetrievalProblem) -> dict[str, object]:
    """Serialize the fixed problem definition without evaluating a model."""

    return {
        "name": problem.name,
        "parameter_names": list(problem.parameter_names),
        "parameter_priors": {
            parameter.name: {
                "lower": float(parameter.prior.lower),
                "upper": float(parameter.prior.upper),
                "unit": parameter.unit,
            }
            for parameter in problem.parameters.parameters
        },
        "component_names": list(problem.likelihood.component_names),
        "prediction_keys": list(problem.likelihood.prediction_keys),
        "component_metadata": {
            component.name: dict(component.metadata)
            for component in problem.likelihood.components
        },
        "metadata": dict(problem.metadata),
        "opacity_identifiers": dict(problem.opacity_identifiers),
    }


def _contract_problem_records() -> dict[str, Mapping[str, object]]:
    """Return problem contracts for a dry run without loading external data."""

    return {
        HRS_MODE: {
            "name": "wasp77ab-smith2024-hrs-only",
            "parameter_names": [
                HRS_KP_PARAMETER,
                HRS_DVSYS_PARAMETER,
                HRS_SCALE_PARAMETER,
            ],
            "parameter_priors": {
                HRS_KP_PARAMETER: {
                    "lower": 172.0,
                    "upper": 212.0,
                    "unit": "km s^-1",
                },
                HRS_DVSYS_PARAMETER: {
                    "lower": -20.0,
                    "upper": 20.0,
                    "unit": "km s^-1",
                },
                HRS_SCALE_PARAMETER: {
                    "lower": -2.0,
                    "upper": 2.0,
                    "unit": "log10",
                },
            },
            "component_names": ["hrs"],
            "prediction_keys": ["hrs"],
        },
        LRS_MODE: {
            "name": "wasp77ab-smith2024-lrs-only",
            "parameter_names": [LRS_SCALE_PARAMETER],
            "parameter_priors": {
                LRS_SCALE_PARAMETER: {
                    "lower": 0.5,
                    "upper": 1.5,
                    "unit": "dimensionless",
                }
            },
            "component_names": ["nirspec_g395h_nrs1", "nirspec_g395h_nrs2"],
            "prediction_keys": ["nirspec_g395h_nrs1", "nirspec_g395h_nrs2"],
        },
        JOINT_MODE: {
            "name": "wasp77ab-smith2024-joint",
            "parameter_names": [
                HRS_KP_PARAMETER,
                HRS_DVSYS_PARAMETER,
                HRS_SCALE_PARAMETER,
                LRS_SCALE_PARAMETER,
            ],
            "parameter_priors": {
                HRS_KP_PARAMETER: {
                    "lower": 172.0,
                    "upper": 212.0,
                    "unit": "km s^-1",
                },
                HRS_DVSYS_PARAMETER: {
                    "lower": -20.0,
                    "upper": 20.0,
                    "unit": "km s^-1",
                },
                HRS_SCALE_PARAMETER: {
                    "lower": -2.0,
                    "upper": 2.0,
                    "unit": "log10",
                },
                LRS_SCALE_PARAMETER: {
                    "lower": 0.5,
                    "upper": 1.5,
                    "unit": "dimensionless",
                },
            },
            "component_names": [
                "hrs",
                "nirspec_g395h_nrs1",
                "nirspec_g395h_nrs2",
            ],
            "prediction_keys": [
                "hrs",
                "nirspec_g395h_nrs1",
                "nirspec_g395h_nrs2",
            ],
        },
    }


def _expected_source_records(
    *,
    smith_data_root: Path,
    nirspec_data_root: Path,
    template_path: Path,
) -> dict[str, Mapping[str, object]]:
    """Describe expected source identities before local files are acquired."""

    cube = _expected_file("cube14v3.pic")
    info = _expected_file("20201214_info.csv")
    template = _expected_file("w77_pre_nirspec_best_fit_R500K_scaled.txt")
    nirspec = target.NIRSPEC_TABLE
    return {
        "hrs_cube": {
            "name": cube.name,
            "role": "Smith et al. (2024) primary pre-eclipse IGRINS cube",
            "url": cube.url,
            "path": str(Path(smith_data_root) / cube.name),
            "expected_size_bytes": SMITH2024_WASP77AB_CUBE14_SIZE,
            "expected_md5": SMITH2024_WASP77AB_CUBE14_MD5,
            "expected_sha256": SMITH2024_WASP77AB_CUBE14_SHA256,
            "verified": False,
        },
        "hrs_metadata": {
            "name": info.name,
            "role": "Smith et al. (2024) primary pre-eclipse frame metadata",
            "url": info.url,
            "path": str(Path(smith_data_root) / info.name),
            "expected_size_bytes": SMITH2024_WASP77AB_INFO_SIZE,
            "expected_md5": SMITH2024_WASP77AB_INFO_MD5,
            "expected_sha256": SMITH2024_WASP77AB_INFO_SHA256,
            "verified": False,
        },
        "official_template": {
            "name": template.name,
            "role": "Smith et al. (2024) official R=500,000 NIRSpec template",
            "url": template.url,
            "path": str(Path(template_path)),
            "expected_size_bytes": SMITH2024_WASP77AB_TEMPLATE_R500K_SIZE,
            "expected_md5": SMITH2024_WASP77AB_TEMPLATE_R500K_MD5,
            "expected_sha256": SMITH2024_WASP77AB_TEMPLATE_R500K_SHA256,
            "verified": False,
        },
        "nirspec_table": {
            "name": nirspec.name,
            "role": "August et al. (2023) / Smith et al. (2024) NIRSpec table",
            "url": nirspec.url,
            "path": str(Path(nirspec_data_root)),
            "expected_size_bytes": nirspec.size_bytes,
            "expected_md5": None,
            "expected_sha256": nirspec.sha256,
            "verified": False,
        },
    }


def _write_plot(
    path: Path,
    inputs: ReferenceInputs,
    best_fit_parameters: Mapping[str, float] | None = None,
) -> str:
    """Write an optional compact NIRSpec data/template comparison plot."""

    import matplotlib.pyplot as plt

    scale = 1.0
    if best_fit_parameters is not None and LRS_SCALE_PARAMETER in best_fit_parameters:
        scale = float(best_fit_parameters[LRS_SCALE_PARAMETER])
    figure, axis = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    colors = ("#2f6f9f", "#d98324")
    for color, dataset in zip(
        colors,
        inputs.nirspec_observations.datasets,
        strict=True,
    ):
        observation = dataset.observation
        template = inputs.lrs_template_spectra[dataset.name]
        axis.errorbar(
            observation.wavelength,
            1.0e6 * observation.flux,
            yerr=1.0e6 * observation.uncertainty,
            fmt="o",
            markersize=2.8,
            linewidth=0.7,
            color=color,
            label=dataset.name.replace("nirspec_g395h_", "").upper(),
        )
        axis.plot(
            template.spectral_grid.values,
            1.0e6 * scale * template.values,
            color=color,
            linewidth=1.0,
        )
    axis.set_xlabel("Wavelength (micron)")
    axis.set_ylabel("Eclipse depth (ppm)")
    axis.set_title("WASP-77Ab Smith fixed-template NIRSpec reference")
    axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return _stream_hashes(path)[2]


def write_report(report: Mapping[str, object], path: Path) -> None:
    """Write one deterministic JSON report with no non-finite values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_workflow(
    *,
    smith_data_root: Path = DEFAULT_SMITH_DATA_ROOT,
    nirspec_data_root: Path = DEFAULT_NIRSPEC_DATA_ROOT,
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    report_path: Path | None = DEFAULT_REPORT_PATH,
    plot_path: Path | None = None,
    modes: Sequence[str] = ALL_MODES,
    preflight: bool = False,
    run_sampler: bool = False,
    dry_run: bool = True,
    max_memory_bytes: int = DEFAULT_PROCESS_RSS_LIMIT_BYTES,
    sampler_runner: object | None = None,
) -> dict[str, object]:
    """Run dry-run, preflight, or the bounded two-seed sampler workflow."""

    selected_modes = tuple(modes)
    if selected_modes == ("all",):
        selected_modes = ALL_MODES
    if not selected_modes or any(mode not in ALL_MODES for mode in selected_modes):
        raise ValueError(f"modes must be drawn from {ALL_MODES}")
    if len(set(selected_modes)) != len(selected_modes):
        raise ValueError("modes must be unique")
    if not 0 < int(max_memory_bytes) < PROCESS_RSS_LIMIT_BYTES:
        raise ValueError("max_memory_bytes must be positive and strictly below 2 GiB")
    if run_sampler:
        dry_run = False

    source_hashes: Mapping[str, Mapping[str, object]] = _expected_source_records(
        smith_data_root=Path(smith_data_root),
        nirspec_data_root=Path(nirspec_data_root),
        template_path=Path(template_path),
    )
    inputs: ReferenceInputs | None = None
    problems: Mapping[str, HeterogeneousRetrievalProblem] | None = None
    sampler_records: dict[str, Mapping[str, object]] = {}
    plot_record: Mapping[str, object] | None = None
    if preflight or run_sampler:
        if preflight:
            source_hashes = preflight_inputs(
                smith_data_root=Path(smith_data_root),
                nirspec_data_root=Path(nirspec_data_root),
                template_path=Path(template_path),
            )
        if run_sampler:
            inputs = load_reference_inputs(
                smith_data_root=Path(smith_data_root),
                nirspec_data_root=Path(nirspec_data_root),
                template_path=Path(template_path),
            )
            source_hashes = {
                name: dict(record) for name, record in inputs.source_hashes.items()
            }
            problems = build_reference_problems(inputs)
            for mode in selected_modes:
                sampler_records[mode] = run_sampler_pair(
                    problems[mode],
                    mode=mode,
                    output_root=Path(output_root),
                    max_memory_bytes=max_memory_bytes,
                    sampler_runner=sampler_runner,
                )
            if plot_path is not None and sampler_records:
                best_fit: Mapping[str, float] | None = None
                joint_runs = sampler_records.get(JOINT_MODE, {}).get("runs", [])
                if joint_runs:
                    best_fit = joint_runs[0].get("best_fit_parameters")
                elif sampler_records:
                    first_mode = next(iter(sampler_records))
                    runs = sampler_records[first_mode].get("runs", [])
                    if runs:
                        best_fit = runs[0].get("best_fit_parameters")
                plot_hash = _write_plot(Path(plot_path), inputs, best_fit)
                plot_record = {"path": str(plot_path), "sha256": plot_hash}

    resources = resource_record(
        max_memory_bytes=max_memory_bytes,
        check_peak=not dry_run,
    )
    if run_sampler:
        status = (
            "pass"
            if bool(sampler_records)
            and all(bool(record.get("gate_passed")) for record in sampler_records.values())
            and bool(resources["gate_passed"])
            else "fail"
        )
    elif preflight:
        status = "preflight_pass" if bool(resources["gate_passed"]) else "fail"
    else:
        status = "dry_run"
    if problems is not None:
        problem_records: Mapping[str, object] = {
            mode: _problem_record(problems[mode]) for mode in selected_modes
        }
    else:
        contracts = _contract_problem_records()
        problem_records = {mode: contracts[mode] for mode in selected_modes}
    if inputs is None:
        hrs_response_record: Mapping[str, object] = {
            "stage_order": "gray_rotation_then_gaussian_lsf",
            "rotation_vsin_km_s": HRS_ROTATION_VSIN_KM_S,
            "rotation_limb_darkening": HRS_ROTATION_LIMB_DARKENING,
            "gaussian_resolving_power": HRS_GAUSSIAN_RESOLVING_POWER,
            "gaussian_kernel_support_sigma": HRS_GAUSSIAN_KERNEL_SUPPORT_SIGMA,
            "pixel_integration": "not applied; Smith interpolates the convolved model",
            "input_resolution": "R=500,000",
            "smith_model_resolution": "R=250,000",
        }
    else:
        hrs_response_record = {
            str(key): value
            for key, value in inputs.hrs_template.metadata.items()
            if str(key).startswith("hrs_")
        }
    report: dict[str, object] = {
        "schema_version": "1.0",
        "status": status,
        "claim": (
            "independent ROBERT fixed-template comparison on Smith et al. "
            "WASP-77Ab HRS/LRS data; no truth-recovery claim and not a Smith-method reproduction"
        ),
        "truth_recovery_claim": False,
        "sampler_policy": {
            "sampler": "PyMultiNest",
            "backend": "MultiNest",
            "seeds": list(SEEDS),
            "n_live_points": N_LIVE_POINTS,
            "max_iter": MAX_ITER,
            "evidence_tolerance": EVIDENCE_TOLERANCE,
            "sampling_efficiency": SAMPLING_EFFICIENCY,
            "mpi_processes": MPI_PROCESSES,
            "ultranest_allowed": False,
        },
        "data": {
            "target": "WASP-77Ab",
            "hrs_role": "Smith et al. 2024 primary 2020-12-14 pre-eclipse IGRINS",
            "lrs_role": "August et al. 2023 NIRSpec G395H comparison input",
            "nirspec_detector_terms": ["NRS1", "NRS2"],
            "nirspec_point_count": target.NIRSPEC_PUBLISHED_TABLE_POINTS,
            "nirspec_count_note": target.NIRSPEC_POINT_COUNT_STATUS,
            "template_role": "official Smith R=500,000 template",
            "hrs_response": (
                "Gray rotation vsini=4.2 km s^-1 followed by Gaussian "
                "R=45,000 LSF; no pixel integration; detector interpolation "
                "is performed by the temporal PCA likelihood"
            ),
            "hrs_model_resolution_note": (
                "Smith used R=250,000; this workflow carries the public "
                "R=500,000 template through the response stages"
            ),
            "hrs_template_crop": (
                "full Kp/dVsys prior velocity range plus rotation/LSF support "
                "margins; recorded on the prepared HRS template metadata"
            ),
            "hrs_response_template": hrs_response_record,
            "lrs_response": (
                "original R=500,000 template with top-hat averages over the "
                "published 150 NIRSpec bins"
            ),
            "abundance_state": "fixed template; no ROBERT abundance parameters; VMR-only policy retained",
            "source_hashes": source_hashes,
        },
        "modes": list(selected_modes),
        "problems": problem_records,
        "runs": sampler_records,
        "resources": resources,
        "plot": plot_record,
    }
    if report_path is not None:
        write_report(report, Path(report_path))
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", *ALL_MODES),
        default="all",
        help="Reference fit to describe or run (default: all).",
    )
    parser.add_argument("--smith-data-root", type=Path, default=DEFAULT_SMITH_DATA_ROOT)
    parser.add_argument("--nirspec-data-root", type=Path, default=DEFAULT_NIRSPEC_DATA_ROOT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--max-memory-gib",
        type=float,
        default=DEFAULT_PROCESS_RSS_LIMIT_BYTES / 1024**3,
        help="Configured strict process RSS guard; must be below 2 GiB.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Verify external source hashes without starting a sampler.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fixed contract without reading external data (default).",
    )
    parser.add_argument(
        "--run-pymultinest",
        action="store_true",
        help="Run the two fixed-seed PyMultiNest fits for the selected mode(s).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Execute a dry run, preflight, or explicitly requested sampler run."""

    args = _parse_args(argv)
    selected_modes = ALL_MODES if args.mode == "all" else (args.mode,)
    max_memory_bytes = int(float(args.max_memory_gib) * 1024**3)
    report = run_workflow(
        smith_data_root=args.smith_data_root,
        nirspec_data_root=args.nirspec_data_root,
        template_path=args.template,
        output_root=args.output_root,
        report_path=args.report,
        plot_path=None if args.no_plot else args.plot,
        modes=selected_modes,
        preflight=bool(args.preflight or args.run_pymultinest),
        run_sampler=bool(args.run_pymultinest),
        dry_run=not bool(args.preflight or args.run_pymultinest),
        max_memory_bytes=max_memory_bytes,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if report["status"] == "fail":
        raise SystemExit("WASP-77Ab reference workflow failed its acceptance gate")


if __name__ == "__main__":
    main()


__all__ = [
    "ALL_MODES",
    "DEFAULT_NIRSPEC_DATA_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PROCESS_RSS_LIMIT_BYTES",
    "DEFAULT_REPORT_PATH",
    "DEFAULT_SMITH_DATA_ROOT",
    "DEFAULT_TEMPLATE_PATH",
    "EVIDENCE_TOLERANCE",
    "HRS_DVSYS_PARAMETER",
    "HRS_DVSYS_PRIOR",
    "HRS_KP_PARAMETER",
    "HRS_KP_PRIOR",
    "HRS_MODE",
    "HRS_SCALE_PARAMETER",
    "HRS_SCALE_PRIOR",
    "JOINT_MODE",
    "LRS_MODE",
    "LRS_SCALE_PARAMETER",
    "LRS_SCALE_PRIOR",
    "LRS_SCALE_PRIOR_DESCRIPTION",
    "MAX_ITER",
    "MAX_THREADS",
    "MPI_PROCESSES",
    "N_LIVE_POINTS",
    "PROCESS_RSS_LIMIT_BYTES",
    "ReferenceInputs",
    "SAMPLING_EFFICIENCY",
    "SEEDS",
    "THREAD_VARIABLES",
    "build_hrs_problem",
    "build_joint_problem",
    "build_lrs_problem",
    "build_reference_problems",
    "component_statistics",
    "load_reference_inputs",
    "main",
    "posterior_summary",
    "preflight_inputs",
    "resource_record",
    "run_sampler_pair",
    "run_workflow",
    "top_hat_bin_average",
    "write_report",
]
