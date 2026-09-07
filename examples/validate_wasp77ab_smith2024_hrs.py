"""Run a bounded, deterministic Smith et al. (2024) WASP-77Ab HRS check.

The workflow uses the public 2020-12-14 IGRINS cube and the three public
Smith templates.  It verifies every file by size, MD5, and SHA-256 before
the ROBERT loaders read it.  It evaluates the exact four-component,
Smith non-relativistic PCA/Brogi-Line likelihood and one small local velocity
scan.  It does not run a sampler.

Each R=250,000 template is cropped to the real cube and the bounded velocity
support, then passed through Gray rotation (4.2 km/s) and a Gaussian IGRINS
LSF (R=45,000).  The likelihood interpolates this processed template onto
the cube pixels; no pixel-bin integration is performed here.

The paper reports pre-eclipse CCF SNR values of 9.4 (full model), 9.2 (H2O),
and 3.6 (CO).  Those values use the paper's full two-dimensional CCF-map
normalisation.  The local scan in this example reports raw summed CCF values;
it is not a reproduction of those SNR values.
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
from typing import Final, Mapping


ROOT: Final = Path(__file__).resolve().parents[1]
THREAD_VARIABLES: Final[tuple[str, ...]] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)


def _configure_thread_limits() -> None:
    """Clamp all numerical thread variables to the inclusive range 1--3."""

    for name in THREAD_VARIABLES:
        raw_value = os.environ.get(name, "3")
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = 3
        os.environ[name] = str(min(3, max(1, value)))


_configure_thread_limits()
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib")
)

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from robert_exoplanets.core import (  # noqa: E402
    RobertCoverageError,
    SpectralGrid,
    Spectrum,
)
from robert_exoplanets.instruments.high_resolution import (  # noqa: E402
    GaussianHighResolutionResponse,
    HighResolutionResponseChain,
    RotationalBroadeningResponse,
)
from robert_exoplanets.instruments.time_resolved_high_resolution import (  # noqa: E402
    HighResolutionEmissionTemplate,
    TimeResolvedHighResolutionObservation,
)
from robert_exoplanets.io import (  # noqa: E402
    SMITH2024_WASP77AB_CUBE14_MD5,
    SMITH2024_WASP77AB_CUBE14_SHA256,
    SMITH2024_WASP77AB_CUBE14_SIZE,
    SMITH2024_WASP77AB_FLUX_RATIO_SCALE,
    SMITH2024_WASP77AB_INFO_MD5,
    SMITH2024_WASP77AB_INFO_SHA256,
    SMITH2024_WASP77AB_INFO_SIZE,
    SMITH2024_WASP77AB_TEMPLATE_CO_MD5,
    SMITH2024_WASP77AB_TEMPLATE_CO_SIZE,
    SMITH2024_WASP77AB_TEMPLATE_FULL_MD5,
    SMITH2024_WASP77AB_TEMPLATE_FULL_SHA256,
    SMITH2024_WASP77AB_TEMPLATE_FULL_SIZE,
    SMITH2024_WASP77AB_TEMPLATE_H2O_MD5,
    SMITH2024_WASP77AB_TEMPLATE_H2O_SIZE,
    load_smith2024_wasp77ab_hrs,
    load_smith2024_wasp77ab_template,
)
from robert_exoplanets.likelihoods import (  # noqa: E402
    TimeResolvedHighResolutionLikelihood,
)


FloatArray = NDArray[np.float64]
MAX_MEMORY_BYTES: Final = 2 * 1024**3
DEFAULT_MAX_MEMORY_BYTES: Final = 1900 * 1024**2
DEFAULT_DATA_ROOT: Final = ROOT / "external_data" / "wasp77ab_smith2024"
DEFAULT_REPORT: Final = (
    ROOT / "examples" / "outputs" / "wasp77ab_smith2024_hrs_validation.json"
)
DEFAULT_PLOT: Final = (
    ROOT / "examples" / "outputs" / "wasp77ab_smith2024_hrs_validation.png"
)
CUBE_FILENAME: Final = "cube14v3.pic"
INFO_FILENAME: Final = "20201214_info.csv"
TEMPLATE_FILENAMES: Final[Mapping[str, str]] = {
    "full": "w77_1DRC_FULL.txt",
    "h2o_only": "w77_1DRC_H2O_ONLY.txt",
    "co_only": "w77_1DRC_CO_ONLY.txt",
}
PAPER_CCF_SNR: Final[Mapping[str, float]] = {
    "full": 9.4,
    "h2o_only": 9.2,
    "co_only": 3.6,
}
PAPER_URL: Final = "https://arxiv.org/abs/2312.13069"
PAPER_DOI: Final = "10.3847/1538-3881/ad17bf"
ZENODO_URL: Final = "https://zenodo.org/records/10382053"
EXPECTED_CUBE_SHAPE: Final[tuple[int, int, int]] = (44, 79, 1848)
PAPER_PRE_PARAMETERS: Final[Mapping[str, float]] = {
    # Smith et al. (2024), Table 3, pre-eclipse column.  Kp is the
    # Table-1 nominal 192 km/s plus the retrieved dKp=-1.26 km/s.
    "Kp": 190.74,
    "dVsys": -5.27,
    "dphi": 0.0,
    "log10_a": 0.15,
}
LOCAL_SCAN_KP: Final[tuple[float, ...]] = (190.74,)
LOCAL_SCAN_DVSYS: Final[tuple[float, ...]] = (-7.27, -5.27, -3.27)
HRS_TEMPLATE_RESOLVING_POWER: Final = 250_000.0
HRS_ROTATION_VSIN_KM_S: Final = 4.2
HRS_ROTATION_LIMB_DARKENING: Final = 0.6
HRS_GAUSSIAN_RESOLVING_POWER: Final = 45_000.0
HRS_GAUSSIAN_KERNEL_SUPPORT_SIGMA: Final = 4.0
HRS_PRIOR_KP_BOUNDS_KM_S: Final[tuple[float, float]] = (172.0, 212.0)
HRS_PRIOR_DVSYS_BOUNDS_KM_S: Final[tuple[float, float]] = (-20.0, 20.0)
HRS_CROP_SAFETY_MARGIN_KM_S: Final = 5.0
HRS_CROP_EDGE_PADDING_POINTS: Final = 8
HRS_RESPONSE_STAGES: Final[tuple[str, str]] = (
    "rotational-broadening",
    "gaussian-high-resolution",
)
HRS_RESPONSE_PIXEL_INTEGRATION: Final = False
_SPEED_OF_LIGHT_KM_S: Final = 299_792.458
_GAUSSIAN_FWHM_TO_SIGMA: Final = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


_FILE_SPECS: Final[Mapping[str, Mapping[str, object]]] = {
    CUBE_FILENAME: {
        "role": "Smith pre-eclipse IGRINS cube",
        "size": SMITH2024_WASP77AB_CUBE14_SIZE,
        "md5": SMITH2024_WASP77AB_CUBE14_MD5,
        "sha256": SMITH2024_WASP77AB_CUBE14_SHA256,
    },
    INFO_FILENAME: {
        "role": "Smith pre-eclipse frame metadata",
        "size": SMITH2024_WASP77AB_INFO_SIZE,
        "md5": SMITH2024_WASP77AB_INFO_MD5,
        "sha256": SMITH2024_WASP77AB_INFO_SHA256,
    },
    TEMPLATE_FILENAMES["full"]: {
        "role": "Smith full atmospheric template",
        "size": SMITH2024_WASP77AB_TEMPLATE_FULL_SIZE,
        "md5": SMITH2024_WASP77AB_TEMPLATE_FULL_MD5,
        "sha256": SMITH2024_WASP77AB_TEMPLATE_FULL_SHA256,
    },
    TEMPLATE_FILENAMES["h2o_only"]: {
        "role": "Smith H2O-only atmospheric template",
        "size": SMITH2024_WASP77AB_TEMPLATE_H2O_SIZE,
        "md5": SMITH2024_WASP77AB_TEMPLATE_H2O_MD5,
        "sha256": "f68acda835e0af3e1747504d1e6243560610e5b98ad675917233859a17ace330",
    },
    TEMPLATE_FILENAMES["co_only"]: {
        "role": "Smith CO-only atmospheric template",
        "size": SMITH2024_WASP77AB_TEMPLATE_CO_SIZE,
        "md5": SMITH2024_WASP77AB_TEMPLATE_CO_MD5,
        "sha256": "b8c9fc1496b25d9b4030ac3fafac3623ac1a5e945cf0b2f3161e96d3e8e3c016",
    },
}


def _peak_rss_bytes() -> int:
    """Return peak resident memory in bytes on macOS and Linux."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _raw_ru_maxrss() -> int:
    """Return the platform-native ``ru_maxrss`` value for provenance."""

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _thread_values() -> dict[str, int]:
    """Return the numerical thread settings recorded in the report."""

    return {name: int(os.environ[name]) for name in THREAD_VARIABLES}


def _file_hashes(path: Path) -> tuple[int, str, str]:
    """Hash one file in bounded memory."""

    digest_md5 = md5()  # noqa: S324 - MD5 is an upstream file identifier.
    digest_sha256 = sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                digest_md5.update(block)
                digest_sha256.update(block)
    except OSError as error:
        raise FileNotFoundError(path) from error
    return size, digest_md5.hexdigest(), digest_sha256.hexdigest()


def _verify_file(path: Path, spec: Mapping[str, object]) -> dict[str, object]:
    """Verify size, MD5, and SHA-256 before a scientific loader reads a file."""

    if not path.is_file():
        raise FileNotFoundError(path)
    size, actual_md5, actual_sha256 = _file_hashes(path)
    expected_size = int(spec["size"])
    expected_md5 = str(spec["md5"])
    expected_sha256 = str(spec["sha256"])
    label = f"Smith input {path.name}"
    if size != expected_size:
        raise ValueError(f"{label} size mismatch: expected {expected_size}, got {size}")
    if actual_md5 != expected_md5:
        raise ValueError(
            f"{label} MD5 mismatch: expected {expected_md5}, got {actual_md5}"
        )
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return {
        "name": path.name,
        "role": str(spec["role"]),
        "path": str(path),
        "size_bytes": size,
        "md5": actual_md5,
        "sha256": actual_sha256,
    }


def _load_observation(data_root: Path) -> tuple[object, list[dict[str, object]]]:
    """Verify and load the cube and its frame metadata."""

    cube_path = data_root / CUBE_FILENAME
    info_path = data_root / INFO_FILENAME
    cube_record = _verify_file(cube_path, _FILE_SPECS[CUBE_FILENAME])
    info_record = _verify_file(info_path, _FILE_SPECS[INFO_FILENAME])
    # The package loader repeats the identity checks immediately before its
    # restricted protocol-2 unpickler.  No unverified pickle is deserialised.
    observation = load_smith2024_wasp77ab_hrs(
        cube_path,
        info_path=info_path,
        verify_checksum=True,
        verify_sha256=True,
    )
    if observation.flux.shape != EXPECTED_CUBE_SHAPE:
        raise ValueError(
            f"unexpected Smith cube shape: {observation.flux.shape}; "
            f"expected {EXPECTED_CUBE_SHAPE}"
        )
    return observation, [cube_record, info_record]


def _load_template(
    data_root: Path,
    key: str,
) -> tuple[object, dict[str, object]]:
    """Verify and load one official template."""

    filename = TEMPLATE_FILENAMES[key]
    path = data_root / filename
    record = _verify_file(path, _FILE_SPECS[filename])
    # H2O and CO have strict workflow SHA checks above.  The package loader
    # has package SHA constants for the full product, so use its SHA check for
    # that product too.  It still repeats the MD5/size identity check.
    template = load_smith2024_wasp77ab_template(
        path,
        verify_checksum=True,
        verify_sha256=key == "full",
        flux_ratio_scale=1.0,
        name=f"Smith 2024 {key} template",
    )
    return template, record


def _velocity_coverage(
    observation: TimeResolvedHighResolutionObservation,
) -> dict[str, object]:
    """Return the Smith velocity range required by the HRS prior.

    The likelihood applies the orbital and fixed velocities while it
    interpolates a processed template onto each detector pixel.  Therefore
    template coverage must include every combination of the bounded Kp and
    systemic-velocity priors, as well as the Table-3 anchor used below.
    """

    phase = np.asarray(observation.phase, dtype=float)
    fixed = np.asarray(observation.fixed_velocity_km_s, dtype=float)
    sine = np.sin(2.0 * np.pi * phase)
    prior_values = np.concatenate(
        tuple(
            fixed
            + float(dVsys)
            + float(kp) * sine
            for kp in HRS_PRIOR_KP_BOUNDS_KM_S
            for dVsys in HRS_PRIOR_DVSYS_BOUNDS_KM_S
        )
    )
    anchor_values = (
        fixed
        + float(PAPER_PRE_PARAMETERS["dVsys"])
        + float(PAPER_PRE_PARAMETERS["Kp"]) * sine
    )
    return {
        "kp_prior_bounds_km_s": list(HRS_PRIOR_KP_BOUNDS_KM_S),
        "dVsys_prior_bounds_km_s": list(HRS_PRIOR_DVSYS_BOUNDS_KM_S),
        "dphi_fixed": float(PAPER_PRE_PARAMETERS["dphi"]),
        "prior_velocity_bounds_km_s": [
            float(np.min(prior_values)),
            float(np.max(prior_values)),
        ],
        "anchor_velocity_bounds_km_s": [
            float(np.min(anchor_values)),
            float(np.max(anchor_values)),
        ],
        "anchor_parameters": dict(PAPER_PRE_PARAMETERS),
    }


def _query_wavelength_bounds(
    observation: TimeResolvedHighResolutionObservation,
    velocity_coverage: Mapping[str, object],
) -> tuple[float, float]:
    """Return rest-frame wavelengths queried by all real pixels and priors."""

    observed_lower = float(np.min(observation.order_wavelengths))
    observed_upper = float(np.max(observation.order_wavelengths))
    velocity_lower, velocity_upper = (
        float(value)
        for value in velocity_coverage["prior_velocity_bounds_km_s"]  # type: ignore[index]
    )
    factors = np.asarray(
        (
            1.0 - velocity_lower / _SPEED_OF_LIGHT_KM_S,
            1.0 - velocity_upper / _SPEED_OF_LIGHT_KM_S,
        ),
        dtype=float,
    )
    return float(observed_lower * np.min(factors)), float(observed_upper * np.max(factors))


def _response_source_bounds(
    query_bounds: tuple[float, float],
) -> tuple[float, float]:
    """Expand query wavelengths by both fixed response kernels."""

    query_lower, query_upper = query_bounds
    rotation_half_width = HRS_ROTATION_VSIN_KM_S / _SPEED_OF_LIGHT_KM_S
    gaussian_fraction = (
        HRS_GAUSSIAN_KERNEL_SUPPORT_SIGMA
        * _GAUSSIAN_FWHM_TO_SIGMA
        / HRS_GAUSSIAN_RESOLVING_POWER
    )
    safety_factor = np.exp(HRS_CROP_SAFETY_MARGIN_KM_S / _SPEED_OF_LIGHT_KM_S)
    source_lower = (
        query_lower
        * (1.0 - gaussian_fraction)
        * np.exp(-rotation_half_width)
        / safety_factor
    )
    source_upper = (
        query_upper
        * (1.0 + gaussian_fraction)
        * np.exp(rotation_half_width)
        * safety_factor
    )
    if not np.isfinite(source_lower) or not np.isfinite(source_upper):
        raise ValueError("HRS response source bounds are not finite")
    if source_lower <= 0.0 or source_upper <= source_lower:
        raise ValueError("HRS response source bounds are invalid")
    return float(source_lower), float(source_upper)


def _crop_template_source(
    template: HighResolutionEmissionTemplate,
    source_bounds: tuple[float, float],
) -> tuple[SpectralGrid, FloatArray, FloatArray, dict[str, object]]:
    """Crop one raw template while retaining one support sample at each edge."""

    values = np.asarray(template.wavelength, dtype=float)
    planet = np.asarray(template.planet_flux, dtype=float)
    stellar = np.asarray(template.stellar_flux, dtype=float)
    descending = bool(values[0] > values[-1])
    if descending:
        values = values[::-1]
        planet = planet[::-1]
        stellar = stellar[::-1]
    lower, upper = source_bounds
    if lower < values[0] or upper > values[-1]:
        raise RobertCoverageError(
            "official Smith template does not cover the real cube and bounded HRS velocities"
        )
    start = max(
        0,
        int(np.searchsorted(values, lower, side="left"))
        - 1
        - HRS_CROP_EDGE_PADDING_POINTS,
    )
    stop = min(
        values.size,
        int(np.searchsorted(values, upper, side="right"))
        + 1
        + HRS_CROP_EDGE_PADDING_POINTS,
    )
    if stop - start < 2:
        raise RobertCoverageError("cropped Smith template has fewer than two source points")
    cropped_values = values[start:stop]
    cropped_planet = planet[start:stop]
    cropped_stellar = stellar[start:stop]
    grid = SpectralGrid(
        values=cropped_values,
        unit=template.wavelength_unit,
        name="smith-r250000-hrs-cropped-source",
        role="hrs_response_source",
        metadata={
            "source_resolution": f"R={HRS_TEMPLATE_RESOLVING_POWER:g}",
            "crop_reason": "real cube plus prior velocity and response support",
        },
    )
    return grid, cropped_planet, cropped_stellar, {
        "raw_orientation": "descending" if descending else "ascending",
        "source_bounds_requested_micron": [float(lower), float(upper)],
        "source_bounds_actual_micron": [
            float(cropped_values[0]),
            float(cropped_values[-1]),
        ],
        "source_points": int(cropped_values.size),
        "edge_padding_points": HRS_CROP_EDGE_PADDING_POINTS,
    }


def _flux_spectrum(grid: SpectralGrid, values: FloatArray, name: str) -> Spectrum:
    """Build a temporary surface-flux spectrum for one response stage."""

    return Spectrum(
        spectral_grid=grid,
        values=values,
        unit="surface_flux",
        observable="surface_flux",
        metadata={"source": name},
    )


def _prepare_template_for_hrs(
    template: HighResolutionEmissionTemplate,
    observation: TimeResolvedHighResolutionObservation,
) -> tuple[HighResolutionEmissionTemplate, dict[str, object]]:
    """Crop and convolve one official template before PCA interpolation.

    The response chain deliberately stops at the convolved high-resolution
    grid.  The Smith likelihood then performs its non-relativistic Doppler
    interpolation onto the real cube pixels.  No pixel-bin integration is
    used in this HRS validator.
    """

    if not isinstance(template, HighResolutionEmissionTemplate):
        raise TypeError("template must be a HighResolutionEmissionTemplate")
    if not isinstance(observation, TimeResolvedHighResolutionObservation):
        raise TypeError(
            "observation must be a TimeResolvedHighResolutionObservation"
        )
    if template.wavelength_unit != observation.wavelength_unit:
        raise ValueError("template and observation wavelength units must match")

    velocity_coverage = _velocity_coverage(observation)
    query_bounds = _query_wavelength_bounds(observation, velocity_coverage)
    source_bounds = _response_source_bounds(query_bounds)
    grid, planet_values, stellar_values, crop_metadata = _crop_template_source(
        template,
        source_bounds,
    )
    response_chain = HighResolutionResponseChain(
        stages=(
            RotationalBroadeningResponse(
                vsini_km_s=HRS_ROTATION_VSIN_KM_S,
                limb_darkening=HRS_ROTATION_LIMB_DARKENING,
            ),
            GaussianHighResolutionResponse(
                resolving_power=HRS_GAUSSIAN_RESOLVING_POWER,
                kernel_support=HRS_GAUSSIAN_KERNEL_SUPPORT_SIGMA,
            ),
        ),
        name="smith-gray-rotation-gaussian-lsf",
    )
    planet_spectrum = _flux_spectrum(grid, planet_values, "planet_surface_flux")
    stellar_spectrum = _flux_spectrum(grid, stellar_values, "stellar_surface_flux")
    working_grid = grid
    applied_stages: list[str] = []
    for stage in response_chain.stages:
        prepared_stage = stage.prepare_on_grid(working_grid)
        planet_spectrum = prepared_stage.observe(planet_spectrum)
        stellar_spectrum = prepared_stage.observe(stellar_spectrum)
        working_grid = prepared_stage.target_grid
        applied_stages.append(str(prepared_stage.name))

    processed_values = np.asarray(working_grid.values, dtype=float)
    processed_lower = float(np.min(processed_values))
    processed_upper = float(np.max(processed_values))
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, processed_upper)
    if processed_lower > query_bounds[0] + tolerance or processed_upper < query_bounds[1] - tolerance:
        raise RobertCoverageError(
            "convolved Smith template does not cover all bounded HRS Doppler queries"
        )
    response_metadata: dict[str, object] = {
        "raw_template_resolution": f"R={HRS_TEMPLATE_RESOLVING_POWER:g}",
        "stage_order": list(HRS_RESPONSE_STAGES),
        "applied_operator_names": applied_stages,
        "rotation_kernel": "Gray linear-limb-darkened rotational broadening",
        "rotation_vsini_km_s": HRS_ROTATION_VSIN_KM_S,
        "rotation_limb_darkening": HRS_ROTATION_LIMB_DARKENING,
        "gaussian_lsf_resolving_power": HRS_GAUSSIAN_RESOLVING_POWER,
        "gaussian_kernel_support_sigma": HRS_GAUSSIAN_KERNEL_SUPPORT_SIGMA,
        "pixel_integration": HRS_RESPONSE_PIXEL_INTEGRATION,
        "post_response_mapping": "Smith likelihood linear interpolation onto real cube pixels",
        "doppler_stage": "Smith likelihood interpolation after fixed response",
        "query_bounds_micron": [float(query_bounds[0]), float(query_bounds[1])],
        "processed_bounds_micron": [processed_lower, processed_upper],
        "processed_points": int(working_grid.size),
        "velocity_coverage": velocity_coverage,
        "crop": crop_metadata,
    }
    metadata = dict(template.metadata)
    metadata.update(
        {
            "response_chain": ",".join(HRS_RESPONSE_STAGES),
            "response_pixel_integration": "false",
            "response_mapping": "likelihood interpolation to real cube pixels",
            "response_template_resolution": f"R={HRS_TEMPLATE_RESOLVING_POWER:g}",
        }
    )
    processed_template = HighResolutionEmissionTemplate(
        wavelength=working_grid.values,
        planet_flux=planet_spectrum.values,
        stellar_flux=stellar_spectrum.values,
        wavelength_unit=template.wavelength_unit,
        name=f"{template.name}-gray-rotation-gaussian-lsf",
        flux_ratio_scale=template.flux_ratio_scale,
        metadata=metadata,
    )
    return processed_template, response_metadata


def _diagnostic(
    prepared: object,
    template: object,
    *,
    key: str,
    record: Mapping[str, object],
    response_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Return finite loglike and raw CCF diagnostics for one template."""

    # These two public calls each use the production order-streaming path.
    # They do not call evaluate_model or allocate complete model cubes.
    loglike = float(prepared.loglike(template, PAPER_PRE_PARAMETERS))
    ccf = np.asarray(
        prepared.cross_correlation(template, PAPER_PRE_PARAMETERS),
        dtype=float,
    )
    finite_ccf = ccf[np.isfinite(ccf)]
    if not np.isfinite(loglike) or finite_ccf.size == 0:
        raise ValueError(f"non-finite Smith {key} diagnostic")
    return {
        "name": key,
        "template": dict(record),
        "loglike": loglike,
        "ccf_sum": float(np.sum(finite_ccf)),
        "ccf_mean": float(np.mean(finite_ccf)),
        "ccf_min": float(np.min(finite_ccf)),
        "ccf_max": float(np.max(finite_ccf)),
        "ccf_finite_values": int(finite_ccf.size),
        "paper_ccf_snr_anchor": PAPER_CCF_SNR[key],
        "paper_ccf_snr_note": (
            "Paper value uses the full 2-D CCF-map median/3-sigma normalization; "
            "this local diagnostic is a raw summed CCF and does not reproduce it."
        ),
        "parameters": dict(PAPER_PRE_PARAMETERS),
        "parameter_source": "Smith et al. 2024 Table 3 pre-eclipse medians",
        "flux_ratio_scale": float(SMITH2024_WASP77AB_FLUX_RATIO_SCALE),
        "doppler_mode": "smith_nonrelativistic",
        "n_components": 4,
        "response": dict(response_metadata),
    }


def _velocity_scan(prepared: object, template: object) -> dict[str, object]:
    """Run one bounded three-point local dVsys scan at fixed Kp."""

    values = prepared.ccf_map(
        template,
        np.asarray(LOCAL_SCAN_KP, dtype=float),
        np.asarray(LOCAL_SCAN_DVSYS, dtype=float),
        parameters={"dphi": 0.0, "log10_a": 0.0},
    )
    return {
        "method": "three-point local dVsys scan at fixed Kp",
        "Kp_km_s": list(LOCAL_SCAN_KP),
        "dVsys_km_s": list(LOCAL_SCAN_DVSYS),
        "raw_summed_ccf": np.asarray(values, dtype=float).tolist(),
        "normalization": "raw sum of per-order/per-frame normalized CCF values",
        "paper_snr_reproduction": False,
    }


def _write_plot(path: Path, scan: Mapping[str, object]) -> str:
    """Write an optional, small local-velocity diagnostic plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    dVsys = np.asarray(scan["dVsys_km_s"], dtype=float)
    ccf = np.asarray(scan["raw_summed_ccf"], dtype=float)[0]
    figure, axis = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    axis.plot(dVsys, ccf, marker="o", color="#2f6f9f")
    axis.axvline(
        PAPER_PRE_PARAMETERS["dVsys"],
        color="0.4",
        linestyle="--",
        linewidth=0.8,
        label="Smith pre-eclipse median",
    )
    axis.set_xlabel("dVsys (km s$^{-1}$)")
    axis.set_ylabel("Raw summed CCF")
    axis.set_title("WASP-77Ab Smith HRS local diagnostic")
    axis.legend(frameon=False, fontsize=8)
    axis.text(
        0.02,
        0.03,
        "Not the paper's 2-D CCF SNR normalization",
        transform=axis.transAxes,
        fontsize=8,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return _file_hashes(path)[2]


def _write_report(path: Path, report: Mapping[str, object]) -> None:
    """Write one JSON report without non-finite values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _dry_run_report(*, max_memory_bytes: int, started: float) -> dict[str, object]:
    """Build a no-I/O report for CI and local command checks."""

    peak_rss = _peak_rss_bytes()
    thread_values = _thread_values()
    return {
        "schema_version": 1,
        "status": "dry_run",
        "scope": "Smith et al. 2024 WASP-77Ab HRS validation workflow",
        "claim": "No external data were loaded; this is a CLI and contract check.",
        "paper": {
            "arxiv": "2312.13069",
            "doi": PAPER_DOI,
            "url": PAPER_URL,
            "ccf_snr_anchors": dict(PAPER_CCF_SNR),
            "ccf_snr_note": (
                "The 9.4/9.2/3.6 values require the paper's full 2-D CCF-map "
                "normalization and are not claimed by a local raw CCF scan."
            ),
        },
        "response": {
            "stage_order": list(HRS_RESPONSE_STAGES),
            "operator_family": "ROBERT sparse spectral response operators",
            "template_resolving_power": HRS_TEMPLATE_RESOLVING_POWER,
            "rotation_vsini_km_s": HRS_ROTATION_VSIN_KM_S,
            "gaussian_lsf_resolving_power": HRS_GAUSSIAN_RESOLVING_POWER,
            "pixel_integration": HRS_RESPONSE_PIXEL_INTEGRATION,
            "post_response_mapping": "Smith likelihood interpolation onto real cube pixels",
        },
        "method": {
            "n_components": 4,
            "doppler_mode": "smith_nonrelativistic",
            "positive_velocity": "redshift",
            "flux_ratio_scale": float(SMITH2024_WASP77AB_FLUX_RATIO_SCALE),
            "sampler": "none",
            "abundance_convention": "ROBERT volume mixing ratios only",
            "response_chain": list(HRS_RESPONSE_STAGES),
            "response_stage_order": "Gray rotation -> Gaussian LSF -> PCA likelihood interpolation",
            "response_template_resolving_power": HRS_TEMPLATE_RESOLVING_POWER,
            "rotation_vsini_km_s": HRS_ROTATION_VSIN_KM_S,
            "gaussian_lsf_resolving_power": HRS_GAUSSIAN_RESOLVING_POWER,
            "gaussian_kernel_support_sigma": HRS_GAUSSIAN_KERNEL_SUPPORT_SIGMA,
            "pixel_integration": HRS_RESPONSE_PIXEL_INTEGRATION,
            "pixel_mapping": "likelihood interpolation onto real cube pixels",
        },
        "resources": {
            "thread_values": thread_values,
            "thread_limit": 3,
            "ru_maxrss": _raw_ru_maxrss(),
            "ru_maxrss_units": "bytes on macOS; KiB on other platforms",
            "peak_rss_bytes": peak_rss,
            "process_memory_limit_bytes": int(max_memory_bytes),
            "strict_below_2_gib": peak_rss < max_memory_bytes < MAX_MEMORY_BYTES,
            "elapsed_seconds": perf_counter() - started,
        },
        "inputs": {"data_root": None, "loaded": False, "zenodo": ZENODO_URL},
    }


def run_validation(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    report_path: Path | None = DEFAULT_REPORT,
    plot_path: Path | None = DEFAULT_PLOT,
    max_memory_bytes: int = DEFAULT_MAX_MEMORY_BYTES,
    run_scan: bool = True,
    dry_run: bool = False,
) -> dict[str, object]:
    """Run the real-data check, or a no-I/O dry run for CI."""

    max_memory_bytes = int(max_memory_bytes)
    if not 0 < max_memory_bytes < MAX_MEMORY_BYTES:
        raise ValueError("max_memory_bytes must be positive and strictly below 2 GiB")
    started = perf_counter()
    if dry_run:
        report = _dry_run_report(
            max_memory_bytes=max_memory_bytes,
            started=started,
        )
        if report_path is not None:
            _write_report(report_path, report)
        return report

    data_root = Path(data_root).expanduser()
    observation, input_records = _load_observation(data_root)
    if observation.n_orders != 44 or observation.n_frames != 79 or observation.n_pixels != 1848:
        raise ValueError("Smith observation does not have the required 44x79x1848 shape")
    likelihood = TimeResolvedHighResolutionLikelihood(
        n_components=4,
        doppler_mode="smith_nonrelativistic",
        flux_ratio_scale=SMITH2024_WASP77AB_FLUX_RATIO_SCALE,
    )
    prepared = likelihood.prepare(observation)
    diagnostics: list[dict[str, object]] = []
    template_records: dict[str, dict[str, object]] = {}
    velocity_scan: dict[str, object] | None = None
    for key in TEMPLATE_FILENAMES:
        template_path = data_root / TEMPLATE_FILENAMES[key]
        if not template_path.is_file():
            if key == "full":
                raise FileNotFoundError(template_path)
            continue
        raw_template, record = _load_template(data_root, key)
        template_records[key] = record
        response_template, response_metadata = _prepare_template_for_hrs(
            raw_template,
            observation,
        )
        diagnostics.append(
            _diagnostic(
                prepared,
                response_template,
                key=key,
                record=record,
                response_metadata=response_metadata,
            )
        )
        if key == "full" and run_scan:
            velocity_scan = _velocity_scan(prepared, response_template)
        del response_template, raw_template

    peak_rss = _peak_rss_bytes()
    thread_values = _thread_values()
    memory_pass = 0 < peak_rss < max_memory_bytes < MAX_MEMORY_BYTES
    cpu_pass = all(1 <= value <= 3 for value in thread_values.values())
    diagnostics_pass = "full" in {record["name"] for record in diagnostics}
    acceptance = {
        "full_template_present": diagnostics_pass,
        "finite_diagnostics": all(
            np.isfinite(float(record["loglike"]))
            and np.isfinite(float(record["ccf_sum"]))
            for record in diagnostics
        ),
        "cpu_threads_at_most_3": cpu_pass,
        "memory_strictly_below_2_gib": memory_pass,
    }
    plot_record: dict[str, object] | None = None
    if plot_path is not None and velocity_scan is not None:
        plot_sha256 = _write_plot(plot_path, velocity_scan)
        plot_record = {"path": str(plot_path), "sha256": plot_sha256}
        peak_rss = _peak_rss_bytes()
        memory_pass = 0 < peak_rss < max_memory_bytes < MAX_MEMORY_BYTES
        acceptance["memory_strictly_below_2_gib"] = memory_pass

    report: dict[str, object] = {
        "schema_version": 1,
        "status": "pass" if all(acceptance.values()) else "fail",
        "scope": "deterministic Smith et al. 2024 pre-eclipse IGRINS HRS diagnostic",
        "claim": (
            "Independent ROBERT evaluation of public Smith data and method; "
            "not a sampler or full literature reproduction."
        ),
        "paper": {
            "arxiv": "2312.13069",
            "doi": PAPER_DOI,
            "url": PAPER_URL,
            "ccf_snr_anchors": dict(PAPER_CCF_SNR),
            "ccf_snr_note": (
                "The paper's 9.4/9.2/3.6 pre-eclipse CCF SNR values use its full "
                "2-D map median/3-sigma normalization. The local scan reports raw "
                "summed CCF values and does not claim to reproduce those SNRs."
            ),
        },
        "response": {
            "stage_order": list(HRS_RESPONSE_STAGES),
            "operator_family": "ROBERT sparse spectral response operators",
            "template_resolving_power": HRS_TEMPLATE_RESOLVING_POWER,
            "rotation_vsini_km_s": HRS_ROTATION_VSIN_KM_S,
            "gaussian_lsf_resolving_power": HRS_GAUSSIAN_RESOLVING_POWER,
            "gaussian_kernel_support_sigma": HRS_GAUSSIAN_KERNEL_SUPPORT_SIGMA,
            "pixel_integration": HRS_RESPONSE_PIXEL_INTEGRATION,
            "post_response_mapping": "Smith likelihood interpolation onto real cube pixels",
            "coverage_source": "per-template response metadata",
        },
        "zenodo": {"url": ZENODO_URL, "files_verified_before_load": True},
        "observation": {
            "shape": list(observation.flux.shape),
            "n_orders": observation.n_orders,
            "n_frames": observation.n_frames,
            "n_pixels": observation.n_pixels,
            "active_samples": int(observation.n_points),
            "phase_min": float(np.min(observation.phase)),
            "phase_max": float(np.max(observation.phase)),
            "fixed_velocity_min_km_s": float(np.min(observation.fixed_velocity_km_s)),
            "fixed_velocity_max_km_s": float(np.max(observation.fixed_velocity_km_s)),
            "clipped_samples": int(prepared.clipped_mask.sum()),
            "retained_samples_after_pca": int(prepared.effective_residual_rank),
            "metadata": dict(observation.metadata),
        },
        "method": {
            "n_components": 4,
            "sigma_clip": 3.0,
            "doppler_mode": "smith_nonrelativistic",
            "positive_velocity": "redshift",
            "flux_ratio_scale": float(SMITH2024_WASP77AB_FLUX_RATIO_SCALE),
            "surface_flux_ratio_geometry": "(Rp/Rs)^2",
            "sampler": "none",
            "abundance_convention": "ROBERT volume mixing ratios only",
            "scale_parameter": "log10_a",
            "evaluation_parameters": dict(PAPER_PRE_PARAMETERS),
            "evaluation_parameter_source": (
                "Smith et al. 2024 Table 3 pre-eclipse medians; "
                "Kp=192+dKp"
            ),
            "response_chain": list(HRS_RESPONSE_STAGES),
            "response_stage_order": "Gray rotation -> Gaussian LSF -> PCA likelihood interpolation",
            "response_template_resolving_power": HRS_TEMPLATE_RESOLVING_POWER,
            "rotation_vsini_km_s": HRS_ROTATION_VSIN_KM_S,
            "gaussian_lsf_resolving_power": HRS_GAUSSIAN_RESOLVING_POWER,
            "gaussian_kernel_support_sigma": HRS_GAUSSIAN_KERNEL_SUPPORT_SIGMA,
            "pixel_integration": HRS_RESPONSE_PIXEL_INTEGRATION,
            "pixel_mapping": "likelihood interpolation onto real cube pixels",
            "data_model_memory_path": "one order at a time for scalar diagnostics",
        },
        "inputs": input_records + list(template_records.values()),
        "optional_templates_missing": [
            key for key in ("h2o_only", "co_only") if key not in template_records
        ],
        "templates": diagnostics,
        "velocity_scan": velocity_scan,
        "acceptance": acceptance,
        "resources": {
            "thread_values": thread_values,
            "thread_limit": 3,
            "ru_maxrss": _raw_ru_maxrss(),
            "ru_maxrss_units": "bytes on macOS; KiB on other platforms",
            "peak_rss_bytes": peak_rss,
            "process_memory_limit_bytes": int(max_memory_bytes),
            "strict_below_2_gib": memory_pass,
            "elapsed_seconds": perf_counter() - started,
        },
        "plot": plot_record,
    }
    if report_path is not None:
        _write_report(report_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-scan", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-memory-gib",
        type=float,
        default=DEFAULT_MAX_MEMORY_BYTES / 1024**3,
        help="Strict RSS limit. The value must remain below 2 GiB.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    max_memory_bytes = int(args.max_memory_gib * 1024**3)
    report = run_validation(
        data_root=args.data_root,
        report_path=args.report,
        plot_path=None if args.no_plot else args.plot,
        max_memory_bytes=max_memory_bytes,
        run_scan=not args.no_scan,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if report["status"] == "fail":
        raise SystemExit("Smith et al. WASP-77Ab HRS validation failed")


if __name__ == "__main__":
    main()
