"""Validate the real WASP-77Ab ROBERT operators with synthetic data.

This example is an operator-level injection test.  It is not a retrieval and
it does not call a sampler.  The real Smith et al. (2024) IGRINS order grids,
phases, fixed velocities, and PCA scale are used to make a synthetic HRS
cube.  The real August et al. (2023) NIRSpec NRS1/NRS2 bin grids are used for
the LRS observations.

The HRS generative equation is explicit::

    y_syn[o, t, p] = (1 + s_hrs * s_template * (Fp/Fs)[o, t, p]) * D[o, t, p]
                     + sigma[o] * epsilon[o, t, p]

where ``D`` is the prepared Smith low-rank ``data_scale``, ``Fp/Fs`` is the
Doppler-sampled ROBERT planet/star ratio, and ``epsilon`` is an independent
standard-normal detector-noise realization.  ``sigma[o]`` is the RMS of the
prepared Smith PCA residual for order ``o`` over its valid pixels.  The
post-PCA Smith scale ``s_hrs`` is neutral (``log10_a_hrs=0``) for this fixed
physical injection; this prevents the raw injection and the likelihood scale
from representing different amplitudes.  The model is then PCA-reprocessed
by a newly prepared likelihood, so this test checks the full injection
operator and not a shortcut in filtered space.

The LRS generative equation is::

    y_syn[i] = y_model[i] + sigma[i] * epsilon[i],
    epsilon[i] ~ Normal(0, 1),

with one fixed NumPy generator seed.  Only scalar diagnostics and source/grid
identities are written to the JSON report.  No synthetic spectrum is saved.

The default command is a dry run.  ``--run-real`` loads the external Smith,
NIRSpec, CIA, and LBL inputs and evaluates bounded truth/null/off-velocity
checks for HRS, LRS, joint, and H-minus on/off models.  It never starts a
posterior sampler.  ROBERT composition remains VMR-only.  All numerical
thread controls are clamped to at most six and the process limit is below
2 GiB.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
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
MAX_THREADS: Final = 6
LRS_NOISE_SEED: Final = 20260831
MAX_MEMORY_BYTES: Final = int(1.9 * 1024**3)
HARD_MEMORY_LIMIT_BYTES: Final = 2 * 1024**3
DEFAULT_REPORT_PATH: Final = (
    ROOT / "docs" / "data" / "wasp77ab_robert_injection_20260831.json"
)

HRS_MODE: Final = "hrs_only"
LRS_MODE: Final = "lrs_only"
JOINT_MODE: Final = "joint"
ALL_MODES: Final = (HRS_MODE, LRS_MODE, JOINT_MODE)
OFF_VELOCITY_DELTA_KM_S: Final = 13.0


def _clamp_thread_environment() -> None:
    """Clamp numerical thread controls before importing NumPy."""

    for name in THREAD_VARIABLES:
        raw = os.environ.get(name, str(MAX_THREADS))
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = MAX_THREADS
        os.environ[name] = str(min(MAX_THREADS, max(1, value)))


_clamp_thread_environment()
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "robert-matplotlib"),
)

import numpy as np  # noqa: E402
from numpy.typing import ArrayLike  # noqa: E402

from examples import run_wasp77ab_robert_joint_pymultinest as robert_workflow  # noqa: E402
from robert_exoplanets import (  # noqa: E402
    GaussianLikelihood,
    HeterogeneousLikelihood,
    HeterogeneousLikelihoodComponent,
    HeterogeneousRetrievalProblem,
    HighResolutionEmissionTemplate,
    Observation,
    ObservationCollection,
    ObservationDataset,
    Spectrum,
    TimeResolvedHighResolutionLikelihood,
    TimeResolvedHighResolutionObservation,
)
from robert_exoplanets.core import RobertValidationError  # noqa: E402


TruthParameters = dict[str, float]


DEFAULT_TRUTH_PARAMETERS: Final[Mapping[str, float]] = {
    "temperature_K": 2700.0,
    "log10_H2O_VMR": -4.02,
    "log10_CO_VMR": -3.91,
    "log10_Hminus_VMR": -8.0,
    "log10_electron_VMR": -3.0,
    "Kp": 190.74,
    "dVsys": -5.27,
    # The raw synthetic HRS injection uses the physical ROBERT ratio.  A
    # neutral Smith post-PCA amplitude is therefore the consistent truth.
    "log10_a_hrs": 0.0,
    "lrs_scale": 1.0,
}

HRS_NOISE_SEED: Final = 20260832
HRS_GENERATIVE_EQUATION: Final = (
    "y_syn[o,t,p] = (1 + s_hrs * template_flux_ratio_scale * "
    "(Fp/Fs)[o,t,p]) * D[o,t,p] + sigma[o] * epsilon[o,t,p], "
    "where D is the prepared Smith low-rank data_scale, sigma[o] is the "
    "RMS of the prepared valid Smith data_residual in order o, epsilon is "
    "independent Normal(0,1), and s_hrs=10**log10_a_hrs=1 for the neutral "
    "injection truth"
)
LRS_GENERATIVE_EQUATION: Final = (
    "y_syn[i] = y_model[i] + sigma[i] * epsilon[i], "
    "epsilon[i] ~ Normal(0,1), with fixed NumPy seed 20260831"
)


def _array_sha256(values: ArrayLike) -> str:
    """Hash a small coordinate array without serialising it to the report."""

    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return sha256(array.tobytes()).hexdigest()


def _truth_parameters(problem: HeterogeneousRetrievalProblem) -> TruthParameters:
    """Select the fixed injection state required by one problem mode."""

    missing = [name for name in problem.parameter_names if name not in DEFAULT_TRUTH_PARAMETERS]
    if missing:
        raise RobertValidationError(
            "injection truth has no value for parameters: " + ", ".join(missing)
        )
    return {
        name: float(DEFAULT_TRUTH_PARAMETERS[name])
        for name in problem.parameter_names
    }


def _prepared_hrs_likelihood(problem: HeterogeneousRetrievalProblem) -> object:
    """Return the prepared Smith HRS likelihood from a joint problem."""

    for component in problem.likelihood.components:
        if component.name == "hrs":
            likelihood = component.likelihood
            if not callable(getattr(likelihood, "evaluate_model", None)):
                raise RobertValidationError("HRS component is not a prepared Smith likelihood")
            if not hasattr(likelihood, "data_scale"):
                raise RobertValidationError("prepared HRS likelihood has no data_scale")
            return likelihood
    raise RobertValidationError("joint problem has no HRS component")


def generate_synthetic_hrs_observation(
    observation: TimeResolvedHighResolutionObservation,
    prepared_likelihood: object,
    template: HighResolutionEmissionTemplate,
    parameters: Mapping[str, float],
    *,
    seed: int = HRS_NOISE_SEED,
) -> TimeResolvedHighResolutionObservation:
    """Generate a noisy raw HRS cube with the exact Smith injection operator.

    ``prepared_likelihood.evaluate_model`` is the public Smith operator.  Its
    velocity and planet/star-ratio arrays are used with the prepared
    ``data_scale`` to form the raw pre-PCA cube.  Noise is drawn one order at
    a time, so this diagnostic does not retain a second full noise cube.
    """

    evaluator = getattr(prepared_likelihood, "evaluate_model", None)
    data_scale = getattr(prepared_likelihood, "data_scale", None)
    data_residual = getattr(prepared_likelihood, "data_residual", None)
    observation_mask = getattr(prepared_likelihood, "observation_mask", None)
    if not callable(evaluator) or data_scale is None or data_residual is None:
        raise TypeError(
            "prepared_likelihood must expose evaluate_model, data_scale, and data_residual"
        )
    if observation_mask is None:
        raise TypeError("prepared_likelihood must expose observation_mask")
    try:
        normalized_seed = int(seed)
    except (TypeError, ValueError) as error:
        raise ValueError("HRS noise seed must be an integer") from error
    if normalized_seed < 0:
        raise ValueError("HRS noise seed must be non-negative")
    result = evaluator(template, parameters)
    ratio = np.asarray(getattr(result, "flux_ratio"), dtype=float)
    model_scale = float(getattr(result, "scale"))
    if not np.isfinite(model_scale) or model_scale <= 0.0:
        raise RobertValidationError("HRS injection scale must be finite and positive")
    expected_shape = observation.flux.shape
    if ratio.shape != expected_shape:
        raise RobertValidationError(
            "synthetic HRS ratio shape does not match the real observation: "
            f"{ratio.shape} != {expected_shape}"
        )
    # Keep only the public ratio/scale outputs.  The other diagnostic cubes in
    # ``TimeResolvedHighResolutionModel`` are not needed for generation and
    # would overlap the raw cube during the memory-sensitive real run.
    del result
    scale_array = np.asarray(data_scale, dtype=float)
    residual_array = np.asarray(data_residual, dtype=float)
    mask_array = np.asarray(observation_mask, dtype=bool)
    if scale_array.shape != expected_shape:
        raise RobertValidationError("prepared HRS data_scale has the wrong shape")
    if residual_array.shape != expected_shape:
        raise RobertValidationError("prepared HRS data_residual has the wrong shape")
    if mask_array.shape != expected_shape:
        raise RobertValidationError("prepared HRS observation_mask has the wrong shape")
    if not np.all(np.isfinite(ratio)):
        raise RobertValidationError("synthetic HRS flux ratio is not finite")
    if not np.all(np.isfinite(scale_array)):
        raise RobertValidationError("prepared HRS data_scale is not finite")
    if not np.all(np.isfinite(residual_array)):
        raise RobertValidationError("prepared HRS data_residual is not finite")

    # ``result.scale`` is the Smith nuisance amplitude applied after PCA.  A
    # synthetic raw injection must carry the same amplitude in its planet
    # ratio.  The production truth uses the neutral value (log10_a_hrs=0),
    # but retaining this explicit factor prevents a silent amplitude mismatch
    # if a caller supplies a different parameter.
    raw_cube = np.array(
        (1.0 + model_scale * ratio) * scale_array,
        dtype=float,
        copy=True,
    )
    if not np.all(np.isfinite(raw_cube)):
        raise RobertValidationError("synthetic HRS cube is not finite")

    order_scales = np.empty(observation.n_orders, dtype=float)
    for order in range(observation.n_orders):
        valid_values = residual_array[order][mask_array[order]]
        if valid_values.size == 0:
            raise RobertValidationError(
                f"HRS order {order} has no valid pixels for noise calibration"
            )
        rms = float(np.sqrt(np.mean(np.square(valid_values))))
        if not np.isfinite(rms) or rms <= 0.0:
            raise RobertValidationError(
                f"HRS order {order} has no finite positive residual RMS"
            )
        order_scales[order] = rms

    # Draw and add one order at a time.  This keeps the noise temporary at
    # O(n_frames*n_pixels), instead of retaining another full HRS cube.
    generator = np.random.default_rng(normalized_seed)
    for order, rms in enumerate(order_scales):
        noise = generator.standard_normal(raw_cube[order].shape) * rms
        raw_cube[order] += noise
    del generator
    if not np.all(np.isfinite(raw_cube)):
        raise RobertValidationError("synthetic HRS cube with noise is not finite")

    order_scale_summary = json.dumps(
        [
            {"order": int(order), "sigma": float(rms)}
            for order, rms in enumerate(order_scales)
        ],
        separators=(",", ":"),
    )
    return TimeResolvedHighResolutionObservation.from_arrays(
        observation.order_wavelengths,
        raw_cube,
        observation.phase,
        observation.fixed_velocity_km_s,
        observation.time_bjd,
        mask=observation.mask,
        order_mask=observation.order_mask,
        frame_mask=observation.frame_mask,
        airmass=observation.airmass,
        humidity_percent=observation.humidity_percent,
        median_snr=observation.median_snr,
        wavelength_unit=observation.wavelength_unit,
        flux_unit=observation.flux_unit,
        observable=observation.observable,
        instrument=observation.instrument,
        name=f"{observation.name} synthetic operator injection",
        metadata={
            **dict(observation.metadata),
            "synthetic_injection": "true",
            "synthetic_injection_claim": "operator validation only; not posterior recovery",
            "synthetic_generation_equation": HRS_GENERATIVE_EQUATION,
            "synthetic_noise": "fixed-seed independent Gaussian detector noise",
            "synthetic_noise_seed": str(normalized_seed),
            "synthetic_noise_scale_definition": (
                "per-order RMS of prepared Smith data_residual over valid pixels"
            ),
            "synthetic_noise_scale_by_order": order_scale_summary,
            "synthetic_noise_scale_sha256": _array_sha256(order_scales),
            "synthetic_noise_chunking": "one order at a time",
            "synthetic_hrs_scale": repr(model_scale),
            "source_data_scale": "prepared Smith low-rank PCA scale",
            "source_data_residual": "prepared Smith clipped PCA residual",
        },
    )


def _hrs_noise_calibration_record(
    observation: TimeResolvedHighResolutionObservation,
    *,
    expected_n_orders: int | None = None,
) -> dict[str, object]:
    """Extract and validate the structured HRS noise calibration metadata."""

    metadata = dict(observation.metadata)
    raw_summary = metadata.get("synthetic_noise_scale_by_order")
    if not isinstance(raw_summary, str):
        raise RobertValidationError(
            "synthetic HRS observation has no per-order noise calibration"
        )
    try:
        entries = json.loads(raw_summary)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RobertValidationError(
            "synthetic HRS noise calibration is not valid JSON"
        ) from error
    if not isinstance(entries, list) or not entries:
        raise RobertValidationError("synthetic HRS noise calibration must be a non-empty list")
    sigmas: list[float] = []
    for expected_order, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RobertValidationError("synthetic HRS noise calibration entries must be objects")
        try:
            order = int(entry["order"])
            sigma = float(entry["sigma"])
        except (KeyError, TypeError, ValueError) as error:
            raise RobertValidationError(
                "synthetic HRS noise calibration entries need order and sigma"
            ) from error
        if order != expected_order:
            raise RobertValidationError(
                "synthetic HRS noise calibration order indices must be contiguous"
            )
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise RobertValidationError(
                "synthetic HRS noise calibration sigmas must be finite and positive"
            )
        sigmas.append(sigma)
    n_orders = len(sigmas)
    if expected_n_orders is not None and n_orders != int(expected_n_orders):
        raise RobertValidationError(
            "synthetic HRS noise calibration order count does not match the real "
            f"Smith grid: {n_orders} != {int(expected_n_orders)}"
        )
    try:
        seed = int(metadata["synthetic_noise_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise RobertValidationError(
            "synthetic HRS noise calibration has no integer seed"
        ) from error
    if seed < 0:
        raise RobertValidationError("synthetic HRS noise calibration seed must be non-negative")
    definition = metadata.get("synthetic_noise_scale_definition")
    chunking = metadata.get("synthetic_noise_chunking")
    if not isinstance(definition, str) or not definition:
        raise RobertValidationError("synthetic HRS noise calibration has no scale definition")
    if not isinstance(chunking, str) or not chunking:
        raise RobertValidationError("synthetic HRS noise calibration has no chunking record")
    sigma_array = np.asarray(sigmas, dtype=float)
    checksum = metadata.get("synthetic_noise_scale_sha256")
    if not isinstance(checksum, str) or checksum != _array_sha256(sigma_array):
        raise RobertValidationError("synthetic HRS noise calibration checksum is invalid")
    return {
        "seed": seed,
        "scale_definition": definition,
        "distribution": str(
            metadata.get(
                "synthetic_noise",
                "fixed-seed independent Gaussian detector noise",
            )
        ),
        "chunking": chunking,
        "n_orders": n_orders,
        "sigma_by_order": [float(value) for value in sigma_array],
        "sigma_sha256": checksum,
        "sigma_summary": {
            "min": float(np.min(sigma_array)),
            "max": float(np.max(sigma_array)),
            "mean": float(np.mean(sigma_array)),
            "rms": float(np.sqrt(np.mean(np.square(sigma_array)))),
        },
    }


def generate_synthetic_lrs_observations(
    observations: ObservationCollection,
    predictions: Mapping[str, object],
    *,
    seed: int = LRS_NOISE_SEED,
) -> ObservationCollection:
    """Add fixed-seed Gaussian noise to real NRS1/NRS2 model bins."""

    try:
        normalized_seed = int(seed)
    except (TypeError, ValueError) as error:
        raise ValueError("LRS noise seed must be an integer") from error
    if normalized_seed < 0:
        raise ValueError("LRS noise seed must be non-negative")
    generator = np.random.default_rng(normalized_seed)
    datasets: list[ObservationDataset] = []
    for dataset in observations.datasets:
        prediction = predictions.get(dataset.name)
        if not isinstance(prediction, Spectrum):
            raise RobertValidationError(
                f"LRS prediction for {dataset.name!r} must be a Spectrum"
            )
        observation = dataset.observation
        if prediction.spectral_grid.unit != observation.wavelength_unit:
            raise RobertValidationError(
                f"LRS prediction and {dataset.name!r} wavelength units differ"
            )
        if prediction.values.shape != observation.flux.shape or not np.allclose(
            prediction.spectral_grid.values,
            observation.wavelength,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise RobertValidationError(
                f"LRS prediction for {dataset.name!r} is not on its observation grid"
            )
        if prediction.unit != observation.flux_unit or prediction.observable != observation.observable:
            raise RobertValidationError(
                f"LRS prediction and {dataset.name!r} observable contracts differ"
            )
        noise = generator.standard_normal(observation.n_points)
        synthetic_flux = np.asarray(prediction.values, dtype=float) + (
            observation.uncertainty * noise
        )
        if not np.all(np.isfinite(synthetic_flux)):
            raise RobertValidationError("synthetic LRS flux is not finite")
        synthetic = Observation(
            wavelength=observation.wavelength,
            flux=synthetic_flux,
            uncertainty=observation.uncertainty,
            wavelength_unit=observation.wavelength_unit,
            flux_unit=observation.flux_unit,
            observable=observation.observable,
            instrument=observation.instrument,
            mask=observation.mask,
            wavelength_bin_edges=observation.wavelength_bin_edges,
            metadata={
                **dict(observation.metadata),
                "synthetic_injection": "true",
                "synthetic_noise_seed": str(normalized_seed),
                "synthetic_generation_equation": LRS_GENERATIVE_EQUATION,
                "synthetic_injection_claim": "operator validation only; not posterior recovery",
            },
        )
        datasets.append(
            ObservationDataset(
                name=dataset.name,
                observation=synthetic,
                offset_parameter=dataset.offset_parameter,
                jitter_parameter=dataset.jitter_parameter,
                uncertainty_scale_parameter=dataset.uncertainty_scale_parameter,
                uncertainty_scale=dataset.uncertainty_scale,
                metadata={
                    **dict(dataset.metadata),
                    "synthetic_injection": "true",
                    "synthetic_noise_seed": str(normalized_seed),
                },
            )
        )
    return ObservationCollection(
        tuple(datasets),
        name=f"{observations.name} synthetic operator injection",
        metadata={
            **dict(observations.metadata),
            "synthetic_injection": "true",
            "synthetic_noise_seed": str(normalized_seed),
            "synthetic_generation_equation": LRS_GENERATIVE_EQUATION,
        },
    )


def build_synthetic_problem(
    real_problem: HeterogeneousRetrievalProblem,
    synthetic_hrs: TimeResolvedHighResolutionObservation,
    synthetic_lrs: ObservationCollection,
    *,
    mode: str,
) -> HeterogeneousRetrievalProblem:
    """Rebuild HRS/LRS likelihoods on the synthetic observations."""

    if mode not in ALL_MODES:
        raise ValueError(f"mode must be one of {ALL_MODES}")
    components: list[HeterogeneousLikelihoodComponent] = []
    if mode in {HRS_MODE, JOINT_MODE}:
        source = next(
            (
                component.likelihood
                for component in real_problem.likelihood.components
                if component.name == "hrs"
            ),
            None,
        )
        if source is None:
            raise RobertValidationError("real problem has no HRS component")
        prepared = TimeResolvedHighResolutionLikelihood(
            n_components=int(getattr(source, "n_components", 4)),
            sigma_clip=float(getattr(source, "sigma_clip", 3.0)),
            kp_parameter=str(getattr(source, "kp_parameter", robert_workflow.HRS_KP_PARAMETER)),
            dVsys_parameter=str(getattr(source, "dVsys_parameter", robert_workflow.HRS_DVSYS_PARAMETER)),
            dphi_parameter=str(getattr(source, "dphi_parameter", "dphi_fixed_zero")),
            scale_parameter=str(getattr(source, "scale_parameter", robert_workflow.HRS_SCALE_PARAMETER)),
            scale_is_log10=bool(getattr(source, "scale_is_log10", True)),
            doppler_mode=str(getattr(source, "doppler_mode", "smith_nonrelativistic")),
            flux_ratio_scale=float(getattr(source, "flux_ratio_scale", 1.0)),
            invalid_model_loglike=float(getattr(source, "invalid_model_loglike", float("-inf"))),
            name="synthetic Smith operator injection HRS likelihood",
        ).prepare(synthetic_hrs)
        components.append(
            HeterogeneousLikelihoodComponent(
                name="hrs",
                prediction_key="hrs",
                likelihood=prepared,
                metadata={
                    "synthetic_data": "true",
                    "generative_equation": HRS_GENERATIVE_EQUATION,
                },
            )
        )
    if mode in {LRS_MODE, JOINT_MODE}:
        gaussian = GaussianLikelihood(
            include_normalization=False,
            offset_parameter=None,
            jitter_parameter=None,
            uncertainty_scale_parameter=None,
        )
        for dataset in synthetic_lrs.datasets:
            components.append(
                HeterogeneousLikelihoodComponent(
                    name=dataset.name,
                    prediction_key=dataset.name,
                    likelihood=gaussian,
                    observation=dataset.observation,
                    metadata={
                        "synthetic_data": "true",
                        "generative_equation": LRS_GENERATIVE_EQUATION,
                    },
                )
            )
    return HeterogeneousRetrievalProblem(
        name=f"{real_problem.name}-synthetic-operator-{mode}",
        parameters=real_problem.parameters,
        forward_model=real_problem.forward_model,
        likelihood=HeterogeneousLikelihood(
            tuple(components),
            name=f"synthetic operator {mode} likelihood",
        ),
        invalid_loglike=real_problem.invalid_loglike,
        metadata={
            **dict(real_problem.metadata),
            "synthetic_operator_validation": "true",
            "truth_recovery_claim": "false",
            "composition_convention": "volume_mixing_ratio",
            "mass_fraction_parameters": "none",
        },
        opacity_identifiers=real_problem.opacity_identifiers,
    )


def _zero_template(template: HighResolutionEmissionTemplate) -> HighResolutionEmissionTemplate:
    """Make a null template on the exact model grid."""

    return HighResolutionEmissionTemplate(
        wavelength=template.wavelength,
        planet_flux=np.zeros(template.wavelength.size, dtype=float),
        stellar_flux=np.ones(template.wavelength.size, dtype=float),
        wavelength_unit=template.wavelength_unit,
        name="synthetic operator flat HRS null",
        metadata={"synthetic_null": "true"},
    )


def _null_prediction(prediction: Mapping[str, object]) -> dict[str, object]:
    """Replace all model signals by finite null predictions."""

    output: dict[str, object] = {}
    for key, value in prediction.items():
        if isinstance(value, HighResolutionEmissionTemplate):
            output[key] = _zero_template(value)
        elif isinstance(value, Spectrum):
            output[key] = Spectrum(
                spectral_grid=value.spectral_grid,
                values=np.zeros_like(value.values),
                unit=value.unit,
                observable=value.observable,
                metadata={"synthetic_null": "true"},
            )
        else:
            output[key] = value
    return output


def _off_velocity_parameters(
    parameters: Mapping[str, float],
    *,
    mode: str,
) -> TruthParameters:
    """Move the HRS orbital solution while retaining the same atmosphere."""

    output = {str(key): float(value) for key, value in parameters.items()}
    if mode in {HRS_MODE, JOINT_MODE}:
        output[robert_workflow.HRS_KP_PARAMETER] = (
            output[robert_workflow.HRS_KP_PARAMETER] + OFF_VELOCITY_DELTA_KM_S
        )
    return output


def _finite_loglike(
    problem: HeterogeneousRetrievalProblem,
    prediction: Mapping[str, object],
    parameters: Mapping[str, float],
) -> float:
    """Return one scalar likelihood, converting an operator error to NaN."""

    try:
        value = float(problem.likelihood.loglike(prediction, parameters))
    except (RobertValidationError, ValueError, FloatingPointError, TypeError, OverflowError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def evaluate_truth_null_off_velocity(
    problem: HeterogeneousRetrievalProblem,
    truth_parameters: Mapping[str, float],
    *,
    mode: str,
) -> dict[str, object]:
    """Evaluate deterministic truth, flat-null, and off-velocity cases."""

    truth = {str(key): float(value) for key, value in truth_parameters.items()}
    prediction = problem.predict(truth)
    truth_loglike = _finite_loglike(problem, prediction, truth)
    null_loglike = _finite_loglike(problem, _null_prediction(prediction), truth)
    off_parameters = _off_velocity_parameters(truth, mode=mode)
    off_prediction = problem.predict(off_parameters)
    off_loglike = _finite_loglike(problem, off_prediction, off_parameters)
    values = {
        "truth_loglike": truth_loglike,
        "null_loglike": null_loglike,
        "off_velocity_loglike": off_loglike,
        "truth_minus_null": truth_loglike - null_loglike,
        "truth_minus_off_velocity": truth_loglike - off_loglike,
        "off_velocity_delta_km_s": (
            OFF_VELOCITY_DELTA_KM_S if mode in {HRS_MODE, JOINT_MODE} else 0.0
        ),
        "off_velocity_applicable": mode in {HRS_MODE, JOINT_MODE},
        "finite": bool(
            np.isfinite(truth_loglike)
            and np.isfinite(null_loglike)
            and np.isfinite(off_loglike)
        ),
        "truth_preferred_to_null": bool(
            np.isfinite(truth_loglike)
            and np.isfinite(null_loglike)
            and truth_loglike > null_loglike
        ),
        "truth_preferred_to_off_velocity": bool(
            np.isfinite(truth_loglike)
            and np.isfinite(off_loglike)
            and (
                mode not in {HRS_MODE, JOINT_MODE}
                or truth_loglike > off_loglike
            )
        ),
    }
    return values


def _grid_record(
    hrs: TimeResolvedHighResolutionObservation,
    lrs: ObservationCollection,
) -> dict[str, object]:
    """Record real coordinate identities without writing arrays."""

    return {
        "hrs": {
            "n_orders": hrs.n_orders,
            "n_frames": hrs.n_frames,
            "n_pixels_per_order": hrs.n_pixels,
            "n_active_points": hrs.n_points,
            "order_wavelength_sha256": _array_sha256(hrs.order_wavelengths),
            "phase_sha256": _array_sha256(hrs.phase),
            "fixed_velocity_sha256": _array_sha256(hrs.fixed_velocity_km_s),
            "wavelength_min_micron": float(np.min(hrs.order_wavelengths)),
            "wavelength_max_micron": float(np.max(hrs.order_wavelengths)),
        },
        "lrs": {
            dataset.name: {
                "n_points": dataset.observation.n_points,
                "wavelength_sha256": _array_sha256(dataset.observation.wavelength),
                "bin_edges_sha256": (
                    None
                    if dataset.observation.wavelength_bin_edges is None
                    else _array_sha256(dataset.observation.wavelength_bin_edges)
                ),
                "wavelength_min_micron": float(np.min(dataset.observation.wavelength)),
                "wavelength_max_micron": float(np.max(dataset.observation.wavelength)),
                "bin_edge_min_micron": (
                    None
                    if dataset.observation.wavelength_bin_edges is None
                    else float(np.min(dataset.observation.wavelength_bin_edges))
                ),
                "bin_edge_max_micron": (
                    None
                    if dataset.observation.wavelength_bin_edges is None
                    else float(np.max(dataset.observation.wavelength_bin_edges))
                ),
            }
            for dataset in lrs.datasets
        },
    }


def _resource_gate(max_memory_bytes: int) -> dict[str, object]:
    """Use the shared workflow resource contract and require a strict limit."""

    limit = int(max_memory_bytes)
    if not 0 < limit < HARD_MEMORY_LIMIT_BYTES:
        raise ValueError("max_memory_bytes must be positive and below 2 GiB")
    record = robert_workflow.resource_record(max_memory_bytes=limit)
    if not bool(record["gate_passed"]):
        raise MemoryError("thread, MPI, or process memory gate failed")
    return record


@dataclass(frozen=True)
class InjectionRunResult:
    """Compact result bundle used internally by the real validation path."""

    truth_parameters: Mapping[str, float]
    hrs_noise_calibration: Mapping[str, object]
    source_hashes: Mapping[str, Mapping[str, object]]
    grid_record: Mapping[str, object]
    on_checks: Mapping[str, Mapping[str, object]]
    off_checks: Mapping[str, Mapping[str, object]]
    resources: Mapping[str, object]


def _run_real_validation(
    *,
    smith_data_root: Path,
    nirspec_data_root: Path,
    h2o_table: Path,
    co_table: Path,
    stride_report: Path | None,
    stellar_spectrum_model: str,
    max_memory_bytes: int,
) -> InjectionRunResult:
    """Run the real-input deterministic checks, without any sampler."""

    _resource_gate(max_memory_bytes)
    inputs_on = robert_workflow.load_real_inputs(
        smith_data_root=smith_data_root,
        nirspec_data_root=nirspec_data_root,
        h2o_table=h2o_table,
        co_table=co_table,
        stride_report=stride_report,
        stellar_spectrum_model=stellar_spectrum_model,
        hminus_enabled=True,
    )
    joint_on = robert_workflow.build_real_problem(
        inputs_on,
        mode=JOINT_MODE,
        hminus_enabled=True,
    )
    truth = _truth_parameters(joint_on)
    prediction = joint_on.predict(truth)
    prepared = _prepared_hrs_likelihood(joint_on)
    synthetic_hrs = generate_synthetic_hrs_observation(
        inputs_on.hrs_observation,
        prepared,
        prediction["hrs"],
        truth,
    )
    synthetic_lrs = generate_synthetic_lrs_observations(
        inputs_on.nirspec_observations,
        prediction,
        seed=LRS_NOISE_SEED,
    )
    hrs_noise_calibration = _hrs_noise_calibration_record(
        synthetic_hrs,
        expected_n_orders=19,
    )
    truth_record = {
        str(name): float(value)
        for name, value in DEFAULT_TRUTH_PARAMETERS.items()
    }
    source_hashes = dict(inputs_on.source_hashes)
    grid_record = _grid_record(inputs_on.hrs_observation, inputs_on.nirspec_observations)

    on_checks: dict[str, Mapping[str, object]] = {}
    for mode in ALL_MODES:
        real_mode = robert_workflow.build_real_problem(
            inputs_on,
            mode=mode,
            hminus_enabled=True,
        )
        synthetic_mode = build_synthetic_problem(
            real_mode,
            synthetic_hrs,
            synthetic_lrs,
            mode=mode,
        )
        on_checks[mode] = evaluate_truth_null_off_velocity(
            synthetic_mode,
            truth,
            mode=mode,
        )
        del synthetic_mode, real_mode
    del joint_on, prediction, prepared, inputs_on
    gc.collect()

    # Load the H-minus-off model after releasing the H-minus-on forward model.
    # This keeps the two full prepared LBL branches out of memory at once.
    inputs_off = robert_workflow.load_real_inputs(
        smith_data_root=smith_data_root,
        nirspec_data_root=nirspec_data_root,
        h2o_table=h2o_table,
        co_table=co_table,
        stride_report=stride_report,
        stellar_spectrum_model=stellar_spectrum_model,
        hminus_enabled=False,
    )
    off_checks: dict[str, Mapping[str, object]] = {}
    for mode in ALL_MODES:
        real_mode = robert_workflow.build_real_problem(
            inputs_off,
            mode=mode,
            hminus_enabled=False,
        )
        synthetic_mode = build_synthetic_problem(
            real_mode,
            synthetic_hrs,
            synthetic_lrs,
            mode=mode,
        )
        off_checks[mode] = evaluate_truth_null_off_velocity(
            synthetic_mode,
            truth,
            mode=mode,
        )
        del synthetic_mode, real_mode
    del inputs_off, synthetic_hrs, synthetic_lrs
    gc.collect()
    resources = _resource_gate(max_memory_bytes)
    return InjectionRunResult(
        truth_parameters=truth_record,
        hrs_noise_calibration=hrs_noise_calibration,
        source_hashes=source_hashes,
        grid_record=grid_record,
        on_checks=on_checks,
        off_checks=off_checks,
        resources=resources,
    )


def run_validation(
    *,
    smith_data_root: Path = robert_workflow.DEFAULT_SMITH_DATA_ROOT,
    nirspec_data_root: Path = robert_workflow.DEFAULT_NIRSPEC_DATA_ROOT,
    h2o_table: Path = robert_workflow.DEFAULT_H2O_TABLE,
    co_table: Path = robert_workflow.DEFAULT_CO_TABLE,
    stride_report: Path | None = robert_workflow.DEFAULT_STRIDE_REPORT,
    stellar_spectrum_model: str = "phoenix",
    report_path: Path | None = None,
    run_real: bool = False,
    dry_run: bool = True,
    max_memory_bytes: int = MAX_MEMORY_BYTES,
) -> dict[str, object]:
    """Run a dry contract or the bounded real operator validation."""

    if run_real:
        dry_run = False
    resources = robert_workflow.resource_record(
        max_memory_bytes=int(max_memory_bytes),
        check_peak=not dry_run,
    )
    if not dry_run and not bool(resources["gate_passed"]):
        raise MemoryError("thread, MPI, or process memory gate failed before validation")

    report: dict[str, object] = {
        "schema_version": "1.0",
        "record_type": "wasp77ab_robert_operator_injection_validation",
        "status": "dry_run" if dry_run else "not_started",
        "claim": (
            "deterministic operator-level injection/null discrimination for "
            "real WASP-77Ab HRS/LRS grids; not posterior recovery"
        ),
        "truth_parameters": {
            str(name): float(value)
            for name, value in DEFAULT_TRUTH_PARAMETERS.items()
        },
        "sampler": {
            "used": False,
            "purpose": "none; deterministic truth/null/off-velocity checks only",
        },
        "composition_policy": {
            "robert_convention": "volume_mixing_ratio",
            "mass_fraction_parameters_in_robert": False,
            "shared_atmosphere": True,
            "hminus_species": ["H-", "H", "e-"],
        },
        "generative_model": {
            "hrs_equation": HRS_GENERATIVE_EQUATION,
            "hrs_noise_seed": HRS_NOISE_SEED,
            "hrs_noise_scale_definition": (
                "per-order RMS of prepared Smith data_residual over valid pixels"
            ),
            "hrs_noise_distribution": "independent Normal(0, sigma_order^2)",
            "hrs_noise_chunking": "one order at a time",
            "hrs_noise_calibration": None,
            "lrs_equation": LRS_GENERATIVE_EQUATION,
            "lrs_noise_seed": LRS_NOISE_SEED,
            "synthetic_spectra_written": False,
        },
        "checks": {
            "modes": list(ALL_MODES),
            "hminus_on_model": "physical H-minus BF/FF from VMR composition",
            "hminus_off_model": "H-minus continuum disabled; same VMR parameters retained",
            "real_data_truth_recovery_claim": False,
        },
        "resource_policy": {
            "thread_variables": list(THREAD_VARIABLES),
            "max_threads": MAX_THREADS,
            "process_memory_limit_bytes": int(max_memory_bytes),
            "hard_process_memory_limit_bytes": HARD_MEMORY_LIMIT_BYTES,
            "mpi_processes": 1,
        },
        "resources": resources,
        "real_data_loaded": False,
    }
    if not run_real:
        if report_path is not None:
            write_report(report, Path(report_path))
        return report

    result = _run_real_validation(
        smith_data_root=Path(smith_data_root),
        nirspec_data_root=Path(nirspec_data_root),
        h2o_table=Path(h2o_table),
        co_table=Path(co_table),
        stride_report=None if stride_report is None else Path(stride_report),
        stellar_spectrum_model=stellar_spectrum_model,
        max_memory_bytes=int(max_memory_bytes),
    )
    mode_records: dict[str, object] = {}
    mode_gates: list[bool] = []
    for mode in ALL_MODES:
        on = dict(result.on_checks[mode])
        off = dict(result.off_checks[mode])
        on_off_delta = float(on["truth_loglike"] - off["truth_loglike"])
        finite_on_off = bool(
            np.isfinite(float(on["truth_loglike"]))
            and np.isfinite(float(off["truth_loglike"]))
        )
        mode_gate = bool(
            on["finite"]
            and on["truth_preferred_to_null"]
            and on["truth_preferred_to_off_velocity"]
            and off["finite"]
            and finite_on_off
        )
        mode_gates.append(mode_gate)
        mode_records[mode] = {
            "hminus_on": on,
            "hminus_off": off,
            "hminus_loglike_on_minus_off": on_off_delta,
            "hminus_on_off_finite": finite_on_off,
            "gate_passed": mode_gate,
        }
    report.update(
        {
            "status": "pass" if all(mode_gates) and result.resources["gate_passed"] else "fail",
            "real_data_loaded": True,
            "truth_parameters": dict(result.truth_parameters),
            "source_hashes": result.source_hashes,
            "real_grids": result.grid_record,
            "mode_results": mode_records,
            "resources": result.resources,
            "generative_model": {
                **dict(report["generative_model"]),
                "hrs_noise_calibration": dict(result.hrs_noise_calibration),
            },
            "acceptance": {
                "all_modes_finite": all(bool(result.on_checks[mode]["finite"]) for mode in ALL_MODES),
                "truth_preferred_to_null": all(
                    bool(result.on_checks[mode]["truth_preferred_to_null"])
                    for mode in ALL_MODES
                ),
                "truth_preferred_to_off_velocity": all(
                    bool(result.on_checks[mode]["truth_preferred_to_off_velocity"])
                    for mode in ALL_MODES
                ),
                "hminus_on_off_evaluated": all(
                    bool(mode_records[mode]["hminus_on_off_finite"])
                    for mode in ALL_MODES
                ),
                "resource_gate": bool(result.resources["gate_passed"]),
                "posterior_recovery_claim": False,
            },
        }
    )
    if report_path is not None:
        write_report(report, Path(report_path))
    return report


def write_report(report: Mapping[str, object], path: Path) -> None:
    """Write one compact finite JSON report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-real", action="store_true", help="load external data and run deterministic checks")
    parser.add_argument("--smith-data-root", type=Path, default=robert_workflow.DEFAULT_SMITH_DATA_ROOT)
    parser.add_argument("--nirspec-data-root", type=Path, default=robert_workflow.DEFAULT_NIRSPEC_DATA_ROOT)
    parser.add_argument("--h2o-table", type=Path, default=robert_workflow.DEFAULT_H2O_TABLE)
    parser.add_argument("--co-table", type=Path, default=robert_workflow.DEFAULT_CO_TABLE)
    parser.add_argument("--stride-report", type=Path, default=robert_workflow.DEFAULT_STRIDE_REPORT)
    parser.add_argument("--stellar-spectrum-model", choices=("phoenix", "blackbody"), default="phoenix")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--max-memory-gib",
        type=float,
        default=MAX_MEMORY_BYTES / 1024**3,
        help="strict process RSS limit; must be below 2 GiB",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dry contract or explicit real operator test."""

    args = _parse_args(argv)
    report = run_validation(
        smith_data_root=args.smith_data_root,
        nirspec_data_root=args.nirspec_data_root,
        h2o_table=args.h2o_table,
        co_table=args.co_table,
        stride_report=args.stride_report,
        stellar_spectrum_model=args.stellar_spectrum_model,
        report_path=args.report,
        run_real=bool(args.run_real),
        dry_run=not bool(args.run_real),
        max_memory_bytes=int(float(args.max_memory_gib) * 1024**3),
    )
    print(
        json.dumps(
            {"status": report["status"], "real_data_loaded": report["real_data_loaded"]},
            sort_keys=True,
        )
    )
    return 0 if report["status"] in {"dry_run", "pass"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
