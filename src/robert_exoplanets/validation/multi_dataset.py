"""Deterministic injection and recovery helpers for named datasets."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike

from robert_exoplanets.core import RobertValidationError, SpectralGrid, Spectrum
from robert_exoplanets.instruments import (
    Observation,
    ObservationCollection,
    ObservationDataset,
)

from .injection import (
    InjectionRecoveryReport,
    evaluate_injection_recovery,
    inject_spectrum,
)


class EffectiveMultiDatasetLikelihood(Protocol):
    """Likelihood contract needed by combined optimal-estimation validation."""

    def effective_inputs_by_dataset(
        self,
        prediction: object,
        observations: ObservationCollection,
        parameters: Mapping[str, float] | None = None,
    ) -> Mapping[str, tuple[ArrayLike, ArrayLike, ArrayLike]]: ...


def _has_covariance_interface(likelihood: object) -> bool:
    """Return whether a multi-dataset likelihood needs an exact hook."""

    if any(
        callable(getattr(likelihood, attribute, None))
        or getattr(likelihood, attribute, None) is not None
        for attribute in (
            "effective_covariance",
            "effective_covariance_by_dataset",
            "covariance",
            "covariance_by_dataset",
        )
    ):
        return True
    children = getattr(likelihood, "likelihoods", None)
    if isinstance(children, Mapping):
        return any(_has_covariance_interface(child) for child in children.values())
    return False


def _validated_chi_square(value: object, dataset_name: str) -> float:
    """Validate one exact non-negative dataset chi-square."""

    if isinstance(value, (bool, np.bool_)):
        raise RobertValidationError(
            f"chi-square for dataset {dataset_name!r} must be finite and non-negative"
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RobertValidationError(
            f"chi-square for dataset {dataset_name!r} must be finite and non-negative"
        ) from error
    if not np.isfinite(result) or result < 0.0:
        raise RobertValidationError(
            f"chi-square for dataset {dataset_name!r} must be finite and non-negative"
        )
    return result


def _validated_rank(value: object, dataset_name: str) -> int:
    """Validate one non-negative integer residual rank."""

    if isinstance(value, (bool, np.bool_)):
        raise RobertValidationError(
            f"effective residual rank for dataset {dataset_name!r} "
            "must be a non-negative integer"
        )
    try:
        rank = int(value)
    except (TypeError, ValueError) as error:
        raise RobertValidationError(
            f"effective residual rank for dataset {dataset_name!r} "
            "must be a non-negative integer"
        ) from error
    if rank != value or rank < 0:
        raise RobertValidationError(
            f"effective residual rank for dataset {dataset_name!r} "
            "must be a non-negative integer"
        )
    return rank


def inject_spectrum_collection(
    spectra: Mapping[str, Spectrum],
    uncertainties: Mapping[str, ArrayLike],
    *,
    seed: int,
    noise_scale: float = 1.0,
    instruments: Mapping[str, str | None] | None = None,
    metadata: Mapping[str, str] | None = None,
) -> ObservationCollection:
    """Create independent, repeatable Gaussian injections for named spectra.

    A ``SeedSequence`` creates one child seed for each data set. This prevents
    equal-length data sets from receiving the same noise vector.
    """

    normalized_seed = int(seed)
    if normalized_seed < 0:
        raise RobertValidationError("injection seed must be non-negative")
    named_spectra = {str(name).strip(): spectrum for name, spectrum in spectra.items()}
    if not named_spectra or any(not name for name in named_spectra):
        raise RobertValidationError("injection spectra require non-empty dataset names")
    if any(not isinstance(spectrum, Spectrum) for spectrum in named_spectra.values()):
        raise RobertValidationError("every injected dataset value must be a Spectrum")
    expected_names = set(named_spectra)
    if set(uncertainties) != expected_names:
        raise RobertValidationError(
            "injection uncertainty names must match spectrum names"
        )
    instrument_names = (
        {name: None for name in named_spectra}
        if instruments is None
        else {str(name): value for name, value in instruments.items()}
    )
    if set(instrument_names) != expected_names:
        raise RobertValidationError(
            "injection instrument names must match spectrum names"
        )

    child_sequences = np.random.SeedSequence(normalized_seed).spawn(
        len(named_spectra)
    )
    datasets: list[ObservationDataset] = []
    for (name, spectrum), child_sequence in zip(
        named_spectra.items(), child_sequences, strict=True
    ):
        child_seed = int(child_sequence.generate_state(1, dtype=np.uint32)[0])
        observation = inject_spectrum(
            spectrum,
            uncertainties[name],
            seed=child_seed,
            noise_scale=noise_scale,
            instrument=instrument_names[name],
            metadata={
                "collection_injection_seed": str(normalized_seed),
                "dataset_name": name,
                **({} if metadata is None else dict(metadata)),
            },
        )
        datasets.append(ObservationDataset(name=name, observation=observation))
    return ObservationCollection(tuple(datasets))


def evaluate_multi_dataset_injection_recovery(
    *,
    case_name: str,
    truth: Mapping[str, float],
    estimates: Mapping[str, float],
    absolute_tolerances: Mapping[str, float],
    observations: ObservationCollection,
    best_fit_spectra: Mapping[str, Spectrum],
    likelihood: EffectiveMultiDatasetLikelihood,
    seed: int,
    likelihood_parameters: Mapping[str, float] | None = None,
    parameter_order: Sequence[str] | None = None,
    posterior_covariance: ArrayLike | None = None,
    inference_converged: bool = True,
    reduced_chi_square_bounds: tuple[float, float] = (0.5, 1.5),
    metadata: Mapping[str, str] | None = None,
) -> InjectionRecoveryReport:
    """Evaluate one recovery report across all named data sets.

    The likelihood supplies its effective model, data, and uncertainty arrays.
    If available, exact per-dataset chi-square and retained residual-dimension
    hooks are used.  This keeps covariance-aware and profiled statistics
    consistent with the retrieval and optimal-estimation calculation.
    """

    if set(best_fit_spectra) != set(observations.names):
        raise RobertValidationError(
            "best-fit spectrum names must match observation names"
        )
    if any(not isinstance(value, Spectrum) for value in best_fit_spectra.values()):
        raise RobertValidationError("every best-fit dataset value must be a Spectrum")
    inputs = likelihood.effective_inputs_by_dataset(
        best_fit_spectra,
        observations,
        likelihood_parameters,
    )
    if set(inputs) != set(observations.names):
        raise RobertValidationError(
            "effective likelihood input names must match observation names"
        )

    model_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    uncertainty_parts: list[np.ndarray] = []
    point_counts: dict[str, int] = {}
    fallback_chi_square_by_dataset: dict[str, float] = {}
    for dataset in observations.datasets:
        values = tuple(inputs[dataset.name])
        if len(values) != 3:
            raise RobertValidationError(
                "effective likelihood inputs must contain model, data, and uncertainty"
            )
        model, data, uncertainty = (
            np.asarray(value, dtype=float) for value in values
        )
        if (
            model.ndim != 1
            or data.shape != model.shape
            or uncertainty.shape != model.shape
            or model.size == 0
        ):
            raise RobertValidationError(
                "effective likelihood arrays must be non-empty matching vectors"
            )
        if (
            not np.all(np.isfinite(model))
            or not np.all(np.isfinite(data))
            or not np.all(np.isfinite(uncertainty))
            or np.any(uncertainty <= 0.0)
        ):
            raise RobertValidationError(
                "effective likelihood arrays must be finite with positive uncertainty"
            )
        model_parts.append(model)
        data_parts.append(data)
        uncertainty_parts.append(uncertainty)
        point_counts[dataset.name] = int(model.size)
        residual = (data - model) / uncertainty
        fallback_chi_square_by_dataset[dataset.name] = _validated_chi_square(
            np.dot(residual, residual),
            dataset.name,
        )

    chi_square_method = getattr(likelihood, "chi_square_by_dataset", None)
    if callable(chi_square_method):
        chi_square_values = chi_square_method(
            best_fit_spectra,
            observations,
            likelihood_parameters,
        )
        if not isinstance(chi_square_values, Mapping) or set(chi_square_values) != set(
            observations.names
        ):
            raise RobertValidationError(
                "exact chi-square names must match observation names"
            )
        chi_square_by_dataset = {
            name: _validated_chi_square(chi_square_values[name], name)
            for name in observations.names
        }
    else:
        if _has_covariance_interface(likelihood):
            raise RobertValidationError(
                "covariance-aware multi-dataset likelihood must provide "
                "chi_square_by_dataset; diagonal effective inputs are not an "
                "exact fallback"
            )
        chi_square_by_dataset = fallback_chi_square_by_dataset

    rank_method = getattr(likelihood, "effective_residual_rank_by_dataset", None)
    if callable(rank_method):
        rank_values = rank_method(
            best_fit_spectra,
            observations,
            likelihood_parameters,
        )
        if not isinstance(rank_values, Mapping) or set(rank_values) != set(
            observations.names
        ):
            raise RobertValidationError(
                "effective residual rank names must match observation names"
            )
        rank_by_dataset = {
            name: _validated_rank(rank_values[name], name)
            for name in observations.names
        }
    else:
        rank_by_dataset = {
            name: point_counts[name]
            for name in observations.names
        }

    for name, rank in rank_by_dataset.items():
        if rank > point_counts[name]:
            raise RobertValidationError(
                f"effective residual rank for dataset {name!r} exceeds "
                "its effective point count"
            )
    effective_n_active_points = sum(rank_by_dataset.values())
    chi_square_override = float(sum(chi_square_by_dataset.values()))

    model_values = np.concatenate(model_parts)
    data_values = np.concatenate(data_parts)
    uncertainty_values = np.concatenate(uncertainty_parts)
    coordinate = np.arange(1, model_values.size + 1, dtype=float)
    effective_grid = SpectralGrid.from_array(
        coordinate,
        unit="effective_index",
        role="observed",
        name="combined effective likelihood inputs",
    )
    effective_spectrum = Spectrum(
        spectral_grid=effective_grid,
        values=model_values,
        unit="effective_flux",
        observable="effective_likelihood_input",
    )
    effective_observation = Observation.from_arrays(
        coordinate,
        data_values,
        uncertainty_values,
        wavelength_unit="effective_index",
        flux_unit="effective_flux",
        observable="effective_likelihood_input",
        instrument="combined synthetic datasets",
    )
    return evaluate_injection_recovery(
        case_name=case_name,
        truth=truth,
        estimates=estimates,
        absolute_tolerances=absolute_tolerances,
        observation=effective_observation,
        best_fit_spectrum=effective_spectrum,
        seed=seed,
        parameter_order=parameter_order,
        posterior_covariance=posterior_covariance,
        inference_converged=inference_converged,
        reduced_chi_square_bounds=reduced_chi_square_bounds,
        chi_square_override=chi_square_override,
        effective_n_active_points=effective_n_active_points,
        metadata={
            "dataset_names": ",".join(observations.names),
            **({} if metadata is None else dict(metadata)),
        },
    )


__all__ = [
    "EffectiveMultiDatasetLikelihood",
    "evaluate_multi_dataset_injection_recovery",
    "inject_spectrum_collection",
]
