"""Portable, observed-grid best-fit prediction artifacts.

The artifact stores named one-dimensional :class:`Spectrum` and
:class:`Observation` pairs.  This is an intentional boundary.  Prepared
high-resolution likelihoods and order/frame/pixel cubes do not have one safe
flattening convention; capture records them as unavailable instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from robert_exoplanets.core import (
    RobertDataError,
    RobertValidationError,
    SpectralGrid,
    Spectrum,
)
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.instruments import Observation


BEST_FIT_PREDICTION_SCHEMA = "robert-best-fit-prediction-v1"
BEST_FIT_PREDICTION_SCHEMA_VERSION = "1.0"
BEST_FIT_PREDICTION_JSON_FILENAME = "best_fit_prediction.json"
BEST_FIT_PREDICTION_ARRAYS_FILENAME = "best_fit_prediction.npz"

_AVAILABLE = "available"
_UNAVAILABLE = "unavailable"
_PARTIAL = "partial"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return immutable_mapping({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _json_value(value: Any, path: str = "value") -> Any:
    """Convert metadata to strict JSON and reject non-finite values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise RobertValidationError(f"{path} must not contain non-finite numbers")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item(), path)
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Path):
        return str(value)
    raise RobertValidationError(
        f"{path} contains unsupported metadata type {type(value).__name__}"
    )


def _text(value: object, name: str, default: str | None = None) -> str:
    if value is None and default is not None:
        value = default
    result = str(value).strip()
    if not result:
        raise RobertValidationError(f"{name} must not be empty")
    return result


def _string_mapping(value: object) -> Mapping[str, str]:
    if value is None:
        return immutable_mapping({})
    if not isinstance(value, Mapping):
        raise RobertValidationError("metadata must be a mapping")
    return immutable_mapping({str(key): str(item) for key, item in value.items()})


def _likelihood_provenance(problem: object) -> dict[str, object]:
    """Record settings needed to reproduce supported saved fit statistics."""

    likelihood = getattr(problem, "likelihood", None)
    kind = "unsupported"
    try:
        from robert_exoplanets.likelihoods import (
            GaussianLikelihood,
            MultiDatasetGaussianLikelihood,
        )
    except ImportError:
        GaussianLikelihood = MultiDatasetGaussianLikelihood = ()  # type: ignore[assignment]
    if isinstance(likelihood, GaussianLikelihood):
        kind = "gaussian"
    elif isinstance(likelihood, MultiDatasetGaussianLikelihood):
        kind = "multi_dataset_gaussian"
    record: dict[str, object] = {
        "kind": kind,
        "type": type(likelihood).__name__ if likelihood is not None else None,
        "name": None if likelihood is None else str(getattr(likelihood, "name", "")),
    }
    for attribute in (
        "include_normalization",
        "invalid_model_loglike",
        "coordinate_rtol",
        "coordinate_atol",
        "offset_parameter",
        "jitter_parameter",
        "uncertainty_scale_parameter",
        "uncertainty_scale",
    ):
        if hasattr(likelihood, attribute):
            value = getattr(likelihood, attribute)
            try:
                record[attribute] = _json_value(value, f"likelihood.{attribute}")
            except RobertValidationError:
                # The default invalid-model floor is -inf.  It is runtime
                # control flow, not a physical setting needed for replay.
                continue
    if kind == "multi_dataset_gaussian":
        collection = getattr(problem, "observations", None)
        records: dict[str, object] = {}
        for dataset in getattr(collection, "datasets", ()):
            records[str(dataset.name)] = {
                "offset_parameter": dataset.offset_parameter,
                "jitter_parameter": dataset.jitter_parameter,
                "uncertainty_scale_parameter": dataset.uncertainty_scale_parameter,
                "uncertainty_scale": _json_value(
                    dataset.uncertainty_scale,
                    f"likelihood.datasets.{dataset.name}.uncertainty_scale",
                ),
            }
        record["datasets"] = records
    return record


@dataclass(frozen=True)
class BestFitPredictionDataset:
    """One named observed-grid prediction and observation."""

    name: str
    spectrum: Spectrum | None = None
    observation: Observation | None = None
    status: str = _AVAILABLE
    reason: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _text(self.name, "prediction dataset name")
        status = _text(self.status, "prediction dataset status")
        if status not in {_AVAILABLE, _UNAVAILABLE}:
            raise RobertValidationError(
                "prediction dataset status must be 'available' or 'unavailable'"
            )
        reason = None if self.reason is None else str(self.reason).strip()
        if status == _AVAILABLE:
            if not isinstance(self.spectrum, Spectrum):
                raise RobertValidationError("available dataset requires a Spectrum")
            if not isinstance(self.observation, Observation):
                raise RobertValidationError("available dataset requires an Observation")
            _validate_pair(self.spectrum, self.observation)
            if reason:
                raise RobertValidationError("available dataset cannot have a reason")
        elif self.spectrum is not None or self.observation is not None:
            raise RobertValidationError("unavailable dataset cannot contain arrays")
        if status == _UNAVAILABLE and not reason:
            reason = "portable prediction data are unavailable"
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))

@dataclass(frozen=True)
class BestFitPredictionArtifact:
    """Immutable schema-versioned best-fit prediction output."""

    datasets: Mapping[str, BestFitPredictionDataset]
    problem_name: str = "best-fit-prediction"
    parameter_names: tuple[str, ...] = ()
    best_fit_parameters: Mapping[str, float] = field(default_factory=dict)
    model_identifier: str = "unknown-forward-model"
    opacity_identifiers: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)
    status: str = _AVAILABLE
    reason: str | None = None
    schema_version: str = BEST_FIT_PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = _text(self.schema_version, "prediction schema version")
        if schema_version != BEST_FIT_PREDICTION_SCHEMA_VERSION:
            raise RobertValidationError(
                f"unsupported best-fit prediction schema version: {schema_version}"
            )
        datasets = {str(name): dataset for name, dataset in self.datasets.items()}
        if any(not name for name in datasets):
            raise RobertValidationError("prediction dataset names must not be empty")
        if any(
            not isinstance(dataset, BestFitPredictionDataset) or dataset.name != name
            for name, dataset in datasets.items()
        ):
            raise RobertValidationError(
                "prediction datasets must be BestFitPredictionDataset instances"
            )
        status = _text(self.status, "prediction artifact status")
        if status not in {_AVAILABLE, _PARTIAL, _UNAVAILABLE}:
            raise RobertValidationError(
                "prediction artifact status must be available, partial, or unavailable"
            )
        available = sum(dataset.status == _AVAILABLE for dataset in datasets.values())
        expected_status = (
            _AVAILABLE
            if datasets and available == len(datasets)
            else _PARTIAL
            if available
            else _UNAVAILABLE
        )
        if status != expected_status:
            raise RobertValidationError(
                f"prediction artifact status {status!r} does not match datasets "
                f"({expected_status!r})"
            )
        names = tuple(str(name) for name in self.parameter_names)
        if len(set(names)) != len(names) or any(not name for name in names):
            raise RobertValidationError("prediction parameter names must be unique and non-empty")
        parameters: dict[str, float] = {}
        for name, value in self.best_fit_parameters.items():
            number = float(value)
            if not np.isfinite(number):
                raise RobertValidationError("best-fit parameters must be finite")
            parameters[str(name)] = number
        reason = None if self.reason is None else str(self.reason).strip()
        if status != _AVAILABLE and not reason:
            reason = (
                "some best-fit prediction datasets are unavailable"
                if status == _PARTIAL
                else "best-fit prediction data are unavailable"
            )
        object.__setattr__(self, "problem_name", _text(self.problem_name, "prediction problem name"))
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "datasets", immutable_mapping(datasets))
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "best_fit_parameters", immutable_mapping(parameters))
        object.__setattr__(self, "model_identifier", _text(self.model_identifier, "prediction model identifier"))
        object.__setattr__(self, "opacity_identifiers", _string_mapping(self.opacity_identifiers))
        object.__setattr__(self, "provenance", _freeze(dict(self.provenance)))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", reason)

    @property
    def spectra(self) -> Mapping[str, Spectrum]:
        """Return available model spectra by dataset name."""

        return immutable_mapping(
            {
                name: dataset.spectrum
                for name, dataset in self.datasets.items()
                if dataset.status == _AVAILABLE and dataset.spectrum is not None
            }
        )

    @property
    def observations(self) -> Mapping[str, Observation]:
        """Return available observations by dataset name."""

        return immutable_mapping(
            {
                name: dataset.observation
                for name, dataset in self.datasets.items()
                if dataset.status == _AVAILABLE and dataset.observation is not None
            }
        )

    @property
    def available_dataset_names(self) -> tuple[str, ...]:
        return tuple(name for name, dataset in self.datasets.items() if dataset.status == _AVAILABLE)

    @property
    def unavailable_dataset_names(self) -> tuple[str, ...]:
        return tuple(name for name, dataset in self.datasets.items() if dataset.status == _UNAVAILABLE)

    def reference_mapping(self) -> dict[str, object]:
        """Return the compact reference embedded in ``result.json``."""

        return {
            "status": self.status,
            "schema_version": self.schema_version,
            "json": BEST_FIT_PREDICTION_JSON_FILENAME,
            "arrays": BEST_FIT_PREDICTION_ARRAYS_FILENAME,
            "datasets": list(self.datasets),
            "available_datasets": list(self.available_dataset_names),
            "unavailable_datasets": list(self.unavailable_dataset_names),
            "reason": self.reason,
        }

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-compatible logical representation."""

        return {
            "schema": BEST_FIT_PREDICTION_SCHEMA,
            "schema_version": self.schema_version,
            "status": self.status,
            "reason": self.reason,
            "problem_name": self.problem_name,
            "parameter_names": list(self.parameter_names),
            "best_fit_parameters": dict(self.best_fit_parameters),
            "model_identifier": self.model_identifier,
            "opacity_identifiers": dict(self.opacity_identifiers),
            "provenance": _thaw(self.provenance),
            "datasets": {
                name: {
                    "status": dataset.status,
                    "reason": dataset.reason,
                    "metadata": _thaw(dataset.metadata),
                }
                for name, dataset in self.datasets.items()
            },
        }

    def write(self, output_dir: str | Path, *, overwrite: bool = True) -> tuple[Path, Path]:
        """Write the artifact's JSON and NPZ files."""

        return write_best_fit_prediction(self, output_dir, overwrite=overwrite)


def _validate_pair(spectrum: Spectrum, observation: Observation) -> None:
    if spectrum.values.shape != observation.flux.shape:
        raise RobertValidationError("best-fit spectrum and observation shapes must match")
    if spectrum.spectral_grid.unit != observation.wavelength_unit:
        raise RobertValidationError("best-fit spectrum and observation units must match")
    if not np.array_equal(spectrum.spectral_grid.values, observation.wavelength):
        raise RobertValidationError("best-fit spectrum must use the observation coordinate grid")
    if spectrum.unit != observation.flux_unit:
        raise RobertValidationError("best-fit spectrum and observation value units must match")
    if spectrum.observable != observation.observable:
        raise RobertValidationError("best-fit spectrum and observation observables must match")
    if (
        spectrum.spectral_grid.bin_edges is not None
        and observation.wavelength_bin_edges is not None
        and not np.array_equal(
            spectrum.spectral_grid.bin_edges,
            observation.wavelength_bin_edges,
        )
    ):
        raise RobertValidationError("best-fit spectrum and observation coordinate edges must match")


def _spectrum_descriptor(
    spectrum: Spectrum,
    arrays: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, object]:
    coordinate_key = f"{prefix}_coordinate"
    values_key = f"{prefix}_values"
    arrays[coordinate_key] = np.asarray(spectrum.spectral_grid.values, dtype=float)
    arrays[values_key] = np.asarray(spectrum.values, dtype=float)
    descriptor: dict[str, object] = {
        "coordinate": coordinate_key,
        "values": values_key,
        "coordinate_unit": spectrum.spectral_grid.unit,
        "value_unit": spectrum.unit,
        "observable": spectrum.observable,
        "name": spectrum.spectral_grid.name,
        "role": spectrum.spectral_grid.role,
        "metadata": dict(spectrum.metadata),
    }
    if spectrum.uncertainty is not None:
        key = f"{prefix}_uncertainty"
        arrays[key] = np.asarray(spectrum.uncertainty, dtype=float)
        descriptor["uncertainty"] = key
    if spectrum.spectral_grid.bin_edges is not None:
        key = f"{prefix}_coordinate_edges"
        arrays[key] = np.asarray(spectrum.spectral_grid.bin_edges, dtype=float)
        descriptor["coordinate_edges"] = key
    return descriptor


def _observation_descriptor(
    observation: Observation,
    arrays: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, object]:
    coordinate_key = f"{prefix}_coordinate"
    values_key = f"{prefix}_values"
    uncertainty_key = f"{prefix}_uncertainty"
    arrays[coordinate_key] = np.asarray(observation.wavelength, dtype=float)
    arrays[values_key] = np.asarray(observation.flux, dtype=float)
    arrays[uncertainty_key] = np.asarray(observation.uncertainty, dtype=float)
    descriptor: dict[str, object] = {
        "coordinate": coordinate_key,
        "values": values_key,
        "uncertainty": uncertainty_key,
        "coordinate_unit": observation.wavelength_unit,
        "value_unit": observation.flux_unit,
        "observable": observation.observable,
        "instrument": observation.instrument,
        "metadata": dict(observation.metadata),
    }
    if observation.mask is not None:
        key = f"{prefix}_mask"
        arrays[key] = np.asarray(observation.mask, dtype=bool)
        descriptor["mask"] = key
    if observation.wavelength_bin_edges is not None:
        key = f"{prefix}_coordinate_edges"
        arrays[key] = np.asarray(observation.wavelength_bin_edges, dtype=float)
        descriptor["coordinate_edges"] = key
    return descriptor


def _storage_mapping(
    artifact: BestFitPredictionArtifact,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    datasets: dict[str, object] = {}
    for index, (name, dataset) in enumerate(artifact.datasets.items()):
        record: dict[str, object] = {
            "status": dataset.status,
            "reason": dataset.reason,
            "metadata": _thaw(dataset.metadata),
        }
        if dataset.status == _AVAILABLE:
            assert dataset.spectrum is not None and dataset.observation is not None
            prefix = f"dataset_{index:04d}"
            record["model"] = _spectrum_descriptor(
                dataset.spectrum,
                arrays,
                f"{prefix}_model",
            )
            record["observation"] = _observation_descriptor(
                dataset.observation,
                arrays,
                f"{prefix}_observation",
            )
        datasets[name] = record

    mapping = artifact.to_mapping()
    mapping["datasets"] = datasets
    mapping["arrays"] = BEST_FIT_PREDICTION_ARRAYS_FILENAME
    return mapping, arrays


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")


def write_best_fit_prediction(
    artifact: BestFitPredictionArtifact,
    output_dir: str | Path,
    *,
    overwrite: bool = True,
) -> tuple[Path, Path]:
    """Atomically write a portable JSON/NPZ prediction artifact."""

    if not isinstance(artifact, BestFitPredictionArtifact):
        raise RobertValidationError("artifact must be a BestFitPredictionArtifact")
    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / BEST_FIT_PREDICTION_JSON_FILENAME
    arrays_path = directory / BEST_FIT_PREDICTION_ARRAYS_FILENAME
    if not overwrite and (json_path.exists() or arrays_path.exists()):
        raise FileExistsError(json_path if json_path.exists() else arrays_path)
    mapping, arrays = _storage_mapping(artifact)
    temporary_arrays = _temporary(arrays_path)
    temporary_json = _temporary(json_path)
    try:
        np.savez_compressed(temporary_arrays, **arrays)
        temporary_arrays.replace(arrays_path)
        temporary_json.write_text(
            json.dumps(_json_value(mapping), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary_json.replace(json_path)
    except (OSError, ValueError, TypeError) as exc:
        for temporary in (temporary_arrays, temporary_json):
            try:
                temporary.unlink()
            except OSError:
                pass
        raise RobertDataError(f"failed to write best-fit prediction under {directory}") from exc
    return json_path, arrays_path


def _array(archive: Any, key: object, label: str, *, boolean: bool = False) -> np.ndarray:
    if not isinstance(key, str) or not key or key not in archive.files:
        raise RobertDataError(f"saved prediction is missing the {label} array")
    try:
        value = np.array(archive[key], copy=True)
    except (TypeError, ValueError) as exc:
        raise RobertDataError(f"saved prediction {label} array is invalid") from exc
    if value.ndim != 1:
        raise RobertDataError(f"saved prediction {label} array must be finite and one-dimensional")
    if boolean:
        if value.dtype.kind != "b":
            raise RobertDataError(f"saved prediction {label} array must have boolean dtype")
        return value
    try:
        value = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RobertDataError(f"saved prediction {label} array is invalid") from exc
    if not np.all(np.isfinite(value)):
        raise RobertDataError(f"saved prediction {label} array must be finite and one-dimensional")
    return value


def _metadata(value: object) -> Mapping[str, str]:
    if value is None:
        return immutable_mapping({})
    if not isinstance(value, Mapping):
        raise RobertDataError("saved prediction metadata must be a mapping")
    return immutable_mapping({str(key): str(item) for key, item in value.items()})


def _load_spectrum(descriptor: Mapping[str, object], archive: Any, label: str) -> Spectrum:
    coordinate = _array(archive, descriptor.get("coordinate"), f"{label} coordinate")
    values = _array(archive, descriptor.get("values"), f"{label} values")
    uncertainty_key = descriptor.get("uncertainty")
    uncertainty = None if uncertainty_key is None else _array(archive, uncertainty_key, f"{label} uncertainty")
    edges_key = descriptor.get("coordinate_edges")
    edges = None if edges_key is None else _array(archive, edges_key, f"{label} coordinate edges")
    try:
        grid = SpectralGrid(
            values=coordinate,
            unit=_text(descriptor.get("coordinate_unit"), f"{label} coordinate unit"),
            name=None if descriptor.get("name") is None else str(descriptor["name"]),
            role=_text(descriptor.get("role", "native"), f"{label} grid role"),
            bin_edges=edges,
            metadata=_metadata(descriptor.get("metadata")),
        )
        return Spectrum(
            spectral_grid=grid,
            values=values,
            unit=_text(descriptor.get("value_unit"), f"{label} value unit"),
            observable=_text(descriptor.get("observable"), f"{label} observable"),
            uncertainty=uncertainty,
            metadata=_metadata(descriptor.get("metadata")),
        )
    except RobertValidationError as exc:
        raise RobertDataError(f"saved {label} spectrum is invalid") from exc


def _load_observation(descriptor: Mapping[str, object], archive: Any, label: str) -> Observation:
    coordinate = _array(archive, descriptor.get("coordinate"), f"{label} coordinate")
    values = _array(archive, descriptor.get("values"), f"{label} values")
    uncertainty = _array(archive, descriptor.get("uncertainty"), f"{label} uncertainty")
    mask_key = descriptor.get("mask")
    mask = None if mask_key is None else _array(archive, mask_key, f"{label} mask", boolean=True)
    edges_key = descriptor.get("coordinate_edges")
    edges = None if edges_key is None else _array(archive, edges_key, f"{label} coordinate edges")
    try:
        return Observation(
            wavelength=coordinate,
            flux=values,
            uncertainty=uncertainty,
            wavelength_unit=_text(descriptor.get("coordinate_unit"), f"{label} coordinate unit"),
            flux_unit=_text(descriptor.get("value_unit"), f"{label} value unit"),
            observable=_text(descriptor.get("observable"), f"{label} observable"),
            instrument=None if descriptor.get("instrument") in (None, "") else str(descriptor["instrument"]),
            mask=mask,
            wavelength_bin_edges=edges,
            metadata=_metadata(descriptor.get("metadata")),
        )
    except RobertValidationError as exc:
        raise RobertDataError(f"saved {label} observation is invalid") from exc


def load_best_fit_prediction(path: str | Path) -> BestFitPredictionArtifact:
    """Load and validate a portable best-fit prediction JSON/NPZ pair."""

    source = Path(path).expanduser()
    json_path = source / BEST_FIT_PREDICTION_JSON_FILENAME if source.is_dir() else source
    if not json_path.is_file():
        raise RobertDataError(f"best-fit prediction JSON does not exist: {json_path}")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RobertDataError(f"failed to read best-fit prediction JSON: {json_path}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != BEST_FIT_PREDICTION_SCHEMA:
        raise RobertDataError("best-fit prediction JSON has an unsupported schema")
    if payload.get("schema_version") != BEST_FIT_PREDICTION_SCHEMA_VERSION:
        raise RobertDataError("best-fit prediction JSON has an unsupported schema version")
    datasets_payload = payload.get("datasets")
    if not isinstance(datasets_payload, Mapping):
        raise RobertDataError("best-fit prediction datasets must be a mapping")
    arrays_reference = payload.get("arrays", BEST_FIT_PREDICTION_ARRAYS_FILENAME)
    if not isinstance(arrays_reference, str) or not arrays_reference:
        raise RobertDataError("best-fit prediction JSON must reference an arrays file")
    arrays_path = json_path.parent / arrays_reference
    if not arrays_path.is_file():
        raise RobertDataError(f"best-fit prediction arrays do not exist: {arrays_path}")
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            datasets: dict[str, BestFitPredictionDataset] = {}
            for name, raw in datasets_payload.items():
                if not isinstance(raw, Mapping):
                    raise RobertDataError(f"prediction dataset {name!r} must be a mapping")
                status = str(raw.get("status", _AVAILABLE))
                metadata = raw.get("metadata", {})
                if not isinstance(metadata, Mapping):
                    raise RobertDataError(
                        f"prediction dataset {name!r} metadata must be a mapping"
                    )
                if status == _UNAVAILABLE:
                    datasets[str(name)] = BestFitPredictionDataset(
                        name=str(name),
                        status=_UNAVAILABLE,
                        reason=None if raw.get("reason") is None else str(raw["reason"]),
                        metadata=metadata,
                    )
                    continue
                if status != _AVAILABLE:
                    raise RobertDataError(f"unsupported prediction dataset status: {status!r}")
                spectrum = raw.get("model")
                observation = raw.get("observation")
                if not isinstance(spectrum, Mapping) or not isinstance(observation, Mapping):
                    raise RobertDataError(f"available prediction dataset {name!r} is incomplete")
                datasets[str(name)] = BestFitPredictionDataset(
                    name=str(name),
                    spectrum=_load_spectrum(spectrum, archive, f"dataset {name!r} spectrum"),
                    observation=_load_observation(observation, archive, f"dataset {name!r} observation"),
                    metadata=metadata,
                )
    except (OSError, ValueError) as exc:
        raise RobertDataError(f"failed to read best-fit prediction arrays: {arrays_path}") from exc
    try:
        return BestFitPredictionArtifact(
            datasets=datasets,
            problem_name=str(payload.get("problem_name", "best-fit-prediction")),
            parameter_names=tuple(str(name) for name in payload.get("parameter_names", ())),
            best_fit_parameters=(
                payload.get("best_fit_parameters")
                if isinstance(payload.get("best_fit_parameters"), Mapping)
                else {}
            ),
            model_identifier=str(payload.get("model_identifier", "unknown-forward-model")),
            opacity_identifiers=(
                payload.get("opacity_identifiers")
                if isinstance(payload.get("opacity_identifiers"), Mapping)
                else {}
            ),
            provenance=(
                payload.get("provenance") if isinstance(payload.get("provenance"), Mapping) else {}
            ),
            status=str(payload.get("status", _AVAILABLE)),
            reason=None if payload.get("reason") is None else str(payload["reason"]),
        )
    except RobertValidationError as exc:
        raise RobertDataError(f"invalid best-fit prediction artifact: {json_path}") from exc


def _problem_entries(problem: object) -> tuple[tuple[str, object | None, object | None], ...]:
    components = getattr(getattr(problem, "likelihood", None), "components", None)
    if isinstance(components, Sequence) and not isinstance(components, (str, bytes)):
        entries = tuple(
            (
                str(component.prediction_key),
                getattr(component, "observation", None),
                component,
            )
            for component in components
            if getattr(component, "prediction_key", None) is not None
        )
        if entries:
            return entries
    collection = getattr(problem, "observations", None)
    datasets = getattr(collection, "datasets", None)
    if isinstance(datasets, Sequence) and not isinstance(datasets, (str, bytes)):
        return tuple((str(dataset.name), getattr(dataset, "observation", None), None) for dataset in datasets)
    return (("primary", getattr(problem, "observation", None), None),)


def _prediction_mapping(prediction: object) -> Mapping[str, object]:
    if isinstance(prediction, Spectrum):
        return {"primary": prediction}
    if isinstance(prediction, Mapping):
        return {str(key): value for key, value in prediction.items()}
    spectra = getattr(prediction, "spectra", None)
    if isinstance(spectra, Mapping):
        return {str(key): value for key, value in spectra.items()}
    observed = getattr(prediction, "observed_spectrum", None)
    if isinstance(observed, Spectrum):
        return {"primary": observed}
    return {"primary": prediction}


def _capture_parameters(problem: object, parameters: object) -> dict[str, float]:
    if isinstance(parameters, Mapping):
        values = parameters
    elif isinstance(getattr(parameters, "best_fit_parameters", None), Mapping):
        values = getattr(parameters, "best_fit_parameters")
    else:
        mapper = getattr(problem, "parameter_mapping", None)
        if not callable(mapper):
            raise RobertValidationError("capture parameters must be a mapping or vector capability")
        values = mapper(parameters)
    if not isinstance(values, Mapping):
        raise RobertValidationError("capture parameters must resolve to a mapping")
    output = {str(key): float(value) for key, value in values.items()}
    if not all(np.isfinite(value) for value in output.values()):
        raise RobertValidationError("capture parameters must be finite")
    return output


def _qualified_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def capture_best_fit_prediction(
    problem: object,
    parameters: object,
    *,
    prediction: object | None = None,
    model_identifier: str | None = None,
    opacity_identifiers: Mapping[str, str] | None = None,
    provenance: Mapping[str, object] | None = None,
) -> BestFitPredictionArtifact:
    """Capture the explicit best-fit prediction capability of a problem.

    The call evaluates ``problem.predict`` only when ``prediction`` is not
    supplied.  Unsupported outputs are represented by unavailable records;
    this function never flattens HRS cubes or invents spectra.
    """

    values = _capture_parameters(problem, parameters)
    model = getattr(problem, "forward_model", None)
    parameter_names = tuple(str(name) for name in getattr(problem, "parameter_names", values))
    expected = _problem_entries(problem)
    evaluation_error: str | None = None
    if prediction is None:
        predictor = getattr(problem, "predict", None)
        if not callable(predictor):
            predicted = None
            evaluation_error = "problem does not expose an explicit predict capability"
        else:
            try:
                predicted = predictor(values)
            except Exception as exc:  # optional product must not invalidate a completed run
                predicted = None
                evaluation_error = f"forward prediction could not be captured: {type(exc).__name__}: {exc}"
    else:
        predicted = prediction
    predictions = {} if predicted is None else _prediction_mapping(predicted)
    records: dict[str, BestFitPredictionDataset] = {}
    for key, observation, component in expected:
        metadata: dict[str, object] = {}
        if component is not None:
            metadata["component_name"] = str(getattr(component, "name", key))
        reason: str | None = evaluation_error
        candidate = predictions.get(key)
        if reason is None and not isinstance(observation, Observation):
            reason = "observation is not a one-dimensional Observation; cube data are not flattened"
        if reason is None and not isinstance(candidate, Spectrum):
            reason = "prediction is not a one-dimensional Spectrum; cube data are not flattened"
        if reason is None:
            try:
                records[key] = BestFitPredictionDataset(
                    name=key,
                    spectrum=candidate,
                    observation=observation,
                    metadata=metadata,
                )
                continue
            except RobertValidationError as exc:
                reason = f"prediction and observation cannot be serialized together: {exc}"
        records[key] = BestFitPredictionDataset(
            name=key,
            status=_UNAVAILABLE,
            reason=reason,
            metadata=metadata,
        )
    if not records:
        records["primary"] = BestFitPredictionDataset(
            name="primary",
            status=_UNAVAILABLE,
            reason=evaluation_error or "problem does not expose named observations",
        )
    available = sum(record.status == _AVAILABLE for record in records.values())
    status = _AVAILABLE if available == len(records) else _PARTIAL if available else _UNAVAILABLE
    reason = evaluation_error
    if status == _PARTIAL and reason is None:
        reason = "some best-fit prediction datasets are unavailable"
    if status == _UNAVAILABLE and reason is None:
        reason = "best-fit prediction data are unavailable"
    return BestFitPredictionArtifact(
        datasets=records,
        problem_name=str(getattr(problem, "name", "best-fit-prediction")),
        parameter_names=parameter_names,
        best_fit_parameters=values,
        model_identifier=model_identifier or (_qualified_type(model) if model is not None else "unknown-forward-model"),
        opacity_identifiers=(
            opacity_identifiers
            if opacity_identifiers is not None
            else getattr(problem, "opacity_identifiers", {})
        ),
        provenance={
            **dict(provenance or {}),
            "capture": "explicit_best_fit_prediction",
            "forward_model": _qualified_type(model) if model is not None else None,
            "prediction_type": None if predicted is None else _qualified_type(predicted),
            "parameter_source": "best_fit_parameters",
            "problem_metadata": dict(getattr(problem, "metadata", {})),
            "likelihood": _likelihood_provenance(problem),
        },
        status=status,
        reason=reason,
    )


__all__ = [
    "BEST_FIT_PREDICTION_ARRAYS_FILENAME",
    "BEST_FIT_PREDICTION_JSON_FILENAME",
    "BEST_FIT_PREDICTION_SCHEMA",
    "BEST_FIT_PREDICTION_SCHEMA_VERSION",
    "BestFitPredictionArtifact",
    "BestFitPredictionDataset",
    "capture_best_fit_prediction",
    "load_best_fit_prediction",
    "write_best_fit_prediction",
]
