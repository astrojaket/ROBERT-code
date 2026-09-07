"""Reproducible retrieval run manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable, Mapping

import numpy as np

from robert_exoplanets._version import __version__
from robert_exoplanets.core import RobertConfigError, RobertDataError, RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping

from .protocols import SamplerRetrievalProblem


RUN_MANIFEST_SCHEMA_VERSION = "1.0"
RUN_MANIFEST_FILENAME = "manifest.json"

# These values mirror the public ``run_multinest`` defaults.  Keeping them in
# the manifest layer makes an omitted value and an explicit default produce the
# same run identity before the optional native adapter is imported.
MULTINEST_DEFAULT_SETTINGS: dict[str, object] = {
    "n_live_points": 400,
    "max_iter": 0,
    "evidence_tolerance": 0.5,
    "sampling_efficiency": 0.8,
    "resume": True,
    "verbose": True,
    "mpi_nprocs": None,
    "seed": None,
    "invalid_loglike_floor": -1.0e100,
    "importance_nested_sampling": True,
    "multimodal": True,
    "n_iter_before_update": 100,
}

# A resumed MultiNest checkpoint keeps the scientific model and the sampler
# geometry fixed.  Budgets, stopping criteria, and progress output belong to
# one attempt and can be changed when a later attempt uses the same checkpoint.
MULTINEST_PER_ATTEMPT_SETTINGS = frozenset(
    {
        "max_iter",
        "evidence_tolerance",
        "verbose",
        "n_iter_before_update",
        "dump_callback",
    }
)
MULTINEST_STRUCTURAL_SETTINGS = frozenset(
    {
        "n_live_points",
        "sampling_efficiency",
        "importance_nested_sampling",
        "multimodal",
        "mpi_nprocs",
        "seed",
    }
)


@dataclass(frozen=True)
class RunManifest:
    """Versioned description of one reproducible retrieval run."""

    problem_name: str
    method: str
    created_at_utc: str
    config_hash: str
    parameter_names: tuple[str, ...]
    parameter_priors: tuple[Mapping[str, object], ...]
    likelihood: Mapping[str, object]
    problem_metadata: Mapping[str, str]
    opacity_identifiers: Mapping[str, str]
    settings: Mapping[str, object]
    random_seed: int | None
    robert_version: str = __version__
    python_version: str = platform.python_version()
    platform: str = platform.platform()
    git_commit: str | None = None
    git_dirty: bool | None = None
    schema_version: str = RUN_MANIFEST_SCHEMA_VERSION
    observation_identity: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "problem_name",
            "method",
            "created_at_utc",
            "config_hash",
            "schema_version",
        ):
            if not str(getattr(self, name)).strip():
                raise RobertValidationError(f"manifest {name} must not be empty")
        object.__setattr__(self, "parameter_names", tuple(self.parameter_names))
        object.__setattr__(
            self,
            "parameter_priors",
            tuple(immutable_mapping(dict(item)) for item in self.parameter_priors),
        )
        object.__setattr__(self, "likelihood", immutable_mapping(self.likelihood))
        object.__setattr__(
            self, "problem_metadata", immutable_mapping(self.problem_metadata)
        )
        object.__setattr__(
            self, "opacity_identifiers", immutable_mapping(self.opacity_identifiers)
        )
        object.__setattr__(self, "settings", immutable_mapping(self.settings))
        object.__setattr__(
            self,
            "observation_identity",
            immutable_mapping(self.observation_identity),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable manifest mapping."""

        return {
            "schema_version": self.schema_version,
            "problem_name": self.problem_name,
            "method": self.method,
            "created_at_utc": self.created_at_utc,
            "config_hash": self.config_hash,
            "parameter_names": list(self.parameter_names),
            "parameter_priors": [dict(item) for item in self.parameter_priors],
            "likelihood": dict(self.likelihood),
            "problem_metadata": dict(self.problem_metadata),
            "opacity_identifiers": dict(self.opacity_identifiers),
            "settings": dict(self.settings),
            "random_seed": self.random_seed,
            "robert_version": self.robert_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "observation_identity": dict(self.observation_identity),
        }


def build_run_manifest(
    problem: SamplerRetrievalProblem,
    *,
    method: str,
    settings: Mapping[str, object],
    random_seed: int | None,
) -> RunManifest:
    """Build a manifest before inference begins."""

    normalized_method = normalize_retrieval_method(method)
    safe_settings = normalize_run_settings(normalized_method, settings)
    prior_records = tuple(
        {
            "name": parameter.name,
            "prior": type(parameter.prior).__name__,
            "lower": parameter.prior.lower,
            "upper": parameter.prior.upper,
            "label": parameter.label,
            "unit": parameter.unit,
            "metadata": dict(parameter.metadata),
        }
        for parameter in problem.parameters.parameters
    )
    likelihood_record = _likelihood_record(problem)
    observation_identity = _observation_identity(problem)
    signature = {
        "problem_name": problem.name,
        "parameter_priors": prior_records,
        "likelihood": likelihood_record,
        "observation_identity": observation_identity,
        "problem_metadata": dict(problem.metadata),
        "opacity_identifiers": dict(problem.opacity_identifiers),
        "method": normalized_method,
        "settings": safe_settings,
        "random_seed": random_seed,
    }
    encoded = json.dumps(
        _json_value(signature), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    git_commit, git_dirty = _git_state()
    return RunManifest(
        problem_name=problem.name,
        method=normalized_method,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        config_hash=sha256(encoded).hexdigest(),
        parameter_names=problem.parameter_names,
        parameter_priors=prior_records,
        likelihood=likelihood_record,
        problem_metadata=dict(problem.metadata),
        opacity_identifiers=dict(problem.opacity_identifiers),
        settings=safe_settings,
        random_seed=random_seed,
        git_commit=git_commit,
        git_dirty=git_dirty,
        observation_identity=observation_identity,
    )


def _likelihood_record(
    problem: SamplerRetrievalProblem,
) -> dict[str, object]:
    likelihood = problem.likelihood
    common = _single_likelihood_record(likelihood)
    components = getattr(likelihood, "components", None)
    if isinstance(components, (tuple, list)):
        common["components"] = [
            {
                "name": component.name,
                "prediction_key": component.prediction_key,
                "observation_dispatch": (
                    "explicit" if component.observation is not None else "prepared"
                ),
                "likelihood": _single_likelihood_record(component.likelihood),
                "metadata": dict(component.metadata),
            }
            for component in components
        ]
    else:
        observations = getattr(problem, "observations", None)
        datasets = getattr(observations, "datasets", None)
        if datasets is None:
            if hasattr(likelihood, "offset_parameter"):
                common["offset_parameter"] = likelihood.offset_parameter
            if hasattr(likelihood, "jitter_parameter"):
                common["jitter_parameter"] = likelihood.jitter_parameter
            return _json_mapping(common)

        component_likelihoods = getattr(likelihood, "likelihoods", None)
        if isinstance(component_likelihoods, Mapping):
            common["datasets"] = [
                {
                    "name": dataset.name,
                    "offset_parameter": dataset.offset_parameter,
                    "jitter_parameter": dataset.jitter_parameter,
                    "uncertainty_scale_parameter": (
                        dataset.uncertainty_scale_parameter
                    ),
                    "uncertainty_scale": dataset.uncertainty_scale,
                    "likelihood": _single_likelihood_record(
                        component_likelihoods[dataset.name]
                    ),
                }
                for dataset in datasets
            ]
        else:
            common["datasets"] = [
                {
                    "name": dataset.name,
                    "offset_parameter": dataset.offset_parameter,
                    "jitter_parameter": dataset.jitter_parameter,
                    "uncertainty_scale_parameter": (
                        dataset.uncertainty_scale_parameter
                    ),
                    "uncertainty_scale": dataset.uncertainty_scale,
                }
                for dataset in datasets
            ]
    return _json_mapping(common)


def normalize_retrieval_method(method: str) -> str:
    """Return the canonical method name stored in a run manifest."""

    normalized = str(method).strip().lower().replace("-", "_")
    if normalized in {"nested", "nested_sampling", "multinest", "multi_nest", "pymultinest"}:
        return "multinest"
    if normalized in {"oe", "optimal_estimation"}:
        return "optimal_estimation"
    return normalized


def normalize_multinest_resume(value: object) -> bool:
    """Normalize the explicit MultiNest resume policy to a boolean."""

    if isinstance(value, bool):
        return value
    text = str(value).strip().lower().replace("-", "_")
    if text in {"resume", "true", "1"}:
        return True
    if text in {"overwrite", "false", "0"}:
        return False
    raise RobertConfigError(
        "MultiNest resume must be True/'resume' or False/'overwrite'; "
        f"received {value!r}"
    )


def normalize_run_settings(
    method: str,
    settings: Mapping[str, object],
) -> dict[str, object]:
    """Canonicalize manifest settings before hashing or compatibility checks."""

    normalized_method = normalize_retrieval_method(method)
    normalized = _json_mapping(settings)
    normalized["method"] = normalized_method
    if normalized_method != "multinest":
        return normalized

    for key, default in MULTINEST_DEFAULT_SETTINGS.items():
        normalized.setdefault(key, default)
    normalized["resume"] = normalize_multinest_resume(normalized["resume"])
    _coerce_setting(normalized, "n_live_points", int)
    _coerce_setting(normalized, "max_iter", int)
    _coerce_setting(normalized, "n_iter_before_update", int)
    _coerce_setting(normalized, "mpi_nprocs", int, allow_none=True)
    for key in (
        "evidence_tolerance",
        "sampling_efficiency",
        "invalid_loglike_floor",
    ):
        _coerce_setting(normalized, key, float)
    for key in ("verbose", "importance_nested_sampling", "multimodal"):
        _coerce_setting(normalized, key, bool)
    _coerce_setting(normalized, "seed", int, allow_none=True)
    return normalized


def _coerce_setting(
    settings: dict[str, object],
    name: str,
    converter: Callable[[Any], object],
    *,
    allow_none: bool = False,
) -> None:
    value = settings.get(name)
    if value is None and allow_none:
        return
    try:
        settings[name] = converter(value)
    except (TypeError, ValueError, OverflowError):
        # Validation remains the sampler adapter's responsibility.  Keeping an
        # invalid value in the manifest still gives the user a useful context
        # if the adapter rejects it before opening a checkpoint.
        return


def resume_compatibility_changes(
    original: RunManifest,
    current: RunManifest,
) -> tuple[str, ...]:
    """Return immutable scientific or sampler settings changed on resume."""

    original_method = normalize_retrieval_method(original.method)
    current_method = normalize_retrieval_method(current.method)
    fields = (
        "problem_name",
        "parameter_names",
        "parameter_priors",
        "likelihood",
        "problem_metadata",
        "opacity_identifiers",
        "random_seed",
        "observation_identity",
    )
    changed = [
        name for name in fields if getattr(original, name) != getattr(current, name)
    ]
    if original_method != current_method:
        changed.append("method")
    original_settings = normalize_run_settings(original.method, original.settings)
    current_settings = normalize_run_settings(current.method, current.settings)
    if original_method == "multinest" or current_method == "multinest":
        original_floor = original_settings.get("invalid_loglike_floor")
        current_floor = current_settings.get("invalid_loglike_floor")
        if original_floor != current_floor:
            changed.append("invalid_loglike_floor")
        keys = set(original_settings) | set(current_settings)
        for key in sorted(keys):
            if key in {"method", "resume", "invalid_loglike_floor"}:
                continue
            if key in MULTINEST_PER_ATTEMPT_SETTINGS:
                continue
            if original_settings.get(key) != current_settings.get(key):
                changed.append(f"settings.{key}")
    return tuple(dict.fromkeys(changed))


def validate_resume_compatibility(
    original: RunManifest,
    current: RunManifest,
) -> None:
    """Raise a configuration error when a checkpoint cannot be resumed."""

    changed = resume_compatibility_changes(original, current)
    if changed:
        raise RobertConfigError(
            "cannot resume because the scientific or sampler structure changed: "
            + ", ".join(changed)
            + ". Use a new output directory or resume='overwrite'."
        )


def _observation_identity(problem: SamplerRetrievalProblem) -> dict[str, object]:
    """Return compact data identities without copying observation arrays."""

    observations = getattr(problem, "observations", None)
    datasets = getattr(observations, "datasets", None)
    if datasets is not None:
        return _with_identity_hash(
            {
                "kind": "observation_collection",
                "name": str(getattr(observations, "name", type(observations).__name__)),
                "metadata": _metadata_mapping(getattr(observations, "metadata", {})),
                "datasets": [
                    {
                        "name": str(getattr(dataset, "name", index)),
                        "metadata": _metadata_mapping(
                            getattr(dataset, "metadata", {})
                        ),
                        "observation": _observation_data_identity(
                            getattr(dataset, "observation", None)
                        ),
                    }
                    for index, dataset in enumerate(datasets)
                ],
            }
        )

    observation = getattr(problem, "observation", None)
    if observation is not None:
        return _with_identity_hash(
            {
                "kind": "observation",
                "observation": _observation_data_identity(observation),
            }
        )

    components = getattr(problem.likelihood, "components", None)
    if isinstance(components, (tuple, list)):
        component_records: list[dict[str, object]] = []
        for component in components:
            explicit_observation = getattr(component, "observation", None)
            if explicit_observation is not None:
                data_identity = _observation_data_identity(explicit_observation)
            else:
                embedded_observation = getattr(
                    getattr(component, "likelihood", None),
                    "observation",
                    None,
                )
                data_identity = _observation_data_identity(embedded_observation)
            component_records.append(
                {
                    "name": str(getattr(component, "name", "component")),
                    "prediction_key": str(
                        getattr(component, "prediction_key", "")
                    ),
                    "metadata": _metadata_mapping(
                        getattr(component, "metadata", {})
                    ),
                    "observation": data_identity,
                }
            )
        return _with_identity_hash(
            {
                "kind": "heterogeneous_observations",
                "components": component_records,
            }
        )

    # Device and prepared problems do not have a universal array layout.  The
    # problem metadata is the explicit identity boundary for these backends;
    # configured builders should put source checksums and physical state hashes
    # there rather than relying on object reprs or model introspection.
    return _with_identity_hash(
        {
            "kind": "declared_metadata",
            "type": _qualified_type(problem),
            "metadata": _metadata_mapping(getattr(problem, "metadata", {})),
            "identity_requirement": "explicit metadata identity",
        }
    )


def _observation_data_identity(observation: object) -> dict[str, object]:
    """Fingerprint known observation layouts or retain explicit metadata."""

    if observation is None:
        return {
            "kind": "prepared_or_opaque",
            "identity_requirement": "explicit metadata identity",
        }

    if all(hasattr(observation, name) for name in ("wavelength", "flux", "uncertainty")):
        descriptor: dict[str, object] = {
            "kind": "one_dimensional",
            "type": _qualified_type(observation),
            "wavelength_unit": str(getattr(observation, "wavelength_unit", "")),
            "flux_unit": str(getattr(observation, "flux_unit", "")),
            "observable": str(getattr(observation, "observable", "")),
            "instrument": getattr(observation, "instrument", None),
            "metadata": _metadata_mapping(getattr(observation, "metadata", {})),
            "arrays": {
                "wavelength": _array_identity(getattr(observation, "wavelength")),
                "wavelength_bin_edges": _array_identity(
                    getattr(observation, "wavelength_bin_edges", None)
                ),
                "flux": _array_identity(getattr(observation, "flux")),
                "uncertainty": _array_identity(
                    getattr(observation, "uncertainty")
                ),
                "mask": _array_identity(getattr(observation, "mask", None), boolean=True),
            },
        }
        return _with_identity_hash(descriptor)

    if all(
        hasattr(observation, name)
        for name in (
            "order_wavelengths",
            "flux",
            "phase",
            "fixed_velocity_km_s",
            "time_bjd",
        )
    ):
        descriptor = {
            "kind": "time_resolved_high_resolution",
            "type": _qualified_type(observation),
            "wavelength_unit": str(getattr(observation, "wavelength_unit", "")),
            "flux_unit": str(getattr(observation, "flux_unit", "")),
            "observable": str(getattr(observation, "observable", "")),
            "instrument": str(getattr(observation, "instrument", "")),
            "name": str(getattr(observation, "name", "")),
            "metadata": _metadata_mapping(getattr(observation, "metadata", {})),
            "arrays": {
                "order_wavelengths": _array_identity(
                    getattr(observation, "order_wavelengths")
                ),
                "flux": _array_identity(getattr(observation, "flux")),
                "phase": _array_identity(getattr(observation, "phase")),
                "fixed_velocity_km_s": _array_identity(
                    getattr(observation, "fixed_velocity_km_s")
                ),
                "time_bjd": _array_identity(getattr(observation, "time_bjd")),
                "mask": _array_identity(getattr(observation, "mask", None), boolean=True),
                "order_mask": _array_identity(
                    getattr(observation, "order_mask", None), boolean=True
                ),
                "frame_mask": _array_identity(
                    getattr(observation, "frame_mask", None), boolean=True
                ),
                "airmass": _array_identity(getattr(observation, "airmass", None)),
                "humidity_percent": _array_identity(
                    getattr(observation, "humidity_percent", None)
                ),
                "median_snr": _array_identity(
                    getattr(observation, "median_snr", None)
                ),
            },
        }
        return _with_identity_hash(descriptor)

    return _with_identity_hash(
        {
            "kind": "prepared_or_opaque",
            "type": _qualified_type(observation),
            "metadata": _metadata_mapping(getattr(observation, "metadata", {})),
            "identity_requirement": "explicit metadata identity",
        }
    )


def _array_identity(values: object, *, boolean: bool = False) -> dict[str, object] | None:
    """Return a digest and shape for one numeric or boolean data array."""

    if values is None:
        return None
    array = np.asarray(values, dtype=np.bool_ if boolean else np.float64)
    contiguous = np.ascontiguousarray(array)
    dtype = contiguous.dtype.str
    shape = [int(value) for value in contiguous.shape]
    digest = sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return {
        "sha256": digest.hexdigest(),
        "dtype": dtype,
        "shape": shape,
        "nbytes": int(contiguous.nbytes),
    }


def _with_identity_hash(value: dict[str, object]) -> dict[str, object]:
    """Add a digest for one canonical identity descriptor."""

    canonical = json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {**value, "sha256": sha256(canonical).hexdigest()}


def _metadata_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return _json_mapping(value)


def _qualified_type(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _single_likelihood_record(likelihood: object) -> dict[str, object]:
    """Return compact, JSON-safe likelihood configuration metadata."""

    record: dict[str, object] = {
        "type": type(likelihood).__name__,
        "name": str(getattr(likelihood, "name", type(likelihood).__name__)),
    }
    for attribute in (
        "include_normalization",
        "coordinate_rtol",
        "coordinate_atol",
        "calibration_scale",
        "calibration_offset",
        "calibration_scale_parameter",
        "calibration_offset_parameter",
        "uncertainty_scale",
        "uncertainty_scale_parameter",
        "offset_parameter",
        "jitter_parameter",
        "degree",
        "detrend_degree",
        "profile_amplitude",
        "require_positive_amplitude",
        "rank_rtol",
        "n_components",
        "sigma_clip",
        "doppler_mode",
        "flux_ratio_scale",
        "scale_parameter",
        "scale_is_log10",
    ):
        if hasattr(likelihood, attribute):
            record[attribute] = getattr(likelihood, attribute)
    projection = getattr(likelihood, "projection", None)
    if projection is not None and hasattr(projection, "degree"):
        record["prepared_projection_degree"] = projection.degree
    return record


def write_run_manifest(manifest: RunManifest, output_dir: str | Path) -> Path:
    """Write a manifest atomically and return its path."""

    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RUN_MANIFEST_FILENAME
    temporary_path = directory / f".{RUN_MANIFEST_FILENAME}.tmp"
    try:
        temporary_path.write_text(
            json.dumps(
                manifest.to_mapping(), indent=2, sort_keys=True, allow_nan=False
            ),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except OSError as exc:
        raise RobertDataError(
            f"failed to write retrieval run manifest: {path}"
        ) from exc
    return path


def read_run_manifest(path_or_directory: str | Path) -> RunManifest:
    """Read and validate a previously written run manifest."""

    path = Path(path_or_directory).expanduser()
    if path.is_dir():
        path = path / RUN_MANIFEST_FILENAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return RunManifest(
            problem_name=value["problem_name"],
            method=value["method"],
            created_at_utc=value["created_at_utc"],
            config_hash=value["config_hash"],
            parameter_names=tuple(value["parameter_names"]),
            parameter_priors=tuple(value["parameter_priors"]),
            likelihood=value["likelihood"],
            problem_metadata=value["problem_metadata"],
            opacity_identifiers=value["opacity_identifiers"],
            settings=value["settings"],
            random_seed=value.get("random_seed"),
            observation_identity=value.get("observation_identity", {}),
            robert_version=value.get("robert_version", __version__),
            python_version=value.get("python_version", "unknown"),
            platform=value.get("platform", "unknown"),
            git_commit=value.get("git_commit"),
            git_dirty=value.get("git_dirty"),
            schema_version=value.get("schema_version", RUN_MANIFEST_SCHEMA_VERSION),
        )
    except (
        OSError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        RobertValidationError,
    ) as exc:
        raise RobertDataError(f"failed to read retrieval run manifest: {path}") from exc


def _json_mapping(values: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _json_value(value) for key, value in values.items()}


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else str(number)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    return repr(value)


def _git_state() -> tuple[str | None, bool | None]:
    repository = Path(__file__).resolve().parents[3]
    if not (repository / ".git").exists():
        return None, None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit or None, bool(status.strip())


__all__ = [
    "MULTINEST_DEFAULT_SETTINGS",
    "MULTINEST_PER_ATTEMPT_SETTINGS",
    "MULTINEST_STRUCTURAL_SETTINGS",
    "RUN_MANIFEST_FILENAME",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RunManifest",
    "build_run_manifest",
    "normalize_multinest_resume",
    "normalize_retrieval_method",
    "normalize_run_settings",
    "read_run_manifest",
    "resume_compatibility_changes",
    "validate_resume_compatibility",
    "write_run_manifest",
]
