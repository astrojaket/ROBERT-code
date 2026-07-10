"""Reproducible retrieval run manifests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

import numpy as np

from robert_exoplanets._version import __version__
from robert_exoplanets.core import RobertDataError, RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping

from .problem import RetrievalProblem

RUN_MANIFEST_SCHEMA_VERSION = "1.0"
RUN_MANIFEST_FILENAME = "manifest.json"


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

    def __post_init__(self) -> None:
        for name in ("problem_name", "method", "created_at_utc", "config_hash", "schema_version"):
            if not str(getattr(self, name)).strip():
                raise RobertValidationError(f"manifest {name} must not be empty")
        object.__setattr__(self, "parameter_names", tuple(self.parameter_names))
        object.__setattr__(
            self,
            "parameter_priors",
            tuple(immutable_mapping(dict(item)) for item in self.parameter_priors),
        )
        object.__setattr__(self, "likelihood", immutable_mapping(self.likelihood))
        object.__setattr__(self, "problem_metadata", immutable_mapping(self.problem_metadata))
        object.__setattr__(self, "opacity_identifiers", immutable_mapping(self.opacity_identifiers))
        object.__setattr__(self, "settings", immutable_mapping(self.settings))

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
        }


def build_run_manifest(
    problem: RetrievalProblem,
    *,
    method: str,
    settings: Mapping[str, object],
    random_seed: int | None,
) -> RunManifest:
    """Build a manifest before inference begins."""

    safe_settings = _json_mapping(settings)
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
    likelihood_record = _json_mapping(
        {
            "type": type(problem.likelihood).__name__,
            "name": problem.likelihood.name,
            "include_normalization": problem.likelihood.include_normalization,
            "offset_parameter": problem.likelihood.offset_parameter,
            "jitter_parameter": problem.likelihood.jitter_parameter,
            "coordinate_rtol": problem.likelihood.coordinate_rtol,
            "coordinate_atol": problem.likelihood.coordinate_atol,
        }
    )
    signature = {
        "problem_name": problem.name,
        "parameter_priors": prior_records,
        "likelihood": likelihood_record,
        "problem_metadata": dict(problem.metadata),
        "opacity_identifiers": dict(problem.opacity_identifiers),
        "method": method,
        "settings": safe_settings,
        "random_seed": random_seed,
    }
    encoded = json.dumps(_json_value(signature), sort_keys=True, separators=(",", ":")).encode("utf-8")
    git_commit, git_dirty = _git_state()
    return RunManifest(
        problem_name=problem.name,
        method=method,
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
    )


def write_run_manifest(manifest: RunManifest, output_dir: str | Path) -> Path:
    """Write a manifest atomically and return its path."""

    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RUN_MANIFEST_FILENAME
    temporary_path = directory / f".{RUN_MANIFEST_FILENAME}.tmp"
    try:
        temporary_path.write_text(
            json.dumps(manifest.to_mapping(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except OSError as exc:
        raise RobertDataError(f"failed to write retrieval run manifest: {path}") from exc
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
            robert_version=value.get("robert_version", __version__),
            python_version=value.get("python_version", "unknown"),
            platform=value.get("platform", "unknown"),
            git_commit=value.get("git_commit"),
            git_dirty=value.get("git_dirty"),
            schema_version=value.get("schema_version", RUN_MANIFEST_SCHEMA_VERSION),
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError, RobertValidationError) as exc:
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
    "RUN_MANIFEST_FILENAME",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RunManifest",
    "build_run_manifest",
    "read_run_manifest",
    "write_run_manifest",
]
