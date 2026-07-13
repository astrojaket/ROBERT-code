"""Typed collections of independently calibrated spectral datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping

from .observation import Observation


@dataclass(frozen=True)
class ObservationDataset:
    """One named observation with optional calibration nuisance parameters."""

    name: str
    observation: Observation
    offset_parameter: str | None = None
    jitter_parameter: str | None = None
    uncertainty_scale_parameter: str | None = None
    uncertainty_scale: float = 1.0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("observation dataset name must not be empty")
        for parameter in (
            self.offset_parameter,
            self.jitter_parameter,
            self.uncertainty_scale_parameter,
        ):
            if parameter is not None and not parameter:
                raise RobertValidationError(
                    "dataset nuisance parameter names must not be empty"
                )
        uncertainty_scale = float(self.uncertainty_scale)
        if not np.isfinite(uncertainty_scale) or uncertainty_scale <= 0.0:
            raise RobertValidationError(
                "dataset uncertainty_scale must be finite and positive"
            )
        object.__setattr__(self, "uncertainty_scale", uncertainty_scale)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ObservationCollection:
    """Multiple datasets sharing physical parameters but retaining identity."""

    datasets: tuple[ObservationDataset, ...]
    name: str = "multi-dataset-observation"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        datasets = tuple(self.datasets)
        if not datasets:
            raise RobertValidationError(
                "observation collection must contain at least one dataset"
            )
        names = tuple(dataset.name for dataset in datasets)
        if len(set(names)) != len(names):
            raise RobertValidationError("observation dataset names must be unique")
        if not self.name:
            raise RobertValidationError("observation collection name must not be empty")
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def n_datasets(self) -> int:
        return len(self.datasets)

    @property
    def n_points(self) -> int:
        return sum(dataset.observation.n_points for dataset in self.datasets)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(dataset.name for dataset in self.datasets)


__all__ = ["ObservationCollection", "ObservationDataset"]
