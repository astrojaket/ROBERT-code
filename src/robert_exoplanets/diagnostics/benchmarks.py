"""Utilities for reading external benchmark spectra."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.core import RobertDataError, RobertValidationError, SpectralGrid, Spectrum
from robert_exoplanets.core._immutability import immutable_mapping


def _readonly_1d(values: list[float], name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise RobertValidationError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise RobertValidationError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class EmissionBenchmark:
    """External emission benchmark spectrum and optional reference curves."""

    name: str
    wavelength_micron: NDArray[np.float64]
    eclipse_depth: NDArray[np.float64]
    references: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)
    source_path: Path | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("benchmark name must not be empty")
        wavelength = np.array(self.wavelength_micron, dtype=float, copy=True)
        eclipse_depth = np.array(self.eclipse_depth, dtype=float, copy=True)
        if wavelength.ndim != 1 or eclipse_depth.ndim != 1:
            raise RobertValidationError("benchmark arrays must be one-dimensional")
        if wavelength.shape != eclipse_depth.shape:
            raise RobertValidationError("benchmark wavelength and value arrays must match")
        if wavelength.size == 0:
            raise RobertValidationError("benchmark must contain at least one point")
        if np.any(wavelength <= 0.0):
            raise RobertValidationError("benchmark wavelengths must be positive")
        if wavelength.size > 1 and not np.all(np.diff(wavelength) > 0.0):
            raise RobertValidationError("benchmark wavelengths must be strictly increasing")
        if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(eclipse_depth)):
            raise RobertValidationError("benchmark arrays must contain only finite values")

        references: dict[str, NDArray[np.float64]] = {}
        for label, values in self.references.items():
            if not label:
                raise RobertValidationError("benchmark reference labels must not be empty")
            reference = np.array(values, dtype=float, copy=True)
            if reference.shape != wavelength.shape:
                raise RobertValidationError("benchmark reference arrays must match wavelengths")
            if not np.all(np.isfinite(reference)):
                raise RobertValidationError("benchmark reference arrays must be finite")
            reference.setflags(write=False)
            references[str(label)] = reference

        wavelength.setflags(write=False)
        eclipse_depth.setflags(write=False)
        object.__setattr__(self, "wavelength_micron", wavelength)
        object.__setattr__(self, "eclipse_depth", eclipse_depth)
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def n_points(self) -> int:
        """Number of benchmark spectral points."""

        return int(self.wavelength_micron.size)

    def to_spectrum(self) -> Spectrum:
        """Return the benchmark emission as a ROBERT spectrum."""

        return Spectrum(
            spectral_grid=SpectralGrid.from_array(self.wavelength_micron, role="benchmark"),
            values=self.eclipse_depth,
            unit="eclipse_depth",
            observable="eclipse_depth",
            metadata={
                "benchmark": self.name,
                "source_path": "" if self.source_path is None else str(self.source_path),
            },
        )


def load_emission_benchmark_csv(
    path: str | Path,
    *,
    name: str | None = None,
    wavelength_column: str = "wavelength_um",
    value_column: str = "value",
    reference_prefix: str = "bb_",
) -> EmissionBenchmark:
    """Load an external emission benchmark CSV.

    The expected value convention is eclipse depth as a dimensionless fraction.
    Columns with `reference_prefix` are carried as named reference curves.
    """

    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        raise RobertDataError(f"benchmark CSV does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RobertDataError("benchmark CSV is missing a header row")
        missing = {wavelength_column, value_column}.difference(reader.fieldnames)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise RobertDataError(f"benchmark CSV is missing required columns: {missing_columns}")

        reference_columns = tuple(
            column for column in reader.fieldnames if column.startswith(reference_prefix)
        )
        wavelength_values: list[float] = []
        eclipse_depth_values: list[float] = []
        references: dict[str, list[float]] = {column: [] for column in reference_columns}

        for row_number, row in enumerate(reader, start=2):
            try:
                wavelength_values.append(float(row[wavelength_column]))
                eclipse_depth_values.append(float(row[value_column]))
                for column in reference_columns:
                    references[column].append(float(row[column]))
            except (TypeError, ValueError) as exc:
                raise RobertDataError(f"invalid numeric value in benchmark CSV row {row_number}") from exc

    wavelength = _readonly_1d(wavelength_values, "benchmark wavelength")
    eclipse_depth = _readonly_1d(eclipse_depth_values, "benchmark eclipse depth")
    reference_arrays = {
        column: _readonly_1d(values, f"{column} benchmark reference")
        for column, values in references.items()
    }
    return EmissionBenchmark(
        name=name or csv_path.stem,
        wavelength_micron=wavelength,
        eclipse_depth=eclipse_depth,
        references=reference_arrays,
        source_path=csv_path,
    )
