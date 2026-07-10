"""Retrieval-facing observation loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robert_exoplanets.core import RobertDataError
from robert_exoplanets.instruments import Observation, infer_wavelength_bin_edges


def load_emission_observation_npz(
    path: str | Path,
    *,
    wavelength_key: str = "wavelength",
    flux_key: str = "data",
    uncertainty_key: str = "err",
    wavelength_bin_edges_key: str | None = None,
    infer_bin_edges: bool = True,
    wavelength_unit: str = "micron",
    flux_unit: str = "eclipse_depth",
    observable: str = "eclipse_depth",
    instrument: str | None = None,
) -> Observation:
    """Load a 1D emission observation from an `.npz` file.

    The default keys match the local HAT-P-32b retrieval benchmark product:
    `wavelength`, `data`, and `err`.
    """

    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise RobertDataError(f"emission observation NPZ does not exist: {file_path}")
    with np.load(file_path, allow_pickle=False) as archive:
        keys = tuple(str(key) for key in archive.files)
        for key, label in (
            (wavelength_key, "wavelength"),
            (flux_key, "flux"),
            (uncertainty_key, "uncertainty"),
        ):
            if key not in keys:
                raise RobertDataError(f"emission observation NPZ is missing {label} key {key!r}")
        wavelength = np.array(archive[wavelength_key], dtype=float, copy=True)
        flux = np.array(archive[flux_key], dtype=float, copy=True)
        uncertainty = np.array(archive[uncertainty_key], dtype=float, copy=True)
        wavelength_bin_edges = None
        if wavelength_bin_edges_key is not None:
            if wavelength_bin_edges_key not in keys:
                raise RobertDataError(
                    f"emission observation NPZ is missing wavelength-bin-edge key {wavelength_bin_edges_key!r}"
                )
            wavelength_bin_edges = np.array(archive[wavelength_bin_edges_key], dtype=float, copy=True)

    if wavelength_bin_edges is None and infer_bin_edges:
        wavelength_bin_edges = infer_wavelength_bin_edges(wavelength)

    return Observation(
        wavelength=wavelength,
        flux=flux,
        uncertainty=uncertainty,
        wavelength_unit=wavelength_unit,
        flux_unit=flux_unit,
        observable=observable,
        instrument=instrument,
        wavelength_bin_edges=wavelength_bin_edges,
        metadata={
            "source_path": str(file_path),
            "source_format": "npz_emission_observation",
            "wavelength_key": wavelength_key,
            "flux_key": flux_key,
            "uncertainty_key": uncertainty_key,
            "wavelength_bin_edges": (
                wavelength_bin_edges_key if wavelength_bin_edges_key is not None else "inferred_midpoints"
            ),
        },
    )


__all__ = ["Observation", "load_emission_observation_npz"]
