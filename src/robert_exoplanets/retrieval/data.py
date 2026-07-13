"""Retrieval-facing observation loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robert_exoplanets.core import RobertDataError
from robert_exoplanets.instruments import Observation, infer_wavelength_bin_edges


ROBERT_OBSERVATION_SCHEMA = "robert-emission-observation-v1"


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
                raise RobertDataError(
                    f"emission observation NPZ is missing {label} key {key!r}"
                )
        wavelength = np.array(archive[wavelength_key], dtype=float, copy=True)
        flux = np.array(archive[flux_key], dtype=float, copy=True)
        uncertainty = np.array(archive[uncertainty_key], dtype=float, copy=True)
        wavelength_bin_edges = None
        if wavelength_bin_edges_key is None and "wavelength_bin_edges" in keys:
            wavelength_bin_edges_key = "wavelength_bin_edges"
        if wavelength_bin_edges_key is not None:
            if wavelength_bin_edges_key not in keys:
                raise RobertDataError(
                    f"emission observation NPZ is missing wavelength-bin-edge key {wavelength_bin_edges_key!r}"
                )
            wavelength_bin_edges = np.array(
                archive[wavelength_bin_edges_key], dtype=float, copy=True
            )
        mask = (
            None
            if "mask" not in keys
            else np.array(archive["mask"], dtype=bool, copy=True)
        )
        stored_metadata: dict[str, str] = {}
        if "metadata_json" in keys:
            try:
                decoded = json.loads(str(archive["metadata_json"]))
                if isinstance(decoded, dict):
                    stored_metadata = {
                        str(key): str(value) for key, value in decoded.items()
                    }
            except (json.JSONDecodeError, TypeError):
                stored_metadata = {}
        if instrument is None and "instrument" in keys:
            stored_instrument = str(archive["instrument"])
            instrument = stored_instrument or None

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
        mask=mask,
        wavelength_bin_edges=wavelength_bin_edges,
        metadata={
            **stored_metadata,
            "source_path": str(file_path),
            "source_format": "npz_emission_observation",
            "wavelength_key": wavelength_key,
            "flux_key": flux_key,
            "uncertainty_key": uncertainty_key,
            "wavelength_bin_edges": (
                wavelength_bin_edges_key
                if wavelength_bin_edges_key is not None
                else "inferred_midpoints"
            ),
        },
    )


def save_emission_observation_npz(
    observation: Observation,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write an observation in ROBERT's portable emission-spectrum format."""

    output = Path(path).expanduser()
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema": ROBERT_OBSERVATION_SCHEMA,
        "wavelength": observation.wavelength,
        "data": observation.flux,
        "err": observation.uncertainty,
        "wavelength_unit": observation.wavelength_unit,
        "flux_unit": observation.flux_unit,
        "observable": observation.observable,
        "instrument": "" if observation.instrument is None else observation.instrument,
        "metadata_json": json.dumps(dict(observation.metadata), sort_keys=True),
    }
    if observation.mask is not None:
        payload["mask"] = observation.mask
    if observation.wavelength_bin_edges is not None:
        payload["wavelength_bin_edges"] = observation.wavelength_bin_edges
    np.savez_compressed(output, **payload)
    return output


def load_emission_observation_table(
    path: str | Path,
    *,
    wavelength_column: str = "wavelength",
    flux_column: str = "flux",
    uncertainty_column: str = "uncertainty",
    bin_low_column: str | None = None,
    bin_high_column: str | None = None,
    delimiter: str | None = None,
    wavelength_input_unit: str = "micron",
    flux_input_unit: str = "eclipse_depth",
    instrument: str | None = None,
    observable: str = "eclipse_depth",
) -> Observation:
    """Read a named-column text table and normalize it to ROBERT units.

    Supported wavelength inputs are micron, nm, angstrom, and m. Supported
    eclipse-depth inputs are fractional eclipse depth, percent, and ppm.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise RobertDataError(f"observation table does not exist: {source}")
    try:
        table = np.genfromtxt(
            source,
            names=True,
            delimiter=delimiter,
            dtype=None,
            encoding="utf-8",
            comments="#",
        )
    except (OSError, ValueError) as exc:
        raise RobertDataError(f"could not read observation table: {source}") from exc
    if table.dtype.names is None:
        raise RobertDataError("observation table must contain a named header row")
    names = set(table.dtype.names)
    required = {wavelength_column, flux_column, uncertainty_column}
    if bin_low_column is not None:
        required.add(bin_low_column)
    if bin_high_column is not None:
        required.add(bin_high_column)
    missing = sorted(required - names)
    if missing:
        raise RobertDataError(
            "observation table is missing columns: " + ", ".join(missing)
        )
    if (bin_low_column is None) != (bin_high_column is None):
        raise RobertDataError(
            "bin_low_column and bin_high_column must be supplied together"
        )

    wavelength_factor = _wavelength_to_micron_factor(wavelength_input_unit)
    flux_factor = _eclipse_depth_factor(flux_input_unit)
    wavelength = (
        np.atleast_1d(np.asarray(table[wavelength_column], dtype=float))
        * wavelength_factor
    )
    flux = np.atleast_1d(np.asarray(table[flux_column], dtype=float)) * flux_factor
    uncertainty = (
        np.atleast_1d(np.asarray(table[uncertainty_column], dtype=float)) * flux_factor
    )
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    flux = flux[order]
    uncertainty = uncertainty[order]
    edges = None
    edge_method = "inferred_midpoints"
    if bin_low_column is not None and bin_high_column is not None:
        low = np.atleast_1d(np.asarray(table[bin_low_column], dtype=float))[order]
        high = np.atleast_1d(np.asarray(table[bin_high_column], dtype=float))[order]
        low *= wavelength_factor
        high *= wavelength_factor
        if np.any(high <= low):
            raise RobertDataError(
                "every wavelength-bin upper edge must exceed its lower edge"
            )
        edges = np.empty(wavelength.size + 1, dtype=float)
        edges[0] = low[0]
        edges[-1] = high[-1]
        if wavelength.size > 1:
            edges[1:-1] = 0.5 * (high[:-1] + low[1:])
        edge_method = "published_lower_upper_columns_joined_at_midpoints"
    elif wavelength.size > 1:
        edges = infer_wavelength_bin_edges(wavelength)

    return Observation(
        wavelength=wavelength,
        flux=flux,
        uncertainty=uncertainty,
        wavelength_unit="micron",
        flux_unit="eclipse_depth",
        observable=observable,
        instrument=instrument,
        wavelength_bin_edges=edges,
        metadata={
            "source_path": str(source),
            "source_format": "named_text_table",
            "source_wavelength_unit": wavelength_input_unit,
            "source_flux_unit": flux_input_unit,
            "wavelength_column": wavelength_column,
            "flux_column": flux_column,
            "uncertainty_column": uncertainty_column,
            "wavelength_bin_edges": edge_method,
        },
    )


def convert_emission_observation_table(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    **table_options: object,
) -> Path:
    """Convert a named-column table directly to a ROBERT observation NPZ."""

    observation = load_emission_observation_table(input_path, **table_options)
    return save_emission_observation_npz(observation, output_path, overwrite=overwrite)


def _wavelength_to_micron_factor(unit: str) -> float:
    normalized = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    factors = {
        "micron": 1.0,
        "microns": 1.0,
        "um": 1.0,
        "nm": 1.0e-3,
        "angstrom": 1.0e-4,
        "angstroms": 1.0e-4,
        "a": 1.0e-4,
        "m": 1.0e6,
    }
    try:
        return factors[normalized]
    except KeyError as exc:
        raise RobertDataError(f"unsupported wavelength input unit: {unit}") from exc


def _eclipse_depth_factor(unit: str) -> float:
    normalized = unit.strip().lower()
    factors = {
        "eclipse_depth": 1.0,
        "fraction": 1.0,
        "fractional": 1.0,
        "percent": 1.0e-2,
        "%": 1.0e-2,
        "ppm": 1.0e-6,
    }
    try:
        return factors[normalized]
    except KeyError as exc:
        raise RobertDataError(f"unsupported eclipse-depth input unit: {unit}") from exc


__all__ = [
    "Observation",
    "ROBERT_OBSERVATION_SCHEMA",
    "convert_emission_observation_table",
    "load_emission_observation_npz",
    "load_emission_observation_table",
    "save_emission_observation_npz",
]
