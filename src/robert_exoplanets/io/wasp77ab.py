"""Published WASP-77Ab JWST/NIRSpec G395H eclipse-spectrum loader.

The checked-in table is the NIRSpec/G395H spectrum published by
August et al. (2023).  It is a low-resolution, binned input for the planned
Smith et al. (2024) comparison; it is not an HRS spectrum produced by Smith
et al.  The archive table uses the VizieR/Firefly pipe-delimited format, so it
needs a small parser before it can enter ROBERT's typed observation model.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re

import numpy as np

from robert_exoplanets.core import RobertDataError
from robert_exoplanets.instruments import (
    Observation,
    ObservationCollection,
    ObservationDataset,
)


AUGUST2023_WASP77AB_SHA256 = (
    "96159824870eb7c8132fd9c8bee30caf9ec95f05e128dbe2fea8627909a40d3f"
)
WASP77AB_AUGUST2023_SHA256 = AUGUST2023_WASP77AB_SHA256
_EXPECTED_POINTS = 150
_DETECTOR_SPLIT_INDEX = 70
_DETECTOR_SPLIT_AFTER_INDEX = _DETECTOR_SPLIT_INDEX - 1
_DETECTOR_GAP_MICRON = 0.119
_TABLE_FILENAME = "WASP_77_A_b_3.11569_5329_1.tbl"
_TABLE_RELATIVE_PATH = Path("spectra/76/89/90/56") / _TABLE_FILENAME
_EXPECTED_COLUMNS = (
    "CENTRALWAVELNG",
    "BANDWIDTH",
    "ESPECLIPDEP",
    "ESPECLIPDEPERR1",
    "ESPECLIPDEPERR2",
    "ESPECLIPDEPLIM",
    "ESPBRITEMP",
    "ESPBRITEMPERR1",
    "ESPBRITEMPERR2",
    "ESPBRITEMPLIM",
    "OBS_DATE",
    "OBS_DATEERR1",
    "OBS_DATEERR2",
)
_NUMERIC_COLUMNS = 5
_DATA_LINE = re.compile(r"^\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")


def load_august2023_wasp77ab(
    path: str | Path,
    *,
    verify_checksum: bool = True,
    miri_offset_parameter: str | None = None,
) -> ObservationCollection:
    """Load the 150-point August et al. (2023) WASP-77Ab eclipse spectrum.

    ``path`` can be the archive root (``data/jwst_emission_spectra``), the
    directory containing the table, or the table itself.  The published
    eclipse depth and both published uncertainty sides are in percent.  The
    returned observations use dimensionless eclipse depth and the arithmetic
    mean of the absolute asymmetric errors as their symmetric Gaussian
    uncertainty.  Both original error columns remain in metadata.  The
    detector gap is represented by two data sets, ``nirspec_g395h_nrs1`` and
    ``nirspec_g395h_nrs2``; no combined data set is created.

    The optional ``miri_offset_parameter`` is accepted for compatibility with
    the configured published-data loader interface.  This NIRSpec data set has
    two detector segments, so no MIRI offset parameter is attached.
    """

    del miri_offset_parameter
    table_path = _resolve_table_path(path)
    raw = table_path.read_bytes()
    checksum = sha256(raw).hexdigest()
    if verify_checksum and checksum != AUGUST2023_WASP77AB_SHA256:
        raise RobertDataError(
            "August et al. WASP-77Ab NIRSpec table checksum mismatch"
        )

    metadata, columns, rows = _read_table(table_path)
    if metadata.get("PL_NAME") != "WASP-77 A b":
        raise RobertDataError("WASP-77Ab table has an unexpected planet name")
    if metadata.get("SPEC_TYPE") != "Eclipse":
        raise RobertDataError("WASP-77Ab table is not an eclipse spectrum")
    if metadata.get("REFERENCE") != "August et al. 2023":
        raise RobertDataError("WASP-77Ab table has an unexpected reference")
    if columns != _EXPECTED_COLUMNS:
        raise RobertDataError("WASP-77Ab table has unexpected columns")
    if len(rows) != _EXPECTED_POINTS:
        raise RobertDataError(
            f"August et al. WASP-77Ab spectrum must contain {_EXPECTED_POINTS} rows"
        )

    values = _numeric_columns(rows, table_path)
    wavelength = values[:, 0]
    bandwidth = values[:, 1]
    depth_percent = values[:, 2]
    error_1_percent = values[:, 3]
    error_2_percent = values[:, 4]
    _validate_source_values(
        wavelength,
        bandwidth,
        depth_percent,
        error_1_percent,
        error_2_percent,
    )
    uncertainty_percent = 0.5 * (
        np.abs(error_1_percent) + np.abs(error_2_percent)
    )
    if np.any(uncertainty_percent <= 0.0):
        raise RobertDataError("WASP-77Ab table contains non-positive uncertainties")

    _validate_detector_split(wavelength, bandwidth)
    common_metadata = {
        "source": "VizieR/MAST archive table, August et al. 2023, Table 2",
        "bibcode": "2023ApJ...953L..24A",
        "data_producer": "August et al. 2023",
        "comparison_input": "Smith et al. 2024",
        "data_role": "August et al. 2023 / Smith et al. 2024 input",
        "smith_produced_data": "false",
        "archive_table": _TABLE_FILENAME,
        "archive_format": "VizieR/Firefly pipe-delimited table",
        "source_path": str(table_path),
        "checksum_sha256": checksum,
        "published_flux_unit": "percent eclipse depth",
        "symmetric_uncertainty": (
            "mean_of_absolute_published_error_1_and_error_2"
        ),
        "bin_edge_policy": (
            "published bandwidth at outer edges; midpoint boundaries between "
            "non-contiguous published bins"
        ),
        "detector_split_index": str(_DETECTOR_SPLIT_INDEX),
        "detector_split_after_index": str(_DETECTOR_SPLIT_AFTER_INDEX),
        "detector_gap_micron": f"{_DETECTOR_GAP_MICRON:g}",
        "detector_gap_edges_micron": "3.712-3.831",
    }
    datasets = []
    for name, detector, start, stop in (
        ("nirspec_g395h_nrs1", "NRS1", 0, _DETECTOR_SPLIT_INDEX),
        ("nirspec_g395h_nrs2", "NRS2", _DETECTOR_SPLIT_INDEX, None),
    ):
        selected = slice(start, stop)
        segment_wavelength = wavelength[selected]
        segment_bandwidth = bandwidth[selected]
        segment_error_1 = error_1_percent[selected]
        segment_error_2 = error_2_percent[selected]
        segment_observation_metadata = {
            **common_metadata,
            "detector": detector,
            "detector_point_count": str(segment_wavelength.size),
            "published_bandwidths_micron": _csv(segment_bandwidth),
            "published_error_1_percent": _csv(segment_error_1),
            "published_error_2_percent": _csv(segment_error_2),
            "published_error_1_absolute_percent": _csv(np.abs(segment_error_1)),
            "published_error_2_absolute_percent": _csv(np.abs(segment_error_2)),
        }
        observation = Observation(
            wavelength=segment_wavelength,
            flux=depth_percent[selected] * 1.0e-2,
            uncertainty=uncertainty_percent[selected] * 1.0e-2,
            wavelength_unit="micron",
            flux_unit="eclipse_depth",
            observable="eclipse_depth",
            instrument=f"JWST/NIRSpec-G395H-{detector}",
            wavelength_bin_edges=_edges_from_bandwidth(
                segment_wavelength, segment_bandwidth
            ),
            metadata={
                **segment_observation_metadata,
                "wavelength_range_micron": (
                    f"{segment_wavelength[0]:g}-{segment_wavelength[-1]:g}"
                ),
            },
        )
        datasets.append(
            ObservationDataset(
                name=name,
                observation=observation,
                metadata={
                    "calibration_group": name,
                    "detector": detector,
                    "data_role": "August et al. 2023 / Smith et al. 2024 input",
                    "smith_produced_data": "false",
                },
            )
        )
    return ObservationCollection(
        datasets=tuple(datasets),
        name="WASP-77Ab August et al. 2023 NIRSpec G395H eclipse spectrum",
        metadata={
            "source_archive": "VizieR/MAST JWST emission-spectrum archive",
            "bibcode": "2023ApJ...953L..24A",
            "data_producer": "August et al. 2023",
            "comparison_input": "Smith et al. 2024",
            "data_role": "August et al. 2023 / Smith et al. 2024 input",
            "smith_produced_data": "false",
            "checksum_sha256": checksum,
            "n_points": str(_EXPECTED_POINTS),
            "detector_split_index": str(_DETECTOR_SPLIT_INDEX),
            "detector_split_after_index": str(_DETECTOR_SPLIT_AFTER_INDEX),
            "detector_gap_micron": f"{_DETECTOR_GAP_MICRON:g}",
            "detector_gap_edges_micron": "3.712-3.831",
            "wavelength_range_micron": (
                f"{wavelength[0]:g}-{wavelength[-1]:g}"
            ),
        },
    )


def _resolve_table_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        options = (
            candidate / _TABLE_RELATIVE_PATH,
            candidate / _TABLE_FILENAME,
        )
        for option in options:
            if option.is_file():
                return option
    raise FileNotFoundError(
        "WASP-77Ab NIRSpec table was not found at "
        f"{candidate} or beneath its archive root"
    )


def _read_table(path: Path) -> tuple[dict[str, str], tuple[str, ...], list[list[str]]]:
    metadata: dict[str, str] = {}
    columns: tuple[str, ...] | None = None
    rows: list[list[str]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("\\"):
            key, separator, value = stripped[1:].partition("=")
            if not separator:
                raise RobertDataError(
                    f"invalid WASP-77Ab metadata line {line_number}"
                )
            metadata[key.strip()] = _unquote(value.strip())
            continue
        if stripped.startswith("|"):
            fields = tuple(item.strip() for item in stripped.strip("|").split("|"))
            if columns is None and fields and fields[0] == "CENTRALWAVELNG":
                columns = fields
            continue
        if _DATA_LINE.match(line):
            if columns is None:
                raise RobertDataError(
                    f"WASP-77Ab data appears before its header at line {line_number}"
                )
            fields = stripped.split()
            if len(fields) != len(columns):
                raise RobertDataError(
                    f"invalid WASP-77Ab row {line_number}: expected "
                    f"{len(columns)} columns, found {len(fields)}"
                )
            rows.append(fields)
            continue
        raise RobertDataError(f"invalid WASP-77Ab table line {line_number}")
    if columns is None:
        raise RobertDataError("WASP-77Ab table is missing its column header")
    return metadata, columns, rows


def _numeric_columns(rows: list[list[str]], path: Path) -> np.ndarray:
    try:
        values = np.asarray(
            [[float(value) for value in row[:_NUMERIC_COLUMNS]] for row in rows],
            dtype=float,
        )
    except ValueError as exc:
        raise RobertDataError(
            f"invalid numeric value in WASP-77Ab table {path.name}"
        ) from exc
    if values.shape != (_EXPECTED_POINTS, _NUMERIC_COLUMNS):
        raise RobertDataError("WASP-77Ab table has invalid numeric columns")
    return values


def _validate_source_values(
    wavelength: np.ndarray,
    bandwidth: np.ndarray,
    depth_percent: np.ndarray,
    error_1_percent: np.ndarray,
    error_2_percent: np.ndarray,
) -> None:
    if not all(
        np.all(np.isfinite(values))
        for values in (
            wavelength,
            bandwidth,
            depth_percent,
            error_1_percent,
            error_2_percent,
        )
    ):
        raise RobertDataError("WASP-77Ab source values must be finite")
    if np.any(wavelength <= 0.0) or np.any(bandwidth <= 0.0):
        raise RobertDataError("WASP-77Ab wavelengths and bandwidths must be positive")
    difference = np.diff(wavelength)
    if not np.all(difference > 0.0):
        raise RobertDataError("WASP-77Ab wavelengths must be strictly increasing")


def _validate_detector_split(
    wavelength: np.ndarray, bandwidth: np.ndarray
) -> None:
    lower = wavelength - 0.5 * bandwidth
    upper = wavelength + 0.5 * bandwidth
    gap = lower[_DETECTOR_SPLIT_INDEX] - upper[_DETECTOR_SPLIT_AFTER_INDEX]
    if not np.isclose(gap, _DETECTOR_GAP_MICRON, rtol=0.0, atol=1.0e-12):
        raise RobertDataError(
            "WASP-77Ab NRS1/NRS2 detector gap is not the expected "
            f"{_DETECTOR_GAP_MICRON:g} micron"
        )
    if not np.isclose(
        upper[_DETECTOR_SPLIT_AFTER_INDEX], 3.712, rtol=0.0, atol=1.0e-12
    ) or not np.isclose(
        lower[_DETECTOR_SPLIT_INDEX], 3.831, rtol=0.0, atol=1.0e-12
    ):
        raise RobertDataError(
            "WASP-77Ab detector gap edges are not the expected 3.712/3.831 micron"
        )


def _edges_from_bandwidth(
    wavelength: np.ndarray, bandwidth: np.ndarray
) -> np.ndarray:
    lower = wavelength - 0.5 * bandwidth
    upper = wavelength + 0.5 * bandwidth
    edges = np.empty(wavelength.size + 1, dtype=float)
    edges[0] = lower[0]
    edges[-1] = upper[-1]
    if wavelength.size > 1:
        edges[1:-1] = 0.5 * (upper[:-1] + lower[1:])
    return edges


def _csv(values: np.ndarray) -> str:
    return ",".join(f"{value:.17g}" for value in values)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


__all__ = [
    "AUGUST2023_WASP77AB_SHA256",
    "WASP77AB_AUGUST2023_SHA256",
    "load_august2023_wasp77ab",
]
