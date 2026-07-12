"""Published WASP-69b emission-spectrum data loaders."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from robert_exoplanets.core import RobertDataError
from robert_exoplanets.instruments import Observation, ObservationCollection, ObservationDataset

SCHLAWIN2024_SP_SHA256 = "ee57f0c0163d3d10e4b58896d5a7c4a5ccee432b0401838213b5208949bcdf0c"
_DATASET_ORDER = ("F322W2", "Avg", "F444W", "LRS")


def load_schlawin2024_wasp69b(
    directory: str | Path,
    *,
    verify_checksum: bool = True,
    miri_offset_parameter: str | None = "miri_offset",
) -> ObservationCollection:
    """Load the native 280-point Schlawin et al. (2024) eclipse spectrum."""

    root = Path(directory).expanduser()
    path = root / "sp.dat"
    if not path.exists():
        raise FileNotFoundError(path)
    if verify_checksum:
        checksum = sha256(path.read_bytes()).hexdigest()
        if checksum != SCHLAWIN2024_SP_SHA256:
            raise RobertDataError("Schlawin et al. (2024) sp.dat checksum mismatch")

    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise RobertDataError(f"invalid sp.dat row {line_number}")
        try:
            rows.append(
                (
                    float(fields[0]),
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                    fields[4],
                )
            )
        except ValueError as exc:
            raise RobertDataError(f"invalid numeric value in sp.dat row {line_number}") from exc
    if len(rows) != 280:
        raise RobertDataError("Schlawin et al. (2024) sp.dat must contain 280 rows")

    datasets = []
    for label in _DATASET_ORDER:
        selected = [row for row in rows if row[4] == label]
        wavelength = np.asarray([row[0] for row in selected], dtype=float)
        width = np.asarray([row[1] for row in selected], dtype=float)
        depth = np.asarray([row[2] for row in selected], dtype=float)
        uncertainty = np.asarray([row[3] for row in selected], dtype=float)
        is_miri = label == "LRS"
        observation = Observation(
            wavelength=wavelength,
            flux=depth,
            uncertainty=uncertainty,
            wavelength_unit="micron",
            flux_unit="eclipse_depth",
            observable="eclipse_depth",
            instrument="JWST/MIRI-LRS" if is_miri else f"JWST/NIRCam-{label}",
            wavelength_bin_edges=_contiguous_edges(wavelength, width),
            metadata={
                "source": "VizieR J/AJ/168/104 sp.dat",
                "bibcode": "2024AJ....168..104S",
                "doi": "10.3847/1538-3881/ad58e0",
                "mast_doi": "10.17909/v2v9-k243",
                "filter_label": label,
                "published_widths_micron": ",".join(f"{value:.8g}" for value in width),
                "checksum_sha256": SCHLAWIN2024_SP_SHA256,
            },
        )
        datasets.append(
            ObservationDataset(
                name=label.lower(),
                observation=observation,
                offset_parameter=miri_offset_parameter if is_miri else None,
                metadata={"calibration_group": "miri" if is_miri else "nircam"},
            )
        )
    return ObservationCollection(
        datasets=tuple(datasets),
        name="WASP-69b Schlawin et al. 2024 native eclipse spectrum",
        metadata={
            "source_catalog": "VizieR J/AJ/168/104",
            "n_points": "280",
            "wavelength_range_micron": "2.45477-11.875",
        },
    )


def _contiguous_edges(wavelength: np.ndarray, width: np.ndarray) -> np.ndarray:
    lower = wavelength - 0.5 * width
    upper = wavelength + 0.5 * width
    edges = np.empty(wavelength.size + 1, dtype=float)
    edges[0] = lower[0]
    edges[-1] = upper[-1]
    if wavelength.size > 1:
        edges[1:-1] = 0.5 * (upper[:-1] + lower[1:])
    return edges


__all__ = ["SCHLAWIN2024_SP_SHA256", "load_schlawin2024_wasp69b"]
