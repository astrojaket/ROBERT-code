"""Published WASP-80b panchromatic emission-spectrum data loader."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from robert_exoplanets.core import RobertDataError
from robert_exoplanets.instruments import (
    Observation,
    ObservationCollection,
    ObservationDataset,
)


WISER2025_SHA256 = {
    "f322w2": "68574057ffa980e9599b9b3bce2d79982c2e6e537336ecf138cef34ed399e6d3",
    "f444w": "b5371718d621b2cc58a38687d1bbab49b64f38ae5f7899a4ba9e4f0138ac4ed0",
    "lrs": "c81ce562d4e11996dae0ba07ec302020baa9a40627162b4f2f2b3bef722befa9",
}
_FILES = {
    "f322w2": "WASP80b_eclipse_F322W2_Eureka.txt",
    "f444w": "WASP80b_eclipse_F444W_Eureka.txt",
    "lrs": "WASP80b_eclipse_LRS_Eureka.txt",
}
_EXPECTED_POINTS = {"f322w2": 100, "f444w": 72, "lrs": 27}


def load_wiser2025_wasp80b(
    directory: str | Path,
    *,
    verify_checksum: bool = True,
    miri_offset_parameter: str | None = "miri_offset",
) -> ObservationCollection:
    """Load the 199-point fiducial Eureka! eclipse spectrum from Wiser et al."""

    root = Path(directory).expanduser()
    datasets = []
    for name, filename in _FILES.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(path)
        if (
            verify_checksum
            and sha256(path.read_bytes()).hexdigest() != WISER2025_SHA256[name]
        ):
            raise RobertDataError(f"Wiser et al. WASP-80b {name} checksum mismatch")
        wavelength, half_width, depth, error_negative, error_positive = _read_ecsv(path)
        if wavelength.size != _EXPECTED_POINTS[name]:
            raise RobertDataError(
                f"Wiser et al. WASP-80b {name} must contain {_EXPECTED_POINTS[name]} rows"
            )
        # ROBERT's Gaussian likelihood is symmetric. Preserve both published
        # errors in metadata and use their arithmetic mean explicitly.
        uncertainty = 0.5 * (error_negative + error_positive)
        is_miri = name == "lrs"
        observation = Observation(
            wavelength=wavelength,
            flux=depth,
            uncertainty=uncertainty,
            wavelength_unit="micron",
            flux_unit="eclipse_depth",
            observable="eclipse_depth",
            instrument="JWST/MIRI-LRS" if is_miri else f"JWST/NIRCam-{name.upper()}",
            wavelength_bin_edges=_edges_from_half_width(wavelength, half_width),
            metadata={
                "source": f"Zenodo 10.5281/zenodo.13146949 {filename}",
                "doi": "10.1073/pnas.2504085122",
                "data_doi": "10.5281/zenodo.13146949",
                "reduction": "Eureka! fiducial",
                "published_half_widths_micron": ",".join(
                    f"{value:.8g}" for value in half_width
                ),
                "published_error_negative": ",".join(
                    f"{value:.17g}" for value in error_negative
                ),
                "published_error_positive": ",".join(
                    f"{value:.17g}" for value in error_positive
                ),
                "symmetric_uncertainty": "mean_of_published_negative_and_positive_errors",
                "checksum_sha256": WISER2025_SHA256[name],
            },
        )
        datasets.append(
            ObservationDataset(
                name=name,
                observation=observation,
                offset_parameter=miri_offset_parameter if is_miri else None,
                metadata={"calibration_group": "miri" if is_miri else "nircam"},
            )
        )
    return ObservationCollection(
        datasets=tuple(datasets),
        name="WASP-80b Wiser et al. panchromatic Eureka! eclipse spectrum",
        metadata={
            "source_archive": "Zenodo 10.5281/zenodo.13146949",
            "n_points": "199",
            "wavelength_range_micron": "2.4575-11.75",
        },
    )


def _read_ecsv(path: Path) -> tuple[np.ndarray, ...]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("wavelength ")
        ):
            continue
        fields = stripped.split()
        if len(fields) != 5:
            raise RobertDataError(
                f"invalid WASP-80b ECSV row {line_number} in {path.name}"
            )
        try:
            rows.append(tuple(float(value) for value in fields))
        except ValueError as exc:
            raise RobertDataError(
                f"invalid numeric value in WASP-80b ECSV row {line_number} in {path.name}"
            ) from exc
    if not rows:
        raise RobertDataError(f"WASP-80b ECSV file contains no data: {path}")
    columns = np.asarray(rows, dtype=float).T
    return tuple(np.asarray(column, dtype=float) for column in columns)


def _edges_from_half_width(
    wavelength: np.ndarray, half_width: np.ndarray
) -> np.ndarray:
    lower = wavelength - half_width
    upper = wavelength + half_width
    edges = np.empty(wavelength.size + 1, dtype=float)
    edges[0] = lower[0]
    edges[-1] = upper[-1]
    if wavelength.size > 1:
        edges[1:-1] = 0.5 * (upper[:-1] + lower[1:])
    return edges


__all__ = ["WISER2025_SHA256", "load_wiser2025_wasp80b"]
