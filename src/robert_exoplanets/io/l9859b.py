"""Published L 98-59 b JWST/NIRSpec transmission-spectrum loader."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from robert_exoplanets.core import RobertDataError
from robert_exoplanets.instruments import (
    Observation,
    ObservationCollection,
    ObservationDataset,
    infer_wavelength_bin_edges,
)


BELLO_ARUFE2025_EUREKA_SHA256 = (
    "fb423903613e2ac8d729c31cbfb40b66f4441080e381c5e9d49451227673d672"
)
_FILENAME = "L9859b_combined_spectrum_eureka.txt"
_EXPECTED_POINTS = 218
_DETECTOR_GAP_MICRON = 0.05


def load_bello_arufe2025_l9859b(
    directory: str | Path,
    *,
    verify_checksum: bool = True,
    miri_offset_parameter: str | None = None,
) -> ObservationCollection:
    """Load the 218-point Eureka! G395H transmission spectrum.

    The upstream spectrum is split at its detector gap into ``nrs1`` and
    ``nrs2`` datasets so a configured inter-detector offset can be fitted.
    Published depths and uncertainties are converted from ppm to dimensionless
    transit depth.
    """

    root = Path(directory).expanduser()
    path = root / _FILENAME
    if not path.exists():
        raise FileNotFoundError(path)
    checksum = sha256(path.read_bytes()).hexdigest()
    if verify_checksum and checksum != BELLO_ARUFE2025_EUREKA_SHA256:
        raise RobertDataError("Bello-Arufe et al. L 98-59 b checksum mismatch")

    values = _read_spectrum(path)
    if values.shape != (_EXPECTED_POINTS, 3):
        raise RobertDataError(
            f"Bello-Arufe et al. L 98-59 b spectrum must contain "
            f"{_EXPECTED_POINTS} rows"
        )
    wavelength, depth_ppm, uncertainty_ppm = values.T
    gaps = np.flatnonzero(np.diff(wavelength) > _DETECTOR_GAP_MICRON)
    if gaps.size != 1:
        raise RobertDataError("L 98-59 b spectrum must contain one NRS1/NRS2 gap")

    datasets = []
    for name, indices in (
        ("nrs1", np.arange(0, gaps[0] + 1)),
        ("nrs2", np.arange(gaps[0] + 1, wavelength.size)),
    ):
        selected_wavelength = wavelength[indices]
        observation = Observation(
            wavelength=selected_wavelength,
            flux=depth_ppm[indices] * 1.0e-6,
            uncertainty=uncertainty_ppm[indices] * 1.0e-6,
            wavelength_unit="micron",
            flux_unit="transit_depth",
            observable="transit_depth",
            instrument=f"JWST/NIRSpec-G395H-{name.upper()}",
            wavelength_bin_edges=infer_wavelength_bin_edges(selected_wavelength),
            metadata={
                "source": f"Zenodo 10.5281/zenodo.14676143 {_FILENAME}",
                "doi": "10.3847/2041-8213/ada7f5",
                "data_doi": "10.5281/zenodo.14676143",
                "reduction": "Eureka! combined",
                "published_flux_unit": "ppm",
                "checksum_sha256": checksum,
            },
        )
        datasets.append(
            ObservationDataset(
                name=name,
                observation=observation,
                offset_parameter=(
                    miri_offset_parameter if name == "nrs2" else None
                ),
                metadata={"calibration_group": name},
            )
        )
    return ObservationCollection(
        datasets=tuple(datasets),
        name="L 98-59 b Bello-Arufe et al. 2025 Eureka! transmission spectrum",
        metadata={
            "source_archive": "Zenodo 10.5281/zenodo.14676143",
            "n_points": str(_EXPECTED_POINTS),
            "wavelength_range_micron": "2.875-5.165",
        },
    )


def _read_spectrum(path: Path) -> np.ndarray:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 3:
            raise RobertDataError(
                f"invalid L 98-59 b spectrum row {line_number} in {path.name}"
            )
        try:
            rows.append(tuple(float(value) for value in fields))
        except ValueError as exc:
            raise RobertDataError(
                f"invalid numeric value in L 98-59 b row {line_number}"
            ) from exc
    return np.asarray(rows, dtype=float)


__all__ = ["BELLO_ARUFE2025_EUREKA_SHA256", "load_bello_arufe2025_l9859b"]
