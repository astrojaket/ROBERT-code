"""Safe loader for the Smith et al. (2024) WASP-77Ab HRS cube.

The Smith supplementary Zenodo record contains ``cube14v3.pic`` and its
frame metadata CSV.  The pickle is an old NumPy protocol-2 list containing
``(order_wavelengths, flux_cube)``.  It is external input, not package data,
so the loader verifies the file identity before deserialising it and uses a
restricted unpickler.  Only the NumPy globals emitted by this exact file
format are allowed.

No abundance conversion is performed here.  ROBERT model inputs remain
volume mixing ratios; this module only loads measured detector values and
their velocity metadata.
"""

from __future__ import annotations

import csv
from hashlib import md5, sha256
from pathlib import Path
import pickle
from types import MappingProxyType
from typing import Any, BinaryIO, Mapping

import numpy as np
from numpy.typing import ArrayLike

from robert_exoplanets.core import RobertDataError
from robert_exoplanets.instruments.time_resolved_high_resolution import (
    HighResolutionEmissionTemplate,
    TimeResolvedHighResolutionObservation,
)


SMITH2024_WASP77AB_ZENODO_DOI = "10.5281/zenodo.10382053"
SMITH2024_WASP77AB_ZENODO_RECORD = "https://zenodo.org/records/10382053"
SMITH2024_WASP77AB_CUBE14_FILENAME = "cube14v3.pic"
SMITH2024_WASP77AB_INFO_FILENAME = "20201214_info.csv"
SMITH2024_WASP77AB_CUBE14_URL = (
    "https://zenodo.org/api/records/10382053/files/cube14v3.pic/content"
)
SMITH2024_WASP77AB_INFO_URL = (
    "https://zenodo.org/api/records/10382053/files/20201214_info.csv/content"
)

SMITH2024_WASP77AB_CUBE14_SIZE = 77_591_894
SMITH2024_WASP77AB_CUBE14_MD5 = "3fd41560b7dbcc60851ef3a0972c852f"
SMITH2024_WASP77AB_CUBE14_SHA256 = (
    "999623a7ba4460aa72ba9bbe284de92e7b5a811ebe09e51d70667084ae592b26"
)
SMITH2024_WASP77AB_FLUX_RATIO_SCALE = 0.01695254415154625
SMITH2024_WASP77AB_TEMPLATE_R500K_FILENAME = (
    "w77_pre_nirspec_best_fit_R500K_scaled.txt"
)
SMITH2024_WASP77AB_TEMPLATE_R500K_SIZE = 72_672_825
SMITH2024_WASP77AB_TEMPLATE_R500K_MD5 = "93841125cab0a2c74ae4730ebf36b7dc"
SMITH2024_WASP77AB_TEMPLATE_R500K_SHA256 = (
    "fe1fb7cd37b6dec52173a19ee1fd7ea1529c421946c76d728d0bdcf4285ad34d"
)
SMITH2024_WASP77AB_TEMPLATE_FULL_FILENAME = "w77_1DRC_FULL.txt"
SMITH2024_WASP77AB_TEMPLATE_FULL_SIZE = 13_899_150
SMITH2024_WASP77AB_TEMPLATE_FULL_MD5 = "dbaa138b7768b727c87ff8f83fcc563f"
SMITH2024_WASP77AB_TEMPLATE_FULL_SHA256 = (
    "8f3fcdf119b42e1c1db5b5aba61c8266bad65cf3c279aa237e55346ed9be7a14"
)
SMITH2024_WASP77AB_TEMPLATE_H2O_FILENAME = "w77_1DRC_H2O_ONLY.txt"
SMITH2024_WASP77AB_TEMPLATE_H2O_SIZE = 13_899_150
SMITH2024_WASP77AB_TEMPLATE_H2O_MD5 = "6cf0ec37cbed5db5b18e6d434838fa40"
SMITH2024_WASP77AB_TEMPLATE_CO_FILENAME = "w77_1DRC_CO_ONLY.txt"
SMITH2024_WASP77AB_TEMPLATE_CO_SIZE = 13_899_150
SMITH2024_WASP77AB_TEMPLATE_CO_MD5 = "d4f0c6dfd4bb50a7ca9ca73ea3318eb3"
SMITH2024_WASP77AB_TEMPLATE_13CO_FILENAME = "w77_1DRC_13CO_ONLY.txt"
SMITH2024_WASP77AB_TEMPLATE_13CO_SIZE = 13_899_150
SMITH2024_WASP77AB_TEMPLATE_13CO_MD5 = "59ec25e625109995f8c702874b771f7c"
SMITH2024_WASP77AB_TEMPLATE_OTHER_GAS_FILENAME = "w77_1DRC_OTHER_GAS.txt"
SMITH2024_WASP77AB_TEMPLATE_OTHER_GAS_SIZE = 13_899_150
SMITH2024_WASP77AB_TEMPLATE_OTHER_GAS_MD5 = "b21937c15985f0fe69abf63ba6403350"
SMITH2024_WASP77AB_INFO_SIZE = 7_261
SMITH2024_WASP77AB_INFO_MD5 = "7c861b1baf8ae87801f95d7af8b2e9c6"
SMITH2024_WASP77AB_INFO_SHA256 = (
    "68f5a80f930e5f1f99cd1861678754f824bf665972d675f617a7d66c4fdad11e"
)

SMITH2024_WASP77AB_N_ORDERS = 44
SMITH2024_WASP77AB_N_FRAMES = 79
SMITH2024_WASP77AB_N_PIXELS = 1_848
SMITH2024_WASP77AB_CUBE_SHAPE = (
    SMITH2024_WASP77AB_N_ORDERS,
    SMITH2024_WASP77AB_N_FRAMES,
    SMITH2024_WASP77AB_N_PIXELS,
)

_INFO_COLUMNS = (
    "Time (BJD)",
    "Phase",
    "RV [km/s]",
    "Air Mass",
    "Humidity [% dew point]",
    "Med. SNR",
)
_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("_codecs", "encode"),
}

_TEMPLATE_PRODUCTS = MappingProxyType(
    {
        SMITH2024_WASP77AB_TEMPLATE_FULL_FILENAME: {
            "size": SMITH2024_WASP77AB_TEMPLATE_FULL_SIZE,
            "md5": SMITH2024_WASP77AB_TEMPLATE_FULL_MD5,
            "sha256": SMITH2024_WASP77AB_TEMPLATE_FULL_SHA256,
        },
        SMITH2024_WASP77AB_TEMPLATE_H2O_FILENAME: {
            "size": SMITH2024_WASP77AB_TEMPLATE_H2O_SIZE,
            "md5": SMITH2024_WASP77AB_TEMPLATE_H2O_MD5,
            "sha256": None,
        },
        SMITH2024_WASP77AB_TEMPLATE_CO_FILENAME: {
            "size": SMITH2024_WASP77AB_TEMPLATE_CO_SIZE,
            "md5": SMITH2024_WASP77AB_TEMPLATE_CO_MD5,
            "sha256": None,
        },
        SMITH2024_WASP77AB_TEMPLATE_13CO_FILENAME: {
            "size": SMITH2024_WASP77AB_TEMPLATE_13CO_SIZE,
            "md5": SMITH2024_WASP77AB_TEMPLATE_13CO_MD5,
            "sha256": None,
        },
        SMITH2024_WASP77AB_TEMPLATE_OTHER_GAS_FILENAME: {
            "size": SMITH2024_WASP77AB_TEMPLATE_OTHER_GAS_SIZE,
            "md5": SMITH2024_WASP77AB_TEMPLATE_OTHER_GAS_MD5,
            "sha256": None,
        },
        SMITH2024_WASP77AB_TEMPLATE_R500K_FILENAME: {
            "size": SMITH2024_WASP77AB_TEMPLATE_R500K_SIZE,
            "md5": SMITH2024_WASP77AB_TEMPLATE_R500K_MD5,
            "sha256": SMITH2024_WASP77AB_TEMPLATE_R500K_SHA256,
        },
    }
)


def _file_digests(path: Path) -> tuple[int, str, str]:
    """Return file size, MD5, and SHA256 using bounded memory."""

    digest_md5 = md5()
    digest_sha256 = sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                digest_md5.update(chunk)
                digest_sha256.update(chunk)
    except OSError as error:
        raise RobertDataError(f"cannot read Smith et al. input file: {path}") from error
    return size, digest_md5.hexdigest(), digest_sha256.hexdigest()


def _verify_file(
    path: Path,
    *,
    expected_size: int,
    expected_md5: str,
    expected_sha256: str,
    verify_checksum: bool,
    verify_sha256: bool,
    label: str,
) -> tuple[int, str, str]:
    """Verify an external file before any scientific parsing or loading."""

    if not path.is_file():
        raise FileNotFoundError(path)
    size, actual_md5, actual_sha256 = _file_digests(path)
    if verify_checksum and size != expected_size:
        raise RobertDataError(
            f"{label} size mismatch: expected {expected_size} bytes, got {size}"
        )
    if verify_checksum and actual_md5 != expected_md5:
        raise RobertDataError(
            f"{label} MD5 mismatch: expected {expected_md5}, got {actual_md5}"
        )
    if verify_sha256 and actual_sha256 != expected_sha256:
        raise RobertDataError(
            f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return size, actual_md5, actual_sha256


class _SmithRestrictedUnpickler(pickle.Unpickler):
    """Unpickle only the NumPy protocol-2 globals in the official cube."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                f"forbidden global in Smith et al. cube: {module}.{name}"
            )
        return super().find_class(module, name)


def _restricted_load(stream: BinaryIO) -> Any:
    """Load one official cube through the allow-list unpickler."""

    return _SmithRestrictedUnpickler(stream).load()


def _read_info_csv(path: Path) -> dict[str, np.ndarray]:
    """Read and validate the 79-row Smith frame metadata table."""

    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != _INFO_COLUMNS:
                raise RobertDataError(
                    "Smith 2024 frame metadata has unexpected CSV columns"
                )
            rows = list(reader)
    except UnicodeDecodeError as error:
        raise RobertDataError("Smith 2024 frame metadata is not UTF-8 CSV") from error
    except csv.Error as error:
        raise RobertDataError("Smith 2024 frame metadata is malformed CSV") from error

    if len(rows) != SMITH2024_WASP77AB_N_FRAMES:
        raise RobertDataError(
            "Smith 2024 frame metadata must contain "
            f"{SMITH2024_WASP77AB_N_FRAMES} rows"
        )

    arrays: dict[str, np.ndarray] = {}
    for column in _INFO_COLUMNS:
        try:
            values = np.asarray([float(row[column]) for row in rows], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise RobertDataError(
                f"Smith 2024 frame metadata column {column!r} is not numeric"
            ) from error
        if not np.all(np.isfinite(values)):
            raise RobertDataError(
                f"Smith 2024 frame metadata column {column!r} is not finite"
            )
        arrays[column] = values

    if not np.all(np.diff(arrays["Time (BJD)"]) > 0.0):
        raise RobertDataError("Smith 2024 frame times must be strictly increasing")
    if not np.all(np.diff(arrays["Phase"]) > 0.0):
        raise RobertDataError("Smith 2024 frame phases must be strictly increasing")
    return arrays


def _validate_cube_payload(payload: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate the exact list/array structure of ``cube14v3.pic``."""

    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        raise RobertDataError(
            "Smith 2024 cube must be a two-item list of wavelengths and flux"
        )
    wavelengths, flux = payload
    wavelengths_array = np.asarray(wavelengths, dtype=float)
    flux_array = np.asarray(flux, dtype=float)
    expected_wavelength_shape = (
        SMITH2024_WASP77AB_N_ORDERS,
        SMITH2024_WASP77AB_N_PIXELS,
    )
    if wavelengths_array.shape != expected_wavelength_shape:
        raise RobertDataError(
            "Smith 2024 order wavelengths must have shape "
            f"{expected_wavelength_shape}, got {wavelengths_array.shape}"
        )
    if flux_array.shape != SMITH2024_WASP77AB_CUBE_SHAPE:
        raise RobertDataError(
            "Smith 2024 flux cube must have shape "
            f"{SMITH2024_WASP77AB_CUBE_SHAPE}, got {flux_array.shape}"
        )
    if not np.all(np.isfinite(wavelengths_array)):
        raise RobertDataError("Smith 2024 order wavelengths contain non-finite values")
    if not np.all(np.isfinite(flux_array)):
        raise RobertDataError("Smith 2024 flux cube contains non-finite values")
    if np.any(wavelengths_array <= 0.0):
        raise RobertDataError("Smith 2024 order wavelengths must be positive")
    differences = np.diff(wavelengths_array, axis=1)
    if not np.all(np.all(differences > 0.0, axis=1)) and not np.all(
        np.all(differences < 0.0, axis=1)
    ):
        raise RobertDataError(
            "Smith 2024 order wavelength grids must each be strictly monotonic"
        )
    return wavelengths_array, flux_array


def load_smith2024_wasp77ab_template(
    path: str | Path,
    *,
    verify_checksum: bool = True,
    verify_sha256: bool = False,
    wavelength_unit: str = "micron",
    flux_ratio_scale: float = 1.0,
    name: str | None = None,
    metadata: Mapping[str, str] | None = None,
) -> HighResolutionEmissionTemplate:
    """Load one official Smith three-column emission template.

    The supported Zenodo text products contain wavelength, planet surface
    flux, and stellar surface flux columns.  The returned ratio is therefore
    a surface-flux ratio.  A caller that has an explicit radius ratio can
    apply it outside this loader; keeping the geometric factor explicit avoids
    silently changing the published template.

    Size and MD5 are checked before parsing.  SHA256 checking is available for
    the two products whose SHA256 values are recorded in this package.  For
    synthetic unit-test files, set ``verify_checksum=False`` explicitly.
    """

    template_path = Path(path).expanduser()
    product = _TEMPLATE_PRODUCTS.get(template_path.name)
    if product is None:
        if verify_checksum:
            raise RobertDataError(
                "unsupported Smith 2024 template filename; expected one of "
                + ", ".join(sorted(_TEMPLATE_PRODUCTS))
            )
        # An explicitly unverified path is useful for isolated tests and for
        # a caller who has already recorded a different upstream checksum.
        product = {"size": 0, "md5": "", "sha256": None}
    expected_sha256 = product["sha256"]
    if verify_sha256 and expected_sha256 is None:
        raise RobertDataError(
            f"no package SHA256 is recorded for Smith template {template_path.name}"
        )
    size, actual_md5, actual_sha256 = _verify_file(
        template_path,
        expected_size=int(product["size"]),
        expected_md5=str(product["md5"]),
        expected_sha256=str(expected_sha256 or ""),
        verify_checksum=verify_checksum,
        verify_sha256=verify_sha256,
        label=f"Smith 2024 template {template_path.name}",
    )
    try:
        values = np.loadtxt(template_path, dtype=float, ndmin=2)
    except (OSError, ValueError) as error:
        raise RobertDataError(
            f"Smith 2024 template {template_path.name} is not numeric three-column text"
        ) from error
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 2:
        raise RobertDataError(
            f"Smith 2024 template {template_path.name} must have shape (n, 3)"
        )
    if not np.all(np.isfinite(values)):
        raise RobertDataError(
            f"Smith 2024 template {template_path.name} contains non-finite values"
        )
    if np.any(values[:, 0] <= 0.0):
        raise RobertDataError(
            f"Smith 2024 template {template_path.name} wavelengths must be positive"
        )
    differences = np.diff(values[:, 0])
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise RobertDataError(
            f"Smith 2024 template {template_path.name} wavelengths must be strictly monotonic"
        )
    source_metadata: dict[str, str] = {
        "source": "Smith et al. 2024 supplementary data",
        "paper_doi": "10.3847/1538-3881/ad17bf",
        "paper_arxiv": "2312.13069",
        "zenodo_doi": SMITH2024_WASP77AB_ZENODO_DOI,
        "template_filename": template_path.name,
        "template_size_bytes": str(size),
        "template_md5": actual_md5,
        "template_sha256": actual_sha256,
        "template_columns": "wavelength, planet surface flux, stellar surface flux",
        "flux_ratio_type": "surface_flux_ratio",
        "flux_ratio_scale": str(float(flux_ratio_scale)),
        "abundance_convention": "ROBERT volume mixing ratios only",
    }
    if metadata is not None:
        source_metadata.update({str(key): str(value) for key, value in metadata.items()})
    return HighResolutionEmissionTemplate(
        wavelength=values[:, 0],
        planet_flux=values[:, 1],
        stellar_flux=values[:, 2],
        wavelength_unit=wavelength_unit,
        flux_ratio_scale=flux_ratio_scale,
        name=template_path.name if name is None else name,
        metadata=source_metadata,
    )


def load_smith2024_wasp77ab_hrs(
    path: str | Path,
    *,
    info_path: str | Path | None = None,
    verify_checksum: bool = True,
    verify_sha256: bool = False,
    mask: ArrayLike | None = None,
    order_mask: ArrayLike | None = None,
    frame_mask: ArrayLike | None = None,
    name: str = "Smith et al. 2024 WASP-77Ab pre-eclipse IGRINS",
    metadata: Mapping[str, str] | None = None,
) -> TimeResolvedHighResolutionObservation:
    """Load the Smith 2024 pre-eclipse IGRINS cube safely.

    ``path`` is the local ``cube14v3.pic`` file.  The companion CSV is found
    next to it unless ``info_path`` is supplied.  Size and MD5 verification is
    enabled by default and occurs before unpickling.  Set ``verify_sha256`` to
    also check the published local SHA256 values.  ``verify_checksum=False``
    is intended only for controlled unit tests with synthetic payloads.
    """

    cube_path = Path(path).expanduser()
    frame_info_path = (
        Path(info_path).expanduser()
        if info_path is not None
        else cube_path.with_name(SMITH2024_WASP77AB_INFO_FILENAME)
    )

    cube_size, cube_md5, cube_sha256 = _verify_file(
        cube_path,
        expected_size=SMITH2024_WASP77AB_CUBE14_SIZE,
        expected_md5=SMITH2024_WASP77AB_CUBE14_MD5,
        expected_sha256=SMITH2024_WASP77AB_CUBE14_SHA256,
        verify_checksum=verify_checksum,
        verify_sha256=verify_sha256,
        label="Smith 2024 cube14v3.pic",
    )
    info_size, info_md5, info_sha256 = _verify_file(
        frame_info_path,
        expected_size=SMITH2024_WASP77AB_INFO_SIZE,
        expected_md5=SMITH2024_WASP77AB_INFO_MD5,
        expected_sha256=SMITH2024_WASP77AB_INFO_SHA256,
        verify_checksum=verify_checksum,
        verify_sha256=verify_sha256,
        label="Smith 2024 20201214_info.csv",
    )

    # The first scientific deserialisation happens only after both files pass
    # their identity checks.
    try:
        with cube_path.open("rb") as stream:
            payload = _restricted_load(stream)
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError) as error:
        raise RobertDataError("could not safely load Smith 2024 cube14v3.pic") from error
    wavelengths, flux = _validate_cube_payload(payload)
    info = _read_info_csv(frame_info_path)

    source_metadata: dict[str, str] = {
        "source": "Smith et al. 2024 supplementary data",
        "paper_doi": "10.3847/1538-3881/ad17bf",
        "paper_arxiv": "2312.13069",
        "zenodo_doi": SMITH2024_WASP77AB_ZENODO_DOI,
        "zenodo_record": SMITH2024_WASP77AB_ZENODO_RECORD,
        "cube_filename": cube_path.name,
        "cube_size_bytes": str(cube_size),
        "cube_md5": cube_md5,
        "cube_sha256": cube_sha256,
        "info_filename": frame_info_path.name,
        "info_size_bytes": str(info_size),
        "info_md5": info_md5,
        "info_sha256": info_sha256,
        "cube_shape": str(SMITH2024_WASP77AB_CUBE_SHAPE),
        "data_role": "Smith 2024 IGRINS pre-eclipse HRS input",
        "velocity_column": "RV [km/s]",
        "velocity_sign_convention": "positive velocity is redshift",
        "abundance_convention": "ROBERT volume mixing ratios only",
    }
    if metadata is not None:
        source_metadata.update({str(key): str(value) for key, value in metadata.items()})

    return TimeResolvedHighResolutionObservation(
        order_wavelengths=wavelengths,
        flux=flux,
        phase=info["Phase"],
        fixed_velocity_km_s=info["RV [km/s]"],
        time_bjd=info["Time (BJD)"],
        order_mask=order_mask,
        frame_mask=frame_mask,
        mask=mask,
        airmass=info["Air Mass"],
        humidity_percent=info["Humidity [% dew point]"],
        median_snr=info["Med. SNR"],
        name=name,
        metadata=source_metadata,
    )


# Concise aliases for callers that do not need the ``_hrs`` suffix.
load_smith2024_wasp77ab = load_smith2024_wasp77ab_hrs
load_smith2024_wasp77ab_igrins = load_smith2024_wasp77ab_hrs
load_smith2024_igrins = load_smith2024_wasp77ab_hrs
load_smith2024_wasp77ab_emission_template = load_smith2024_wasp77ab_template

# Short constant aliases for scripts that use the paper's file names.
SMITH_CUBE14_SIZE = SMITH2024_WASP77AB_CUBE14_SIZE
SMITH_CUBE14_MD5 = SMITH2024_WASP77AB_CUBE14_MD5
SMITH_CUBE14_SHA256 = SMITH2024_WASP77AB_CUBE14_SHA256
SMITH_INFO14_SIZE = SMITH2024_WASP77AB_INFO_SIZE
SMITH_INFO14_MD5 = SMITH2024_WASP77AB_INFO_MD5
SMITH_INFO14_SHA256 = SMITH2024_WASP77AB_INFO_SHA256


__all__ = [
    "SMITH2024_WASP77AB_CUBE14_FILENAME",
    "SMITH2024_WASP77AB_CUBE14_MD5",
    "SMITH2024_WASP77AB_CUBE14_SHA256",
    "SMITH2024_WASP77AB_CUBE14_SIZE",
    "SMITH2024_WASP77AB_CUBE14_URL",
    "SMITH2024_WASP77AB_CUBE_SHAPE",
    "SMITH2024_WASP77AB_FLUX_RATIO_SCALE",
    "SMITH2024_WASP77AB_INFO_FILENAME",
    "SMITH2024_WASP77AB_INFO_MD5",
    "SMITH2024_WASP77AB_INFO_SHA256",
    "SMITH2024_WASP77AB_INFO_SIZE",
    "SMITH2024_WASP77AB_INFO_URL",
    "SMITH2024_WASP77AB_N_FRAMES",
    "SMITH2024_WASP77AB_N_ORDERS",
    "SMITH2024_WASP77AB_N_PIXELS",
    "SMITH2024_WASP77AB_ZENODO_DOI",
    "SMITH2024_WASP77AB_ZENODO_RECORD",
    "SMITH2024_WASP77AB_TEMPLATE_13CO_FILENAME",
    "SMITH2024_WASP77AB_TEMPLATE_13CO_MD5",
    "SMITH2024_WASP77AB_TEMPLATE_13CO_SIZE",
    "SMITH2024_WASP77AB_TEMPLATE_CO_FILENAME",
    "SMITH2024_WASP77AB_TEMPLATE_CO_MD5",
    "SMITH2024_WASP77AB_TEMPLATE_CO_SIZE",
    "SMITH2024_WASP77AB_TEMPLATE_FULL_FILENAME",
    "SMITH2024_WASP77AB_TEMPLATE_FULL_MD5",
    "SMITH2024_WASP77AB_TEMPLATE_FULL_SHA256",
    "SMITH2024_WASP77AB_TEMPLATE_FULL_SIZE",
    "SMITH2024_WASP77AB_TEMPLATE_H2O_FILENAME",
    "SMITH2024_WASP77AB_TEMPLATE_H2O_MD5",
    "SMITH2024_WASP77AB_TEMPLATE_H2O_SIZE",
    "SMITH2024_WASP77AB_TEMPLATE_OTHER_GAS_FILENAME",
    "SMITH2024_WASP77AB_TEMPLATE_OTHER_GAS_MD5",
    "SMITH2024_WASP77AB_TEMPLATE_OTHER_GAS_SIZE",
    "SMITH2024_WASP77AB_TEMPLATE_R500K_FILENAME",
    "SMITH2024_WASP77AB_TEMPLATE_R500K_MD5",
    "SMITH2024_WASP77AB_TEMPLATE_R500K_SHA256",
    "SMITH2024_WASP77AB_TEMPLATE_R500K_SIZE",
    "SMITH_CUBE14_MD5",
    "SMITH_CUBE14_SHA256",
    "SMITH_CUBE14_SIZE",
    "SMITH_INFO14_MD5",
    "SMITH_INFO14_SHA256",
    "SMITH_INFO14_SIZE",
    "load_smith2024_wasp77ab",
    "load_smith2024_wasp77ab_hrs",
    "load_smith2024_wasp77ab_igrins",
    "load_smith2024_igrins",
    "load_smith2024_wasp77ab_emission_template",
    "load_smith2024_wasp77ab_template",
]
