"""Opacity metadata, inspectors, and fixture interfaces."""

from .archive import (
    RobertOpacityArchive,
    inspect_robert_npy_directory,
    inspect_robert_npz_archive,
    load_robert_npy_directory,
    load_robert_npz_archive,
    write_robert_npy_directory,
    write_robert_npz_archive,
)
from .fixture import CoverageReport, EvaluatedOpacity, FixtureOpacityProvider, PreparedOpacity
from .inspectors import (
    file_sha256,
    inspect_exomol_directory,
    inspect_hitran_cia_file,
    inspect_hitran_par_file,
    inspect_kta_file,
)
from .metadata import (
    GridCoverage,
    OpacityCoverageReport,
    OpacityDatabase,
    OpacityDataProduct,
    OpacityDataSource,
    OpacityMode,
    OpacityStorageFormat,
    SpectralCoverage,
    pressure_values_in_unit,
    spectral_grid_values_in_unit,
)

__all__ = [
    "CoverageReport",
    "EvaluatedOpacity",
    "FixtureOpacityProvider",
    "GridCoverage",
    "OpacityCoverageReport",
    "OpacityDatabase",
    "OpacityDataProduct",
    "OpacityDataSource",
    "OpacityMode",
    "OpacityStorageFormat",
    "PreparedOpacity",
    "RobertOpacityArchive",
    "SpectralCoverage",
    "file_sha256",
    "inspect_exomol_directory",
    "inspect_hitran_cia_file",
    "inspect_hitran_par_file",
    "inspect_kta_file",
    "inspect_robert_npy_directory",
    "inspect_robert_npz_archive",
    "load_robert_npy_directory",
    "load_robert_npz_archive",
    "pressure_values_in_unit",
    "spectral_grid_values_in_unit",
    "write_robert_npy_directory",
    "write_robert_npz_archive",
]
