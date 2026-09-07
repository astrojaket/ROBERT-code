"""Opacity metadata, inspectors, and fixture interfaces."""

from typing import Union

from robert_exoplanets._data import (
    BUNDLED_K_TABLE_RESOLUTION,
    BUNDLED_K_TABLE_SPECIES,
    bundled_k_table_directory,
    bundled_k_table_paths,
    bundled_opacity_manifest,
)

from .archive import (
    RobertOpacityArchive,
    inspect_robert_npy_directory,
    inspect_robert_npz_archive,
    load_robert_npy_directory,
    load_robert_npz_archive,
    write_robert_npy_directory,
    write_robert_npz_archive,
)
from .correlated_k import (
    CorrelatedKCoverageReport,
    CorrelatedKOpacityProvider,
    CorrelatedKTable,
    EvaluatedCorrelatedKOpacity,
    PreparedCorrelatedKOpacity,
)
from .download import download_exomol_r1000
from .inspectors import (
    file_sha256,
    inspect_exomol_directory,
    inspect_hitran_cia_file,
    inspect_hitran_par_file,
    inspect_kta_file,
)
from .kta import (
    KtaHeader,
    KtaTable,
    convert_kta_to_robert_archive,
    kta_product_from_header,
    read_kta,
    read_kta_header,
)
from .line_by_line import (
    EvaluatedLineByLineMixture,
    EvaluatedLineByLineOpacity,
    LineByLineCoverageReport,
    LineByLineOpacityProvider,
    LineByLineTable,
    PreparedLineByLineOpacity,
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
from .opacity_sampling import (
    EvaluatedOpacitySampling,
    EvaluatedOpacitySamplingMixture,
    OpacitySamplingCoverageReport,
    OpacitySamplingProvider,
    OpacitySamplingTable,
    PreparedOpacitySampling,
)

OpacityProvider = Union[
    CorrelatedKOpacityProvider,
    LineByLineOpacityProvider,
    OpacitySamplingProvider,
]
PreparedOpacity = Union[
    PreparedCorrelatedKOpacity,
    PreparedLineByLineOpacity,
    PreparedOpacitySampling,
]
EvaluatedOpacity = Union[
    EvaluatedCorrelatedKOpacity,
    EvaluatedLineByLineOpacity,
    EvaluatedLineByLineMixture,
    EvaluatedOpacitySampling,
    EvaluatedOpacitySamplingMixture,
]

__all__ = [
    "BUNDLED_K_TABLE_RESOLUTION",
    "BUNDLED_K_TABLE_SPECIES",
    "CorrelatedKCoverageReport",
    "CorrelatedKOpacityProvider",
    "CorrelatedKTable",
    "EvaluatedCorrelatedKOpacity",
    "EvaluatedLineByLineMixture",
    "EvaluatedLineByLineOpacity",
    "EvaluatedOpacity",
    "EvaluatedOpacitySampling",
    "EvaluatedOpacitySamplingMixture",
    "GridCoverage",
    "KtaHeader",
    "KtaTable",
    "LineByLineCoverageReport",
    "LineByLineOpacityProvider",
    "LineByLineTable",
    "OpacityCoverageReport",
    "OpacityDatabase",
    "OpacityDataProduct",
    "OpacityDataSource",
    "OpacityMode",
    "OpacityProvider",
    "OpacitySamplingCoverageReport",
    "OpacitySamplingProvider",
    "OpacitySamplingTable",
    "OpacityStorageFormat",
    "PreparedCorrelatedKOpacity",
    "PreparedLineByLineOpacity",
    "PreparedOpacity",
    "PreparedOpacitySampling",
    "RobertOpacityArchive",
    "SpectralCoverage",
    "convert_kta_to_robert_archive",
    "bundled_k_table_directory",
    "bundled_k_table_paths",
    "bundled_opacity_manifest",
    "download_exomol_r1000",
    "file_sha256",
    "inspect_exomol_directory",
    "inspect_hitran_cia_file",
    "inspect_hitran_par_file",
    "inspect_kta_file",
    "inspect_robert_npy_directory",
    "inspect_robert_npz_archive",
    "load_robert_npy_directory",
    "load_robert_npz_archive",
    "kta_product_from_header",
    "pressure_values_in_unit",
    "read_kta",
    "read_kta_header",
    "spectral_grid_values_in_unit",
    "write_robert_npy_directory",
    "write_robert_npz_archive",
]
