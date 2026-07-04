"""Radiative-transfer-facing reference helpers."""

from .emission import (
    ClearSkyEmissionResult,
    disk_average_quadrature,
    solve_clear_sky_emission,
)
from .extinction import (
    LayerOpticalDepth,
    NemesisCiaTable,
    cia_optical_depth,
    rayleigh_scattering_optical_depth,
    read_nemesis_cia_table,
)
from .optical_depth import GasOpticalDepth, assemble_gas_optical_depth
from .geometry import (
    DiscGeometry,
    DiscPoint,
    gauss_legendre_disk_geometry,
    geometry_from_emission_angles,
    nemesis_lobatto_phase_geometry,
    normal_emission_geometry,
)
from .random_overlap import (
    random_overlap_species_tau,
    random_overlap_tau_vectors,
    rank_rebin_distribution,
)

__all__ = [
    "ClearSkyEmissionResult",
    "DiscGeometry",
    "DiscPoint",
    "GasOpticalDepth",
    "LayerOpticalDepth",
    "NemesisCiaTable",
    "assemble_gas_optical_depth",
    "cia_optical_depth",
    "disk_average_quadrature",
    "gauss_legendre_disk_geometry",
    "geometry_from_emission_angles",
    "nemesis_lobatto_phase_geometry",
    "normal_emission_geometry",
    "random_overlap_species_tau",
    "random_overlap_tau_vectors",
    "rank_rebin_distribution",
    "rayleigh_scattering_optical_depth",
    "read_nemesis_cia_table",
    "solve_clear_sky_emission",
]
