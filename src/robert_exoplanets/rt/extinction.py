"""Continuum and scattering-extinction optical-depth helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib.resources import as_file, files
from pathlib import Path
import struct
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import (
    PressureGrid,
    RobertCoverageError,
    RobertValidationError,
    SpectralGrid,
)
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.opacity import (
    pressure_values_in_unit,
    spectral_grid_values_in_unit,
)

from .optical_depth import GasOpticalDepth

try:  # pragma: no cover - exercised when the optional perf extra is installed.
    from numba import njit, prange
except Exception:  # pragma: no cover - dependency availability is environment-specific.
    njit = None
    prange = range

_NUMBA_AVAILABLE = njit is not None

BOLTZMANN_CONSTANT_J_K = 1.380649e-23
AMAGAT_MOLECULES_CM3 = 2.68675e19

DEFAULT_CIA_PAIR_ORDER = (
    "H2-H2_equilibrium",
    "H2-He_equilibrium",
    "H2-H2_normal",
    "H2-He_normal",
    "H2-N2",
    "N2-CH4",
    "N2-N2",
    "CH4-CH4",
    "H2-CH4",
)

# The H-minus fits below follow the implementation used by
# petitRADTRANS 3.3.3.  The bound-free polynomial is from Gray (2008),
# pp. 155--156.  Keep the coefficients in Angstrom in the source convention.
# The helpers below convert the Gray/pRT cm^2 result to m^2 before it is
# combined with ROBERT's SI molecular column density.
HMINUS_BOUND_FREE_COEFFICIENTS = (
    1.99654,
    -1.18267e-5,
    2.64243e-6,
    -4.40524e-10,
    3.23992e-14,
    -1.39568e-18,
    2.78701e-23,
)
HMINUS_BOUND_FREE_THRESHOLD_ANGSTROM = 1.64e4
HMINUS_FREE_FREE_MIN_ANGSTROM = 2600.0
HMINUS_FREE_FREE_MAX_ANGSTROM = 113900.0
HMINUS_FREE_FREE_MIN_TEMPERATURE_K = 2500.0


@dataclass(frozen=True)
class LayerOpticalDepth:
    """Named optical-depth contribution on layer and spectral axes."""

    name: str
    tau: ArrayLike
    spectral_grid: SpectralGrid
    pressure_grid: PressureGrid
    kind: str = "extinction"
    unit: str = "dimensionless"
    phase_function_moments: ArrayLike | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("layer optical-depth name must not be empty")
        if not self.kind:
            raise RobertValidationError("layer optical-depth kind must not be empty")
        if not self.unit:
            raise RobertValidationError("layer optical-depth unit must not be empty")
        tau = _readonly_array(
            self.tau,
            "tau",
            (self.pressure_grid.n_layers, self.spectral_grid.size),
        )
        if np.any(tau < 0.0):
            raise RobertValidationError("layer optical depth must be non-negative")
        phase_moments = None
        if self.phase_function_moments is not None:
            phase_moments = _readonly_phase_function_moments(
                self.phase_function_moments,
                (self.pressure_grid.n_layers, self.spectral_grid.size),
            )
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "phase_function_moments", phase_moments)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    def cumulative_tau_from_top(self) -> NDArray[np.float64]:
        """Return cumulative optical depth from the top of the atmosphere."""

        order = _top_to_bottom_order(self.pressure_grid)
        cumulative = np.cumsum(self.tau[order], axis=0)
        return _restore_layer_order(cumulative, order)


@dataclass(frozen=True)
class HMinusContinuumConfig:
    """Configuration for explicit H-minus continuum abundances.

    The three species are read from the atmospheric VMR composition.  This
    class does not calculate an equilibrium abundance and does not introduce
    an additional retrieval scale.  To retrieve these quantities, include
    ``H-``, ``H``, and ``e-`` in a :class:`~robert_exoplanets.atmosphere.FreeChemistry`
    model (or provide equivalent profiles from another chemistry model).

    The Gray (2008) free-free fit is evaluated only for temperatures at or
    above 2500 K.  ``temperature_extrapolation="zero"`` matches the current
    petitRADTRANS behaviour below that bound.  ``"raise"`` is available for
    strict coverage checks.  The free-free wavelength fit is valid from
    2600 to 113900 Angstrom (0.26 to 11.39 micron); the spectral policy
    controls requests outside this range.
    """

    hminus_species: str = "H-"
    hydrogen_species: str = "H"
    electron_species: str = "e-"
    temperature_extrapolation: str = "zero"
    spectral_extrapolation: str = "zero"
    name: str = "H-minus bound-free/free-free continuum"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        species = tuple(
            str(value).strip()
            for value in (
                self.hminus_species,
                self.hydrogen_species,
                self.electron_species,
            )
        )
        if any(not value for value in species):
            raise RobertValidationError(
                "H-minus continuum species names must not be empty"
            )
        if len(set(species)) != len(species):
            raise RobertValidationError(
                "H-minus continuum species names must be unique"
            )
        temperature_policy = str(self.temperature_extrapolation).strip().lower()
        if temperature_policy not in {"zero", "raise"}:
            raise RobertValidationError(
                "H-minus temperature_extrapolation must be 'zero' or 'raise'"
            )
        spectral_policy = str(self.spectral_extrapolation).strip().lower()
        if spectral_policy not in {"zero", "raise"}:
            raise RobertValidationError(
                "H-minus spectral_extrapolation must be 'zero' or 'raise'"
            )
        name = str(self.name).strip()
        if not name:
            raise RobertValidationError("H-minus continuum name must not be empty")
        object.__setattr__(self, "hminus_species", species[0])
        object.__setattr__(self, "hydrogen_species", species[1])
        object.__setattr__(self, "electron_species", species[2])
        object.__setattr__(self, "temperature_extrapolation", temperature_policy)
        object.__setattr__(self, "spectral_extrapolation", spectral_policy)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def species(self) -> tuple[str, str, str]:
        """Return H-minus, neutral-H, and electron composition keys."""

        return (self.hminus_species, self.hydrogen_species, self.electron_species)


@dataclass(frozen=True)
class CiaTable:
    """CIA coefficient table in ROBERT's pair, temperature, wavenumber order."""

    wavenumber_cm_inverse: ArrayLike
    temperature_K: ArrayLike
    k_cia: ArrayLike
    pair_order: tuple[str, ...] = DEFAULT_CIA_PAIR_ORDER
    unit: str = "cm^-1 amagat^-2"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wavenumber = _readonly_1d(self.wavenumber_cm_inverse, "wavenumber_cm_inverse")
        temperature = _readonly_1d(self.temperature_K, "temperature_K")
        k_cia = np.array(self.k_cia, dtype=float, copy=True)
        if np.any(wavenumber < 0.0):
            raise RobertValidationError("CIA wavenumber values must be non-negative")
        if wavenumber.size > 1 and not np.all(np.diff(wavenumber) > 0.0):
            raise RobertValidationError(
                "CIA wavenumber grid must be strictly increasing"
            )
        if np.any(temperature <= 0.0):
            raise RobertValidationError("CIA temperature values must be positive")
        if temperature.size > 1 and not np.all(np.diff(temperature) > 0.0):
            raise RobertValidationError(
                "CIA temperature grid must be strictly increasing"
            )
        if k_cia.shape != (len(self.pair_order), temperature.size, wavenumber.size):
            raise RobertValidationError(
                "k_cia must have shape pair x temperature x wavenumber"
            )
        if not np.all(np.isfinite(k_cia)) or np.any(k_cia < 0.0):
            raise RobertValidationError(
                "CIA coefficients must be finite and non-negative"
            )
        if not self.unit:
            raise RobertValidationError("CIA coefficient unit must not be empty")

        k_cia.setflags(write=False)
        object.__setattr__(self, "wavenumber_cm_inverse", wavenumber)
        object.__setattr__(self, "temperature_K", temperature)
        object.__setattr__(self, "k_cia", k_cia)
        object.__setattr__(self, "pair_order", tuple(self.pair_order))
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def n_pairs(self) -> int:
        """Number of CIA pairs stored in the table."""

        return len(self.pair_order)

    @classmethod
    def from_petitradtrans_hdf(
        cls,
        path: str | Path,
        *,
        collision_pair: str,
    ) -> "CiaTable":
        """Load one H2-H2 or H2-He petitRADTRANS CIA HDF5 table.

        The pRT ``alpha`` values are absorption coefficients at unit amagat
        density product. They are placed in both equilibrium/normal hydrogen
        slots because these pRT tables do not encode a separate spin isomer.
        """

        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RobertValidationError(
                "loading petitRADTRANS HDF5 CIA requires h5py"
            ) from exc

        normalized_pair = collision_pair.strip().upper().replace("_", "-")
        while "--" in normalized_pair:
            normalized_pair = normalized_pair.replace("--", "-")
        pair_indices = {"H2-H2": (0, 2), "H2-HE": (1, 3)}
        if normalized_pair not in pair_indices:
            raise RobertValidationError("collision_pair must be 'H2-H2' or 'H2-He'")
        source = Path(path).expanduser()
        required = ("t", "wavenumbers", "alpha", "mol_name", "mol_mass")
        try:
            with h5py.File(source, "r") as handle:
                missing = tuple(name for name in required if name not in handle)
                if missing:
                    raise RobertValidationError(
                        "petitRADTRANS CIA file is missing datasets: "
                        + ", ".join(missing)
                    )
                temperature = np.asarray(handle["t"], dtype=float)
                wavenumber = np.asarray(handle["wavenumbers"], dtype=float)
                alpha = np.asarray(handle["alpha"], dtype=float)
                declared_unit = str(handle["alpha"].attrs.get("units", "")).strip()
                molecules = "+".join(_decode_hdf_strings(handle["mol_name"]))
                molar_masses = ",".join(
                    f"{value:.17g}"
                    for value in np.asarray(handle["mol_mass"], dtype=float)
                )
                doi = _decode_hdf_first(handle.get("DOI"))
        except OSError as exc:
            raise RobertValidationError(
                f"could not read petitRADTRANS CIA file: {source}"
            ) from exc
        if declared_unit != "cm^-1":
            raise RobertValidationError(
                "petitRADTRANS CIA alpha dataset must declare units of cm^-1"
            )
        if alpha.shape != (temperature.size, wavenumber.size):
            raise RobertValidationError(
                "petitRADTRANS CIA alpha must have temperature x wavenumber shape"
            )
        k_cia = np.zeros(
            (len(DEFAULT_CIA_PAIR_ORDER), temperature.size, wavenumber.size),
            dtype=float,
        )
        for index in pair_indices[normalized_pair]:
            k_cia[index] = alpha
        return cls(
            wavenumber_cm_inverse=wavenumber,
            temperature_K=temperature,
            k_cia=k_cia,
            pair_order=DEFAULT_CIA_PAIR_ORDER,
            unit="cm^-1 amagat^-2",
            metadata={
                "source_format": "petitradtrans_cia_hdf5",
                "source_path": str(source.resolve()),
                "source_alpha_unit": declared_unit,
                "collision_pair": collision_pair,
                "molecules": molecules,
                "molar_masses_amu": molar_masses,
                "doi": doi,
                "hydrogen_spin_state": "not_separated_in_source",
            },
        )


def read_cia_table(
    path: str | Path,
    *,
    dnu: float = 10.0,
    n_pairs: int = 9,
    endian: str | None = None,
) -> CiaTable:
    """Read an unformatted binary CIA table.

    The file format stores one float64 temperature record followed by one
    float32 coefficient record. Coefficients are stored by wavenumber,
    temperature, then CIA pair.
    """

    table_path = Path(path).expanduser()
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    if dnu <= 0.0 or not np.isfinite(dnu):
        raise RobertValidationError(
            "CIA wavenumber spacing dnu must be finite and positive"
        )
    if n_pairs < 1:
        raise RobertValidationError("CIA table must contain at least one pair")

    endians = (endian,) if endian is not None else ("<", ">")
    last_error: Exception | None = None
    for byte_order in endians:
        if byte_order not in {"<", ">"}:
            raise RobertValidationError("CIA table endian must be '<', '>', or None")
        try:
            return _read_cia_table_with_endian(
                table_path,
                dnu=float(dnu),
                n_pairs=int(n_pairs),
                endian=byte_order,
            )
        except RobertValidationError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RobertValidationError("could not read CIA table")


def load_nemesispy_cia_table() -> CiaTable:
    """Load ROBERT's vendored NemesisPy v1.0.1 CIA reference table."""

    resource = files("robert_exoplanets").joinpath(
        "data/cia/exocia_hitran12_200-3800K.tab"
    )
    with as_file(resource) as path:
        table = read_cia_table(path)
    return replace(
        table,
        metadata={
            **dict(table.metadata),
            "source_project": "NEMESISPY",
            "source_tag": "v1.0.1",
            "source_commit": "a883805fcd402eab341308f39715670b0ae74cb8",
            "checksum_sha256": "5b519f02f98b205f20628ee5ec7f2829528d0bd356b449c4221ba8b2ef86ea0e",
            "license": "BSD-3-Clause",
        },
    )


def cia_optical_depth(
    gas_optical_depth: GasOpticalDepth,
    cia_table: CiaTable,
    *,
    normal_hydrogen: bool = True,
    temperature_extrapolation: str = "raise",
    spectral_extrapolation: str = "raise",
    coefficient_interpolation: str = "linear",
    name: str = "H2-H2/H2-He CIA",
) -> LayerOpticalDepth:
    """Compute CIA optical depth using a uniform-layer path estimate.

    ROBERT does not yet carry a full height grid here, so this helper derives
    the path length from the hydrostatic column density and ideal-gas number
    density at the layer pressure and temperature. The approximation is
    recorded in metadata.
    """

    if temperature_extrapolation not in {"raise", "clip"}:
        raise RobertValidationError(
            "temperature_extrapolation must be 'raise' or 'clip'"
        )
    if spectral_extrapolation not in {"raise", "zero"}:
        raise RobertValidationError("spectral_extrapolation must be 'raise' or 'zero'")
    if coefficient_interpolation not in {"linear", "log"}:
        raise RobertValidationError(
            "coefficient_interpolation must be 'linear' or 'log'"
        )

    atmosphere = gas_optical_depth.atmosphere
    wavenumber = spectral_grid_values_in_unit(gas_optical_depth.spectral_grid, "cm^-1")
    pressure_pa = pressure_values_in_unit(
        atmosphere.pressure_grid.centers,
        atmosphere.pressure_grid.unit,
        "pa",
    )
    temperature = atmosphere.temperature
    number_density_m3 = pressure_pa / (BOLTZMANN_CONSTANT_J_K * temperature)
    path_length_m = (
        gas_optical_depth.layer_column_density_molecules_m2 / number_density_m3
    )
    xlen_cm = path_length_m * 100.0
    total_amount_cm2 = gas_optical_depth.layer_column_density_molecules_m2 * 1.0e-4
    amagat_density = total_amount_cm2 / xlen_cm / AMAGAT_MOLECULES_CM3
    tau_path = xlen_cm * amagat_density**2

    if not np.all(np.isfinite(tau_path)) or np.any(tau_path < 0.0):
        raise RobertValidationError("CIA path calculation produced invalid values")

    h2 = _composition_profile(atmosphere.composition, "H2", atmosphere.n_layers)
    he = _composition_profile(atmosphere.composition, "He", atmosphere.n_layers)
    n2 = _composition_profile(atmosphere.composition, "N2", atmosphere.n_layers)
    ch4 = _composition_profile(atmosphere.composition, "CH4", atmosphere.n_layers)

    h2_h2_index = 2 if normal_hydrogen else 0
    h2_he_index = 3 if normal_hydrogen else 1
    pair_terms = (
        (h2_h2_index, h2 * h2),
        (h2_he_index, h2 * he),
        (4, h2 * n2),
        (5, n2 * ch4),
        (6, n2 * n2),
        (7, ch4 * ch4),
        (8, h2 * ch4),
    )

    mixing_factors = np.zeros((cia_table.n_pairs, atmosphere.n_layers), dtype=float)
    active_pairs: list[str] = []
    for pair_index, mixing_factor in pair_terms:
        if pair_index >= cia_table.n_pairs:
            if np.any(mixing_factor > 0.0):
                raise RobertValidationError(
                    "CIA table does not contain a required active pair"
                )
            continue
        mixing_factors[pair_index] += mixing_factor
        if np.any(mixing_factor > 0.0):
            active_pairs.append(cia_table.pair_order[pair_index])

    if _NUMBA_AVAILABLE:
        tau = _numba_cia_optical_depth(
            cia_table,
            temperature,
            wavenumber,
            mixing_factors,
            tau_path,
            temperature_extrapolation=temperature_extrapolation,
            spectral_extrapolation=spectral_extrapolation,
            coefficient_interpolation=coefficient_interpolation,
        )
    else:
        tau = _numpy_cia_optical_depth(
            cia_table,
            temperature,
            wavenumber,
            mixing_factors,
            tau_path,
            temperature_extrapolation=temperature_extrapolation,
            spectral_extrapolation=spectral_extrapolation,
            coefficient_interpolation=coefficient_interpolation,
        )

    if not np.all(np.isfinite(tau)) or np.any(tau < 0.0):
        raise RobertValidationError(
            "CIA optical-depth calculation produced invalid values"
        )
    return LayerOpticalDepth(
        name=name,
        tau=tau,
        spectral_grid=gas_optical_depth.spectral_grid,
        pressure_grid=gas_optical_depth.pressure_grid,
        kind="absorption_continuum",
        metadata={
            "source_format": str(cia_table.metadata.get("source_format", "cia_binary")),
            "source_path": str(cia_table.metadata.get("source_path", "")),
            "path_model": "uniform_layer_ideal_gas_from_hydrostatic_column",
            "hydrogen_spin_state": "normal" if normal_hydrogen else "equilibrium",
            "temperature_extrapolation": temperature_extrapolation,
            "spectral_extrapolation": spectral_extrapolation,
            "coefficient_interpolation": coefficient_interpolation,
            "active_pairs": ",".join(sorted(set(active_pairs))),
        },
    )


def hminus_optical_depth(
    gas_optical_depth: GasOpticalDepth,
    config: HMinusContinuumConfig | None = None,
) -> LayerOpticalDepth:
    """Compute H-minus bound-free and free-free optical depth.

    The calculation uses explicit number fractions from the atmosphere
    composition.  It never derives H-minus or electron abundances from a
    Saha or other equilibrium closure.  For each layer, the bound-free term
    is ``sigma_bf * N(H-)``.  The free-free term is
    ``sigma_ff(lambda, T, p_e) * N(H)``, where ``p_e = P * x_e`` and the
    Gray fit receives the electron partial pressure in dyn cm^-2.

    The bound-free cross-section is the Gray (2008) polynomial used by
    petitRADTRANS.  The source polynomial is evaluated in its cm^2
    convention, then returned by the helper in m^2 per H-minus particle.  It
    is zero above the 16400 Angstrom threshold.  The free-free fit is the
    corresponding Gray polynomial used by petitRADTRANS and is valid for
    2600--113900 Angstrom and temperatures at least 2500 K.  The returned
    optical depth is dimensionless on the gas layer and spectral axes, and
    works for correlated-k, opacity-sampling, and line-by-line gas optical
    depths because it uses only their physical column density.

    ``config`` defaults to the petitRADTRANS composition keys ``H-``, ``H``,
    and ``e-``.  All three keys must be present when this contribution is
    enabled.  Missing keys are an error rather than an implicit zero or an
    equilibrium assumption.
    """

    continuum = HMinusContinuumConfig() if config is None else config
    if not isinstance(continuum, HMinusContinuumConfig):
        raise RobertValidationError(
            "hminus config must be an HMinusContinuumConfig"
        )

    gas = gas_optical_depth
    atmosphere = gas.atmosphere
    n_layers = atmosphere.n_layers
    hminus = _required_vmr_profile(
        atmosphere.composition,
        continuum.hminus_species,
        n_layers,
        context="H-minus continuum",
    )
    hydrogen = _required_vmr_profile(
        atmosphere.composition,
        continuum.hydrogen_species,
        n_layers,
        context="H-minus continuum",
    )
    electron = _required_vmr_profile(
        atmosphere.composition,
        continuum.electron_species,
        n_layers,
        context="H-minus continuum",
    )
    layer_column = np.asarray(
        gas.layer_column_density_molecules_m2,
        dtype=float,
    )
    if layer_column.shape != (n_layers,) or not np.all(np.isfinite(layer_column)):
        raise RobertValidationError(
            "H-minus continuum requires finite layer molecular columns"
        )
    if np.any(layer_column <= 0.0):
        raise RobertValidationError(
            "H-minus continuum requires positive layer molecular columns"
        )

    wavelength_micron = spectral_grid_values_in_unit(
        gas.spectral_grid,
        "micron",
    )
    wavelength_angstrom = wavelength_micron * 1.0e4
    if continuum.spectral_extrapolation == "raise":
        outside = (wavelength_angstrom < HMINUS_FREE_FREE_MIN_ANGSTROM) | (
            wavelength_angstrom > HMINUS_FREE_FREE_MAX_ANGSTROM
        )
        if np.any(outside):
            raise RobertCoverageError(
                "requested spectrum is outside the H-minus free-free wavelength "
                "range 2600--113900 Angstrom"
            )
    temperature = np.asarray(atmosphere.temperature, dtype=float)
    if continuum.temperature_extrapolation == "raise" and np.any(
        temperature < HMINUS_FREE_FREE_MIN_TEMPERATURE_K
    ):
        raise RobertCoverageError(
            "atmosphere temperature is below the H-minus free-free fit range "
            "of 2500 K"
        )

    if gas.spectral_grid.bin_edges is None:
        bound_free = _hminus_bound_free_point_cross_section(
            wavelength_angstrom
        )
        bound_free_sampling = "point_evaluation_without_bin_edges"
    else:
        edge_grid = SpectralGrid.from_array(
            gas.spectral_grid.bin_edges,
            unit=gas.spectral_grid.unit,
            role="hminus-bin-edges",
        )
        edge_wavelength_angstrom = (
            spectral_grid_values_in_unit(edge_grid, "micron") * 1.0e4
        )
        bound_free = _hminus_bound_free_bin_cross_section(
            edge_wavelength_angstrom
        )
        bound_free_sampling = "gray_polynomial_bin_average"

    pressure_pa = pressure_values_in_unit(
        atmosphere.pressure_grid.centers,
        atmosphere.pressure_grid.unit,
        "pa",
    )
    tau_bound_free = layer_column[:, None] * hminus[:, None] * bound_free[None, :]
    tau_free_free = np.zeros_like(tau_bound_free)
    for layer_index, layer_temperature in enumerate(temperature):
        electron_partial_pressure_dyn_cm2 = (
            pressure_pa[layer_index] * 10.0 * electron[layer_index]
        )
        cross_section_cm2 = _hminus_free_free_cross_section(
            wavelength_angstrom,
            float(layer_temperature),
            float(electron_partial_pressure_dyn_cm2),
        )
        # The fit returns cm^2 per H particle after multiplication by p_e.
        # Convert cm^2 to m^2 before multiplying the SI molecular column.
        tau_free_free[layer_index] = (
            layer_column[layer_index]
            * hydrogen[layer_index]
            * cross_section_cm2
            * 1.0e-4
        )

    tau = tau_bound_free + tau_free_free
    if not np.all(np.isfinite(tau)) or np.any(tau < 0.0):
        raise RobertValidationError(
            "H-minus optical-depth calculation produced invalid values"
        )
    return LayerOpticalDepth(
        name=continuum.name,
        tau=tau,
        spectral_grid=gas.spectral_grid,
        pressure_grid=gas.pressure_grid,
        kind="absorption_continuum",
        metadata={
            "source": "Gray (2008), pp. 155-156, as implemented by petitRADTRANS",
            "source_convention": "petitRADTRANS 3.3.3 H-minus bound/free-free fits",
            "hminus_species": continuum.hminus_species,
            "hydrogen_species": continuum.hydrogen_species,
            "electron_species": continuum.electron_species,
            "composition_source": "explicit_atmosphere_vmr",
            "equilibrium_closure": "none",
            "bound_free_cross_section_unit": "m^2 per H- particle",
            "bound_free_threshold_angstrom": f"{HMINUS_BOUND_FREE_THRESHOLD_ANGSTROM:g}",
            "bound_free_sampling": bound_free_sampling,
            "free_free_cross_section_unit": (
                "m^2 per H particle after electron-pressure multiplication"
            ),
            "free_free_electron_pressure_unit": "dyn cm^-2",
            "free_free_wavelength_range_angstrom": (
                f"{HMINUS_FREE_FREE_MIN_ANGSTROM:g}-"
                f"{HMINUS_FREE_FREE_MAX_ANGSTROM:g}"
            ),
            "free_free_temperature_min_K": f"{HMINUS_FREE_FREE_MIN_TEMPERATURE_K:g}",
            "temperature_extrapolation": continuum.temperature_extrapolation,
            "spectral_extrapolation": continuum.spectral_extrapolation,
            **dict(continuum.metadata),
        },
    )


def _hminus_bound_free_bin_cross_section(
    wavelengths_bin_edges_angstrom: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return Gray H-minus bound-free cross-section bin averages in m^2."""

    edges = np.asarray(wavelengths_bin_edges_angstrom, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise RobertValidationError(
            "H-minus wavelength bin edges must contain at least two values"
        )
    if not np.all(np.isfinite(edges)) or np.any(edges <= 0.0):
        raise RobertValidationError(
            "H-minus wavelength bin edges must be finite and positive"
        )
    differences = np.diff(edges)
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise RobertValidationError(
            "H-minus wavelength bin edges must be strictly monotonic"
        )
    if np.all(differences < 0.0):
        # Evaluate in the canonical ascending orientation and restore the
        # caller's order.  The bin average is orientation independent, while
        # directly applying the source masks to descending edges can mishandle
        # a bin that straddles the 16400 Angstrom threshold.
        return _hminus_bound_free_bin_cross_section(edges[::-1])[::-1]

    left = edges[:-1]
    right = edges[1:]
    threshold = HMINUS_BOUND_FREE_THRESHOLD_ANGSTROM
    # This is the source algorithm for ascending wavelength edges.  It keeps
    # the pRT midpoint treatment for bins that straddle the threshold.
    integral = np.zeros_like(right)
    below_threshold = right <= threshold
    for index, coefficient in enumerate(HMINUS_BOUND_FREE_COEFFICIENTS):
        power = index + 1
        integral[below_threshold] += coefficient * (
            right[below_threshold] ** power - left[below_threshold] ** power
        ) / power
    bracketed = (left < threshold) & (right > threshold)
    for index, coefficient in enumerate(HMINUS_BOUND_FREE_COEFFICIENTS):
        power = index + 1
        integral[bracketed] += coefficient * (
            threshold**power - left[bracketed] ** power
        ) / power
    cross_section = integral
    cross_section[0.5 * (left + right) > threshold] = 0.0
    cross_section[cross_section < 0.0] = 0.0
    # The Gray polynomial has a 1e-18 cm^2 scale.  Convert cm^2 to m^2 before
    # multiplying by ROBERT's SI molecular column density.
    return cross_section * 1.0e-18 * 1.0e-4 / differences


def _hminus_bound_free_point_cross_section(
    wavelengths_angstrom: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return Gray H-minus bound-free cross-sections at point samples in m^2."""

    wavelengths = np.asarray(wavelengths_angstrom, dtype=float)
    if wavelengths.ndim != 1 or not np.all(np.isfinite(wavelengths)):
        raise RobertValidationError(
            "H-minus wavelengths must be a finite one-dimensional array"
        )
    if np.any(wavelengths <= 0.0):
        raise RobertValidationError("H-minus wavelengths must be positive")
    cross_section = np.zeros_like(wavelengths)
    active = wavelengths <= HMINUS_BOUND_FREE_THRESHOLD_ANGSTROM
    for index, coefficient in enumerate(HMINUS_BOUND_FREE_COEFFICIENTS):
        # The integrated pRT expression contains lambda**(i+1)/(i+1), so its
        # point polynomial is a_i * lambda**i.
        cross_section[active] += coefficient * wavelengths[active] ** index
    # The Gray polynomial has a 1e-18 cm^2 scale; return SI m^2.
    cross_section *= 1.0e-18 * 1.0e-4
    cross_section[cross_section < 0.0] = 0.0
    return cross_section


def _hminus_free_free_cross_section(
    wavelengths_angstrom: NDArray[np.float64],
    temperature_K: float,
    electron_partial_pressure_dyn_cm2: float,
) -> NDArray[np.float64]:
    """Return Gray H-minus free-free cross-section in cm^2 per H particle."""

    wavelengths = np.asarray(wavelengths_angstrom, dtype=float)
    cross_section = np.zeros_like(wavelengths)
    if temperature_K < HMINUS_FREE_FREE_MIN_TEMPERATURE_K:
        return cross_section
    active = (wavelengths >= HMINUS_FREE_FREE_MIN_ANGSTROM) & (
        wavelengths <= HMINUS_FREE_FREE_MAX_ANGSTROM
    )
    if not np.any(active):
        return cross_section
    log_wavelength = np.log10(wavelengths[active])
    theta = 5040.0 / temperature_K
    log_theta = np.log10(theta)
    f0 = (
        -2.2763
        - 1.6850 * log_wavelength
        + 0.76661 * log_wavelength**2
        - 0.053346 * log_wavelength**3
    )
    f1 = (
        15.2827
        - 9.2846 * log_wavelength
        + 1.99381 * log_wavelength**2
        - 0.142631 * log_wavelength**3
    )
    f2 = (
        -197.789
        + 190.266 * log_wavelength
        - 67.9775 * log_wavelength**2
        + 10.6913 * log_wavelength**3
        - 0.625151 * log_wavelength**4
    )
    cross_section[active] = 1.0e-26 * electron_partial_pressure_dyn_cm2 * 10.0 ** (
        f0 + f1 * log_theta + f2 * log_theta**2
    )
    return cross_section


def rayleigh_scattering_optical_depth(
    gas_optical_depth: GasOpticalDepth,
    *,
    default_h2_fraction_of_h2_he: float | None = None,
    name: str = "H2/He Rayleigh scattering",
) -> LayerOpticalDepth:
    """Compute H2/He Rayleigh scattering extinction optical depth.

    The calculation applies separate H2 and He refractivity models and adds
    their number-weighted molecular cross-sections.  Cross-sections, rather
    than refractivities, are additive for an ideal gas mixture. An atmosphere
    without explicit H2 and He raises unless the caller deliberately supplies
    a fallback H2 fraction for an assumed H2/He background.
    """

    default_fraction = None
    if default_h2_fraction_of_h2_he is not None:
        default_fraction = float(default_h2_fraction_of_h2_he)
        if not np.isfinite(default_fraction) or not 0.0 <= default_fraction <= 1.0:
            raise RobertValidationError(
                "default_h2_fraction_of_h2_he must be in [0, 1]"
            )

    atmosphere = gas_optical_depth.atmosphere
    wavelength_micron = spectral_grid_values_in_unit(
        gas_optical_depth.spectral_grid, "micron"
    )
    h2 = _composition_profile(atmosphere.composition, "H2", atmosphere.n_layers)
    he = _composition_profile(atmosphere.composition, "He", atmosphere.n_layers)
    h2_he_fraction = h2 + he
    composition_fallback = "none"
    if not np.any(h2_he_fraction > 0.0):
        opacity_free_species = tuple(
            name.strip()
            for name in atmosphere.metadata.get("opacity_free_species", "").split(",")
            if name.strip()
        )
        opacity_free_background = any(
            name in atmosphere.composition
            and np.any(np.asarray(atmosphere.composition[name]) > 0.0)
            for name in opacity_free_species
        )
        if default_fraction is None and not opacity_free_background:
            raise RobertValidationError(
                "Rayleigh scattering requires explicit H2/He composition or an explicit "
                "default_h2_fraction_of_h2_he"
            )
        if default_fraction is not None:
            h2 = np.full(atmosphere.n_layers, default_fraction, dtype=float)
            he = 1.0 - h2
            composition_fallback = "explicit_H2/He"
        else:
            composition_fallback = "zero_H2/He_with_opacity_free_background"

    lambda_m = wavelength_micron * 1.0e-6
    inverse_micron = 1.0 / wavelength_micron
    ah2 = 13.58e-5
    bh2 = 7.52e-3
    ahe = 3.48e-5
    bhe = 2.30e-3
    refractivity_h2 = ah2 * (1.0 + bh2 * inverse_micron**2)
    refractivity_he = ahe * (1.0 + bhe * inverse_micron**2)
    n0 = 1.01325e5 / (BOLTZMANN_CONSTANT_J_K * 273.15)
    faniso = 1.0

    common_factor = (
        32.0 * np.pi**3 * faniso / (3.0 * (n0 * lambda_m[None, :] ** 2) ** 2)
    )
    cross_section_h2_m2 = common_factor * refractivity_h2[None, :] ** 2
    cross_section_he_m2 = common_factor * refractivity_he[None, :] ** 2
    tau = gas_optical_depth.layer_column_density_molecules_m2[:, None] * (
        h2[:, None] * cross_section_h2_m2 + he[:, None] * cross_section_he_m2
    )
    if not np.all(np.isfinite(tau)) or np.any(tau < 0.0):
        raise RobertValidationError(
            "Rayleigh optical-depth calculation produced invalid values"
        )

    return LayerOpticalDepth(
        name=name,
        tau=tau,
        spectral_grid=gas_optical_depth.spectral_grid,
        pressure_grid=gas_optical_depth.pressure_grid,
        kind="scattering_extinction",
        phase_function_moments=np.repeat(
            np.array([1.0, 0.0, 0.5, 0.0, 0.0])[:, None],
            gas_optical_depth.spectral_grid.size,
            axis=1,
        ),
        metadata={
            "source": "H2/He refractivity model",
            "column_model": "species VMR times hydrostatic layer column",
            "mixture_rule": "number-weighted sum of molecular cross-sections",
            "composition_fallback": composition_fallback,
            "scattering_source_function": "not_included",
        },
    )


def _read_cia_table_with_endian(
    path: Path,
    *,
    dnu: float,
    n_pairs: int,
    endian: str,
) -> CiaTable:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        temperature_record = _read_fortran_record(
            handle, endian=endian, file_size=file_size
        )
        coefficients_record = _read_fortran_record(
            handle, endian=endian, file_size=file_size
        )

    if len(temperature_record) % 8 != 0:
        raise RobertValidationError("CIA temperature record is not float64-aligned")
    temperature = np.frombuffer(
        temperature_record, dtype=np.dtype(f"{endian}f8")
    ).astype(float)
    coefficients = np.frombuffer(
        coefficients_record, dtype=np.dtype(f"{endian}f4")
    ).astype(float)
    denominator = n_pairs * temperature.size
    if denominator <= 0 or coefficients.size % denominator != 0:
        raise RobertValidationError(
            "CIA coefficient record has inconsistent dimensions"
        )
    n_wavenumber = coefficients.size // denominator
    wavenumber = np.linspace(0.0, dnu * (n_wavenumber - 1), n_wavenumber)
    k_cia = coefficients.reshape(n_wavenumber, temperature.size, n_pairs).transpose(
        2, 1, 0
    )
    pair_order = DEFAULT_CIA_PAIR_ORDER[:n_pairs]
    if len(pair_order) != n_pairs:
        pair_order = tuple(f"pair_{index}" for index in range(n_pairs))
    return CiaTable(
        wavenumber_cm_inverse=wavenumber,
        temperature_K=temperature,
        k_cia=k_cia,
        pair_order=pair_order,
        metadata={
            "source_format": "fortran_unformatted_cia",
            "source_path": str(path),
            "dnu_cm^-1": f"{dnu:g}",
            "endian": endian,
        },
    )


def _read_fortran_record(handle, *, endian: str, file_size: int) -> bytes:
    prefix = handle.read(4)
    if len(prefix) != 4:
        raise RobertValidationError(
            "unexpected end of CIA table while reading record marker"
        )
    (record_size,) = struct.unpack(f"{endian}i", prefix)
    if record_size <= 0 or record_size > file_size - 8:
        raise RobertValidationError("invalid CIA Fortran record size")
    record = handle.read(record_size)
    suffix = handle.read(4)
    if len(record) != record_size or len(suffix) != 4:
        raise RobertValidationError("unexpected end of CIA table while reading record")
    (trailer_size,) = struct.unpack(f"{endian}i", suffix)
    if trailer_size != record_size:
        raise RobertValidationError("CIA Fortran record markers do not match")
    return record


def _numba_cia_optical_depth(
    table: CiaTable,
    temperature_k: NDArray[np.float64],
    wavenumber_cm_inverse: NDArray[np.float64],
    mixing_factors: NDArray[np.float64],
    tau_path: NDArray[np.float64],
    *,
    temperature_extrapolation: str,
    spectral_extrapolation: str,
    coefficient_interpolation: str,
) -> NDArray[np.float64]:
    (
        temperature_lower,
        temperature_upper,
        temperature_fraction,
    ) = _interpolation_coordinates(
        temperature_k,
        table.temperature_K,
        extrapolation=temperature_extrapolation,
        quantity="atmosphere temperature",
    )
    (
        spectral_lower,
        spectral_upper,
        spectral_fraction,
    ) = _interpolation_coordinates(
        wavenumber_cm_inverse,
        table.wavenumber_cm_inverse,
        extrapolation=spectral_extrapolation,
        quantity="requested spectrum",
    )
    spectral_inside = (wavenumber_cm_inverse >= table.wavenumber_cm_inverse[0]) & (
        wavenumber_cm_inverse <= table.wavenumber_cm_inverse[-1]
    )
    return _numba_cia_optical_depth_kernel(
        table.k_cia,
        temperature_lower,
        temperature_upper,
        temperature_fraction,
        spectral_lower,
        spectral_upper,
        spectral_fraction,
        spectral_inside,
        mixing_factors,
        tau_path,
        coefficient_interpolation == "log",
    )


def _numpy_cia_optical_depth(
    table: CiaTable,
    temperature_k: NDArray[np.float64],
    wavenumber_cm_inverse: NDArray[np.float64],
    mixing_factors: NDArray[np.float64],
    tau_path: NDArray[np.float64],
    *,
    temperature_extrapolation: str,
    spectral_extrapolation: str,
    coefficient_interpolation: str,
) -> NDArray[np.float64]:
    tau = np.zeros((temperature_k.size, wavenumber_cm_inverse.size), dtype=float)
    for layer_index, layer_temperature in enumerate(temperature_k):
        coefficients = _interpolate_cia_coefficients(
            table,
            layer_temperature,
            wavenumber_cm_inverse,
            temperature_extrapolation=temperature_extrapolation,
            spectral_extrapolation=spectral_extrapolation,
            coefficient_interpolation=coefficient_interpolation,
        )
        tau[layer_index] = (
            np.einsum(
                "pw,p->w",
                coefficients,
                mixing_factors[:, layer_index],
                optimize=True,
            )
            * tau_path[layer_index]
        )
    return tau


def _interpolation_coordinates(
    values: NDArray[np.float64],
    grid: NDArray[np.float64],
    *,
    extrapolation: str,
    quantity: str,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    below = values < grid[0]
    above = values > grid[-1]
    if np.any(below | above) and extrapolation == "raise":
        if quantity == "atmosphere temperature":
            raise RobertCoverageError(
                "atmosphere temperature is outside the CIA table temperature grid"
            )
        raise RobertCoverageError(
            "requested spectrum is outside the CIA table wavenumber grid"
        )

    clipped = np.clip(values, grid[0], grid[-1])
    if grid.size == 1:
        indices = np.zeros(values.size, dtype=np.int64)
        return indices, indices.copy(), np.zeros(values.size, dtype=float)

    upper = np.searchsorted(grid, clipped, side="left")
    upper = np.clip(upper, 1, grid.size - 1).astype(np.int64)
    lower = upper - 1
    fraction = (clipped - grid[lower]) / (grid[upper] - grid[lower])
    at_lower_boundary = clipped <= grid[0]
    lower[at_lower_boundary] = 0
    upper[at_lower_boundary] = 0
    fraction[at_lower_boundary] = 0.0
    return lower, upper, fraction


def _interpolate_cia_coefficients(
    table: CiaTable,
    temperature_k: float,
    wavenumber_cm_inverse: NDArray[np.float64],
    *,
    temperature_extrapolation: str,
    spectral_extrapolation: str,
    coefficient_interpolation: str,
) -> NDArray[np.float64]:
    temperature = float(temperature_k)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise RobertValidationError(
            "CIA interpolation temperature must be finite and positive"
        )
    if temperature < table.temperature_K[0] or temperature > table.temperature_K[-1]:
        if temperature_extrapolation == "raise":
            raise RobertCoverageError(
                "atmosphere temperature is outside the CIA table temperature grid"
            )
    if temperature <= table.temperature_K[0]:
        native = table.k_cia[:, 0, :]
    elif temperature >= table.temperature_K[-1]:
        native = table.k_cia[:, -1, :]
    else:
        upper = int(np.searchsorted(table.temperature_K, temperature))
        lower = upper - 1
        fraction = (temperature - table.temperature_K[lower]) / (
            table.temperature_K[upper] - table.temperature_K[lower]
        )
        lower_values = table.k_cia[:, lower, :]
        upper_values = table.k_cia[:, upper, :]
        if coefficient_interpolation == "linear":
            native = (1.0 - fraction) * lower_values + fraction * upper_values
        else:
            both_positive = (lower_values > 0.0) & (upper_values > 0.0)
            native = (1.0 - fraction) * lower_values + fraction * upper_values
            native[both_positive] = np.exp(
                (1.0 - fraction) * np.log(lower_values[both_positive])
                + fraction * np.log(upper_values[both_positive])
            )

    requested_min = float(np.min(wavenumber_cm_inverse))
    requested_max = float(np.max(wavenumber_cm_inverse))
    if (
        requested_min < float(table.wavenumber_cm_inverse[0])
        or requested_max > float(table.wavenumber_cm_inverse[-1])
    ) and spectral_extrapolation == "raise":
        raise RobertCoverageError(
            "requested spectrum is outside the CIA table wavenumber grid"
        )
    coefficients = np.vstack(
        [
            np.interp(
                wavenumber_cm_inverse,
                table.wavenumber_cm_inverse,
                native[pair_index],
                left=0.0,
                right=0.0,
            )
            for pair_index in range(table.n_pairs)
        ]
    )
    coefficients.setflags(write=False)
    return coefficients


if _NUMBA_AVAILABLE:

    @njit(parallel=True)
    def _numba_cia_optical_depth_kernel(
        k_cia,
        temperature_lower,
        temperature_upper,
        temperature_fraction,
        spectral_lower,
        spectral_upper,
        spectral_fraction,
        spectral_inside,
        mixing_factors,
        tau_path,
        use_log_temperature,
    ):
        n_pairs, _, _ = k_cia.shape
        n_layers = temperature_lower.size
        n_spectral = spectral_lower.size
        tau = np.zeros((n_layers, n_spectral), dtype=np.float64)
        for layer_index in prange(n_layers):
            lower_temperature = temperature_lower[layer_index]
            upper_temperature = temperature_upper[layer_index]
            temperature_weight = temperature_fraction[layer_index]
            for spectral_index in range(n_spectral):
                if not spectral_inside[spectral_index]:
                    continue
                lower_spectral = spectral_lower[spectral_index]
                upper_spectral = spectral_upper[spectral_index]
                spectral_weight = spectral_fraction[spectral_index]
                coefficient = 0.0
                for pair_index in range(n_pairs):
                    mixing_factor = mixing_factors[pair_index, layer_index]
                    if mixing_factor <= 0.0:
                        continue
                    lower_left = k_cia[
                        pair_index,
                        lower_temperature,
                        lower_spectral,
                    ]
                    upper_left = k_cia[
                        pair_index,
                        upper_temperature,
                        lower_spectral,
                    ]
                    lower_right = k_cia[
                        pair_index,
                        lower_temperature,
                        upper_spectral,
                    ]
                    upper_right = k_cia[
                        pair_index,
                        upper_temperature,
                        upper_spectral,
                    ]
                    if use_log_temperature and lower_left > 0.0 and upper_left > 0.0:
                        left_value = np.exp(
                            (1.0 - temperature_weight) * np.log(lower_left)
                            + temperature_weight * np.log(upper_left)
                        )
                    else:
                        left_value = (
                            1.0 - temperature_weight
                        ) * lower_left + temperature_weight * upper_left
                    if use_log_temperature and lower_right > 0.0 and upper_right > 0.0:
                        right_value = np.exp(
                            (1.0 - temperature_weight) * np.log(lower_right)
                            + temperature_weight * np.log(upper_right)
                        )
                    else:
                        right_value = (
                            1.0 - temperature_weight
                        ) * lower_right + temperature_weight * upper_right
                    interpolated = (
                        1.0 - spectral_weight
                    ) * left_value + spectral_weight * right_value
                    coefficient += interpolated * mixing_factor
                tau[layer_index, spectral_index] = coefficient * tau_path[layer_index]
        return tau

else:

    def _numba_cia_optical_depth_kernel(*args):
        raise RobertValidationError(
            "compiled CIA optical-depth interpolation requires numba"
        )


def _composition_profile(
    composition: Mapping[str, NDArray[np.float64]],
    species: str,
    n_layers: int,
) -> NDArray[np.float64]:
    if species not in composition:
        profile = np.zeros(n_layers, dtype=float)
        profile.setflags(write=False)
        return profile
    profile = np.array(composition[species], dtype=float, copy=True)
    if profile.shape != (n_layers,):
        raise RobertValidationError(
            f"{species} composition must match pressure grid layers"
        )
    if not np.all(np.isfinite(profile)) or np.any(profile < 0.0):
        raise RobertValidationError(
            f"{species} composition must be finite and non-negative"
        )
    profile.setflags(write=False)
    return profile


def _required_vmr_profile(
    composition: Mapping[str, NDArray[np.float64]],
    species: str,
    n_layers: int,
    *,
    context: str,
) -> NDArray[np.float64]:
    """Return one required, physical VMR profile from an atmosphere."""

    if species not in composition:
        raise RobertValidationError(
            f"{context} requires explicit composition species: {species}"
        )
    profile = np.array(composition[species], dtype=float, copy=True)
    if profile.shape != (n_layers,):
        raise RobertValidationError(
            f"{context} species {species} must match pressure-grid layers"
        )
    if not np.all(np.isfinite(profile)) or np.any(profile < 0.0):
        raise RobertValidationError(
            f"{context} species {species} must be finite and non-negative"
        )
    if np.any(profile > 1.0):
        raise RobertValidationError(
            f"{context} species {species} VMR must not exceed one"
        )
    profile.setflags(write=False)
    return profile


def _decode_hdf_strings(dataset: object) -> tuple[str, ...]:
    values = np.asarray(dataset)
    decoded = []
    for value in values.reshape(-1):
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return tuple(decoded)


def _decode_hdf_first(dataset: object | None) -> str:
    if dataset is None:
        return ""
    values = _decode_hdf_strings(dataset)
    return "" if not values else values[0]


def _top_to_bottom_order(pressure_grid: PressureGrid) -> NDArray[np.int64]:
    pressure = pressure_values_in_unit(pressure_grid.centers, pressure_grid.unit, "pa")
    return np.argsort(pressure).astype(np.int64)


def _restore_layer_order(
    values_in_top_to_bottom_order: NDArray[np.float64],
    order: NDArray[np.int64],
) -> NDArray[np.float64]:
    restored = np.empty_like(values_in_top_to_bottom_order)
    restored[order] = values_in_top_to_bottom_order
    restored.setflags(write=False)
    return restored


def _readonly_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1 or array.size < 1:
        raise RobertValidationError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _readonly_array(
    values: ArrayLike,
    name: str,
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.shape != shape:
        raise RobertValidationError(f"{name} has incorrect shape")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _readonly_phase_function_moments(
    values: ArrayLike,
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.shape == (5, shape[1]):
        array = np.repeat(array[:, None, :], shape[0], axis=1)
    elif array.shape != (5, shape[0], shape[1]):
        raise RobertValidationError(
            "phase_function_moments must have shape (5, spectral) or "
            "(5, layer, spectral)"
        )
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(
            "phase_function_moments must contain only finite values"
        )
    limits = (2.0 * np.arange(5) + 1.0)[:, None, None]
    if np.any(np.abs(array) > limits * (1.0 + 1.0e-10)):
        raise RobertValidationError("phase_function_moments exceed physical bounds")
    if not np.allclose(array[0], 1.0, rtol=0.0, atol=1.0e-12):
        raise RobertValidationError("phase_function_moments[0] must equal one")
    array.setflags(write=False)
    return array
