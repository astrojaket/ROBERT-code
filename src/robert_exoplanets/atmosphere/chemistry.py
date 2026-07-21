"""Simple chemistry parameterizations."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
import re
from typing import Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.core import PressureGrid, RobertConfigError, RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping


DEFAULT_MOLECULAR_MASSES: Mapping[str, float] = {
    "H2": 2.01588,
    "He": 4.002602,
    "H2O": 18.01528,
    "CO": 28.0101,
    "CO2": 44.0095,
    "CH4": 16.04246,
    "NH3": 17.03052,
    "HCN": 27.0253,
    "N2": 28.0134,
    "O2": 31.9988,
    "Na": 22.98976928,
    "K": 39.0983,
    "SO2": 64.066,
    "H2S": 34.0809,
}


class ChemistryModel(Protocol):
    """Protocol shared by chemistry parameterizations."""

    name: str
    convention: str

    @property
    def species(self) -> tuple[str, ...]:
        """Species produced by this chemistry model."""

    def required_parameters(self) -> tuple[str, ...]:
        """Return required parameter names."""

    def evaluate(
        self,
        parameters: Mapping[str, float],
        pressure_grid: PressureGrid,
        temperature: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Evaluate composition on a pressure grid."""


class MeanMolecularWeightModel(Protocol):
    """Protocol shared by mean-molecular-weight models."""

    unit: str

    def required_parameters(self) -> tuple[str, ...]:
        """Return parameters needed to evaluate molecular masses."""

    def evaluate(
        self,
        composition: Mapping[str, NDArray[np.float64]],
        pressure_grid: PressureGrid,
        parameters: Mapping[str, float] | None = None,
    ) -> NDArray[np.float64]:
        """Evaluate mean molecular weight on a pressure grid."""


def _validate_species_name(species: str) -> str:
    name = str(species)
    if not name:
        raise RobertValidationError("species names must not be empty")
    return name


def _validate_temperature_layers(
    temperature: NDArray[np.float64],
    pressure_grid: PressureGrid,
) -> None:
    if temperature.shape != pressure_grid.centers.shape:
        raise RobertValidationError("temperature must match pressure grid layers")


def _parameter_value(
    parameters: Mapping[str, float],
    parameter_name: str,
    *,
    context: str,
) -> float:
    try:
        value = float(parameters[parameter_name])
    except KeyError as exc:
        raise RobertConfigError(f"missing required {context} parameter: {parameter_name}") from exc
    except (TypeError, ValueError) as exc:
        raise RobertValidationError(f"{context} parameter must be numeric: {parameter_name}") from exc
    if not np.isfinite(value):
        raise RobertValidationError(f"{context} parameter must be finite: {parameter_name}")
    return value


def _readonly_constant_profile(value: float, n_layers: int) -> NDArray[np.float64]:
    profile = np.full(n_layers, value, dtype=float)
    profile.setflags(write=False)
    return profile


def _readonly_layer_array(
    values: NDArray[np.float64],
    name: str,
    n_layers: int,
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise RobertValidationError(f"{name} must be one-dimensional")
    if array.shape != (n_layers,):
        raise RobertValidationError(f"{name} must match pressure grid layers")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class _FastChemRuntime:
    pyfastchem: object
    fastchem: object
    solar_abundances: NDArray[np.float64]
    species_indices: NDArray[np.int64]


def _pressure_values_pa(pressure_grid: PressureGrid) -> NDArray[np.float64]:
    unit = pressure_grid.unit.strip().lower()
    values = np.asarray(pressure_grid.centers, dtype=float)
    if unit in {"pa", "pascal", "pascals"}:
        pressure = np.array(values, dtype=float, copy=True)
    elif unit in {"bar", "bars"}:
        pressure = np.array(values * 1.0e5, dtype=float, copy=True)
    elif unit in {"mbar", "millibar", "millibars"}:
        pressure = np.array(values * 1.0e2, dtype=float, copy=True)
    elif unit in {"atm", "atmosphere", "atmospheres"}:
        pressure = np.array(values * 101325.0, dtype=float, copy=True)
    else:
        raise RobertValidationError(f"unsupported pressure unit for FastChem chemistry: {pressure_grid.unit}")
    pressure.setflags(write=False)
    return pressure


def _pressure_values_bar(pressure_grid: PressureGrid) -> NDArray[np.float64]:
    """Return immutable layer-center pressures converted explicitly to bar."""

    pressure = np.array(_pressure_values_pa(pressure_grid) / 1.0e5, copy=True)
    pressure.setflags(write=False)
    return pressure


_PARAMETER_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MOLECULAR_FORMULA_PATTERN = re.compile(r"(?:[A-Z][a-z]?\d*)+")
_MOLECULAR_TOKEN_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")


def _validate_parameter_name(parameter_name: str, *, context: str) -> str:
    name = str(parameter_name).strip()
    if not _PARAMETER_NAME_PATTERN.fullmatch(name):
        raise RobertValidationError(
            f"{context} must be a non-empty identifier containing only letters, "
            "digits, and underscores"
        )
    return name


@dataclass(frozen=True)
class QuenchGroup:
    """Species sharing one retrieved pressure-quench parameter.

    ``pressure_parameter`` represents ``log10(P_q / bar)``. A one-species
    group is molecular quenching; a multi-species group is coupled quenching.
    """

    pressure_parameter: str
    species: tuple[str, ...]

    def __post_init__(self) -> None:
        parameter = _validate_parameter_name(
            self.pressure_parameter,
            context="quench pressure parameter name",
        )
        species = tuple(_validate_species_name(item) for item in self.species)
        if not species:
            raise RobertValidationError(
                "quench groups must contain at least one species"
            )
        if len(set(species)) != len(species):
            raise RobertValidationError(
                "species within a quench group must be unique"
            )
        object.__setattr__(self, "pressure_parameter", parameter)
        object.__setattr__(self, "species", species)

    @classmethod
    def molecular(
        cls,
        species: str,
        pressure_parameter: str | None = None,
    ) -> "QuenchGroup":
        """Build a one-species molecular-quenching group."""

        label = _validate_species_name(species)
        return cls(
            pressure_parameter=(
                pressure_parameter
                if pressure_parameter is not None
                else f"log_Pq_{label}"
            ),
            species=(label,),
        )


@dataclass(frozen=True)
class QuenchDiagnostics:
    """Composition closure diagnostics before and after pressure quenching."""

    base_vmr_sum: NDArray[np.float64]
    quenched_vmr_sum: NDArray[np.float64]
    vmr_sum_drift: NDArray[np.float64]
    elemental_budget_drift: Mapping[str, NDArray[np.float64]] | None
    unparsed_species: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        arrays = {}
        for name in ("base_vmr_sum", "quenched_vmr_sum", "vmr_sum_drift"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            arrays[name] = value
        elemental = None
        if self.elemental_budget_drift is not None:
            values: dict[str, NDArray[np.float64]] = {}
            for element, profile in self.elemental_budget_drift.items():
                value = np.array(profile, dtype=float, copy=True)
                value.setflags(write=False)
                values[str(element)] = value
            elemental = immutable_mapping(values)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "elemental_budget_drift", elemental)
        object.__setattr__(
            self,
            "unparsed_species",
            tuple(str(item) for item in self.unparsed_species),
        )


@dataclass(frozen=True)
class PressureQuenchChemistry:
    """Generic pressure-quench decorator around a base chemistry model.

    The transform follows Taylor et al. (2026), arXiv:2607.06491: below the
    quench level (higher pressure) the base profile is retained, while at and
    above it (lower pressure) the profile is frozen to ``X_eq(P_q)``.

    The paper does not specify an off-grid sampling convention. ROBERT assumes
    linear VMR interpolation in ``log10(P / bar)`` between layer centers. The
    inclusive layer-center domain is enforced; extrapolation is forbidden.
    Quenched profiles are not renormalized.
    """

    base_model: ChemistryModel
    groups: tuple[QuenchGroup, ...]
    name: str = "pressure-quench"
    preset: str = "custom"
    convention: str = field(init=False)
    metadata: Mapping[str, str] = field(init=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("chemistry name must not be empty")
        if not str(self.preset).strip():
            raise RobertValidationError("quench preset name must not be empty")
        groups = tuple(self.groups)
        if not groups:
            raise RobertValidationError(
                "pressure-quench chemistry requires at least one quench group"
            )
        if any(not isinstance(group, QuenchGroup) for group in groups):
            raise RobertValidationError(
                "pressure-quench groups must be QuenchGroup instances"
            )

        parameters = tuple(group.pressure_parameter for group in groups)
        if len(set(parameters)) != len(parameters):
            raise RobertValidationError(
                "quench pressure parameter names must be unique"
            )
        selected_species = tuple(
            species for group in groups for species in group.species
        )
        if len(set(selected_species)) != len(selected_species):
            raise RobertValidationError(
                "a species may belong to only one quench group"
            )
        missing = tuple(
            species
            for species in selected_species
            if species not in self.base_model.species
        )
        if missing:
            raise RobertValidationError(
                "quench species missing from base chemistry: " + ", ".join(missing)
            )

        base_parameters = tuple(self.base_model.required_parameters())
        collisions = sorted(set(base_parameters).intersection(parameters))
        if collisions:
            raise RobertValidationError(
                "quench pressure parameters collide with base chemistry parameters: "
                + ", ".join(collisions)
            )
        if len(set(base_parameters)) != len(base_parameters):
            raise RobertValidationError(
                "base chemistry required parameters must be unique"
            )

        convention = str(self.base_model.convention)
        if convention != "volume_mixing_ratio":
            raise RobertValidationError(
                "pressure-quench chemistry currently requires volume_mixing_ratio"
            )
        base_metadata = dict(getattr(self.base_model, "metadata", {}))
        group_record = ";".join(
            f"{group.pressure_parameter}:{'|'.join(group.species)}"
            for group in groups
        )
        base_metadata.update(
            {
                "quench_scheme": "pressure_quench",
                "quench_preset": str(self.preset),
                "quench_groups": group_record,
                "quench_pressure_semantics": "log10(P_q/bar)",
                "quench_pressure_domain": "inclusive_layer_centers",
                "quench_interpolation": "linear_vmr_in_log10_pressure",
                "quench_profile_rule": "base_for_P_gt_Pq;freeze_to_Xeq_Pq_for_P_le_Pq",
                "quench_closure_policy": "no_renormalization",
                "quench_elemental_grouping": (
                    "grouped_molecular_profiles_not_elemental_conservation"
                ),
                "quench_reference": "Taylor_et_al_2026_arXiv_2607.06491",
                "quench_code_convention": "ROBERT_assumption_not_NemesisPy_parity",
                "quench_validation_level": (
                    "unit_tested_not_cross_framework_or_science_demonstrated"
                ),
            }
        )
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "preset", str(self.preset))
        object.__setattr__(self, "convention", convention)
        object.__setattr__(self, "metadata", immutable_mapping(base_metadata))

    @property
    def species(self) -> tuple[str, ...]:
        """Species produced by the decorated base model."""

        return self.base_model.species

    def required_parameters(self) -> tuple[str, ...]:
        """Return base parameters followed by quench parameters in group order."""

        return (
            *self.base_model.required_parameters(),
            *(group.pressure_parameter for group in self.groups),
        )

    @classmethod
    def molecular(
        cls,
        base_model: ChemistryModel,
        species_parameters: Mapping[str, str],
    ) -> "PressureQuenchChemistry":
        """Decorate arbitrary base species with independent quench pressures."""

        return cls(
            base_model=base_model,
            groups=tuple(
                QuenchGroup.molecular(species, parameter)
                for species, parameter in species_parameters.items()
            ),
        )

    @classmethod
    def taylor_2026_hot_jupiter_element_grouped(
        cls,
        base_model: ChemistryModel,
    ) -> "PressureQuenchChemistry":
        """Build the paper-exact species grouping (which deliberately omits N2)."""

        return cls(
            base_model=base_model,
            groups=(
                QuenchGroup(
                    pressure_parameter="log_Pq_C",
                    species=("H2O", "CO", "CO2", "CH4"),
                ),
                QuenchGroup(
                    pressure_parameter="log_Pq_N",
                    species=("NH3",),
                ),
            ),
            preset="taylor_2026_hot_jupiter_element_grouped",
        )

    def evaluate(
        self,
        parameters: Mapping[str, float],
        pressure_grid: PressureGrid,
        temperature: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Evaluate and pressure-quench the base composition."""

        composition, _ = self.evaluate_with_diagnostics(
            parameters,
            pressure_grid,
            temperature,
        )
        return composition

    def evaluate_with_diagnostics(
        self,
        parameters: Mapping[str, float],
        pressure_grid: PressureGrid,
        temperature: NDArray[np.float64],
    ) -> tuple[dict[str, NDArray[np.float64]], QuenchDiagnostics]:
        """Evaluate profiles and report VMR-sum and feasible elemental drift."""

        _validate_temperature_layers(temperature, pressure_grid)
        base = self.base_model.evaluate(parameters, pressure_grid, temperature)
        if tuple(base) != self.base_model.species:
            raise RobertValidationError(
                "base chemistry output species do not match its declared species"
            )
        profiles = {
            species: _readonly_layer_array(
                values,
                f"{species} base chemistry VMR",
                pressure_grid.n_layers,
            )
            for species, values in base.items()
        }
        pressure_bar = _pressure_values_bar(pressure_grid)
        log_pressure_bar = np.log10(pressure_bar)
        domain_min = float(np.min(log_pressure_bar))
        domain_max = float(np.max(log_pressure_bar))
        order = np.argsort(log_pressure_bar)

        for group in self.groups:
            log_quench_pressure = _parameter_value(
                parameters,
                group.pressure_parameter,
                context="pressure-quench chemistry",
            )
            if not domain_min <= log_quench_pressure <= domain_max:
                raise RobertValidationError(
                    f"quench pressure {group.pressure_parameter}={log_quench_pressure:.17g} "
                    "lies outside the inclusive pressure-grid layer-center domain "
                    f"[{domain_min:.17g}, {domain_max:.17g}] log10(bar)"
                )
            quench_pressure_bar = 10.0**log_quench_pressure
            freeze_mask = pressure_bar <= quench_pressure_bar
            for species in group.species:
                equilibrium = np.asarray(profiles[species], dtype=float)
                quench_vmr = float(
                    np.interp(
                        log_quench_pressure,
                        log_pressure_bar[order],
                        equilibrium[order],
                    )
                )
                transformed = np.array(equilibrium, dtype=float, copy=True)
                transformed[freeze_mask] = quench_vmr
                transformed.setflags(write=False)
                profiles[species] = transformed

        diagnostics = _quench_diagnostics(base, profiles, pressure_grid.n_layers)
        return profiles, diagnostics


def _quench_diagnostics(
    base: Mapping[str, NDArray[np.float64]],
    quenched: Mapping[str, NDArray[np.float64]],
    n_layers: int,
) -> QuenchDiagnostics:
    base_sum = np.zeros(n_layers, dtype=float)
    quenched_sum = np.zeros(n_layers, dtype=float)
    for species in base:
        base_sum += np.asarray(base[species], dtype=float)
        quenched_sum += np.asarray(quenched[species], dtype=float)

    formulae: dict[str, Mapping[str, int]] = {}
    unparsed: list[str] = []
    for species in base:
        formula = _molecular_formula(species)
        if formula is None:
            unparsed.append(species)
        else:
            formulae[species] = formula

    elemental_drift: dict[str, NDArray[np.float64]] | None = None
    if not unparsed:
        elemental_drift = {}
        elements = sorted(
            {element for formula in formulae.values() for element in formula}
        )
        for element in elements:
            drift = np.zeros(n_layers, dtype=float)
            for species, formula in formulae.items():
                coefficient = formula.get(element, 0)
                if coefficient:
                    drift += coefficient * (
                        np.asarray(quenched[species], dtype=float)
                        - np.asarray(base[species], dtype=float)
                    )
            drift.setflags(write=False)
            elemental_drift[element] = drift

    return QuenchDiagnostics(
        base_vmr_sum=base_sum,
        quenched_vmr_sum=quenched_sum,
        vmr_sum_drift=quenched_sum - base_sum,
        elemental_budget_drift=elemental_drift,
        unparsed_species=tuple(unparsed),
    )


def _molecular_formula(species: str) -> Mapping[str, int] | None:
    if not _MOLECULAR_FORMULA_PATTERN.fullmatch(species):
        return None
    counts: dict[str, int] = {}
    for element, count in _MOLECULAR_TOKEN_PATTERN.findall(species):
        counts[element] = counts.get(element, 0) + (int(count) if count else 1)
    return counts or None


@dataclass(frozen=True)
class BackgroundGasMixture:
    """Relative split for background gases that fill unused VMR budget.

    Fractions are interpreted as relative shares of the remaining atmospheric
    volume mixing ratio and are normalized during validation.
    """

    fractions: Mapping[str, float] = field(
        default_factory=lambda: {"H2": 0.8547, "He": 0.1453}
    )
    name: str = "background-gas-mixture"

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("background mixture name must not be empty")
        if not self.fractions:
            raise RobertValidationError("background fractions must contain at least one species")

        fractions: dict[str, float] = {}
        for species, value in self.fractions.items():
            name = _validate_species_name(species)
            fraction = float(value)
            if not np.isfinite(fraction) or fraction < 0.0:
                raise RobertValidationError(
                    "background gas fractions must be finite and non-negative"
                )
            fractions[name] = fraction

        total = float(sum(fractions.values()))
        if total <= 0.0:
            raise RobertValidationError("background gas fractions must sum to a positive value")

        normalized = {species: fraction / total for species, fraction in fractions.items()}
        object.__setattr__(self, "fractions", normalized)

    @classmethod
    def hydrogen_helium(cls, h2_fraction: float = 0.8547) -> "BackgroundGasMixture":
        """Build the standard H2/He background split used by the HAT-P-32b workflow."""

        h2_value = float(h2_fraction)
        if not np.isfinite(h2_value) or h2_value < 0.0 or h2_value > 1.0:
            raise RobertValidationError("h2_fraction must be finite and between 0 and 1")
        return cls({"H2": h2_value, "He": 1.0 - h2_value})

    @property
    def species(self) -> tuple[str, ...]:
        """Background species in insertion order."""

        return tuple(self.fractions)


@dataclass(frozen=True)
class FixedMeanMolecularWeight:
    """Layer-constant mean molecular weight model."""

    value: float = 2.3
    unit: str = "amu"

    def __post_init__(self) -> None:
        value = float(self.value)
        if not np.isfinite(value) or value <= 0.0:
            raise RobertValidationError("mean molecular weight must be finite and positive")
        if not self.unit:
            raise RobertValidationError("mean molecular weight unit must not be empty")
        object.__setattr__(self, "value", value)

    def evaluate(
        self,
        composition: Mapping[str, NDArray[np.float64]],
        pressure_grid: PressureGrid,
        parameters: Mapping[str, float] | None = None,
    ) -> NDArray[np.float64]:
        """Return a fixed mean molecular weight for every layer."""

        del composition, parameters
        return _readonly_constant_profile(self.value, pressure_grid.n_layers)

    def required_parameters(self) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class CompositionMeanMolecularWeight:
    """Mean molecular weight derived from VMR composition profiles.

    Molecular masses are in atomic mass units. With the default
    `normalization="require"` policy, each layer's VMR sum must be close to
    one so trace-only chemistry cannot accidentally define a bulk atmosphere.
    `normalization="raw_sum"` preserves the direct weighted sum used by the
    local FastChem benchmark path when only selected species are retained.
    """

    molecular_masses: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MOLECULAR_MASSES)
    )
    molecular_mass_parameters: Mapping[str, str] = field(default_factory=dict)
    normalization: str = "require"
    sum_tolerance: float = 1.0e-8
    unit: str = "amu"

    def __post_init__(self) -> None:
        if self.normalization not in {"require", "normalize", "raw_sum"}:
            raise RobertValidationError("normalization must be 'require', 'normalize', or 'raw_sum'")
        if not self.unit:
            raise RobertValidationError("mean molecular weight unit must not be empty")
        tolerance = float(self.sum_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise RobertValidationError("sum_tolerance must be finite and non-negative")

        masses: dict[str, float] = {}
        for species, value in self.molecular_masses.items():
            name = _validate_species_name(species)
            mass = float(value)
            if not np.isfinite(mass) or mass <= 0.0:
                raise RobertValidationError("molecular masses must be finite and positive")
            masses[name] = mass

        mass_parameters: dict[str, str] = {}
        for species, parameter in self.molecular_mass_parameters.items():
            name = _validate_species_name(species)
            parameter_name = str(parameter).strip()
            if not parameter_name:
                raise RobertValidationError(
                    "molecular-mass parameter names must not be empty"
                )
            mass_parameters[name] = parameter_name
        if len(set(mass_parameters.values())) != len(mass_parameters):
            raise RobertValidationError(
                "molecular-mass parameter names must be unique"
            )

        object.__setattr__(self, "molecular_masses", masses)
        object.__setattr__(
            self,
            "molecular_mass_parameters",
            immutable_mapping(mass_parameters),
        )
        object.__setattr__(self, "sum_tolerance", tolerance)

    def required_parameters(self) -> tuple[str, ...]:
        return tuple(self.molecular_mass_parameters.values())

    def evaluate(
        self,
        composition: Mapping[str, NDArray[np.float64]],
        pressure_grid: PressureGrid,
        parameters: Mapping[str, float] | None = None,
    ) -> NDArray[np.float64]:
        """Return composition-weighted mean molecular weight by layer."""

        if not composition:
            raise RobertValidationError("composition must contain at least one species")

        numerator = np.zeros(pressure_grid.n_layers, dtype=float)
        total = np.zeros(pressure_grid.n_layers, dtype=float)
        parameter_values = {} if parameters is None else parameters
        for species, values in composition.items():
            name = _validate_species_name(species)
            if name in self.molecular_mass_parameters:
                molecular_mass = _parameter_value(
                    parameter_values,
                    self.molecular_mass_parameters[name],
                    context=f"{name} molecular mass",
                )
                if molecular_mass <= 0.0:
                    raise RobertValidationError(
                        f"{name} molecular mass must be positive"
                    )
            elif name in self.molecular_masses:
                molecular_mass = self.molecular_masses[name]
            else:
                raise RobertValidationError(f"missing molecular mass for species: {name}")
            profile = _readonly_layer_array(
                values,
                f"{name} composition",
                pressure_grid.n_layers,
            )
            if np.any(profile < 0.0):
                raise RobertValidationError("composition values must be non-negative")
            numerator += profile * molecular_mass
            total += profile

        if np.any(total <= 0.0):
            raise RobertValidationError("composition VMR sums must be positive")
        if self.normalization == "raw_sum":
            mean_molecular_weight = numerator
            mean_molecular_weight.setflags(write=False)
            return mean_molecular_weight
        if self.normalization == "require" and np.any(np.abs(total - 1.0) > self.sum_tolerance):
            raise RobertValidationError("composition VMRs must sum to one to derive mean molecular weight")

        mean_molecular_weight = numerator / total
        mean_molecular_weight.setflags(write=False)
        return mean_molecular_weight


@dataclass(frozen=True)
class FastChemEquilibriumChemistry:
    """FastChem equilibrium chemistry wrapper.

    FastChem is an optional runtime dependency. Metallicity is log10 relative
    to solar, C/O is imposed by setting the carbon abundance relative to
    oxygen, and pressure is passed to FastChem in bar.
    """

    fastchem_path: str | Path
    fastchem_species: tuple[str, ...] = (
        "H2O1",
        "C1O2",
        "C1O1",
        "C1H4",
        "H3N1",
        "C1H1N1_1",
        "H2",
        "He",
    )
    labels: tuple[str, ...] = (
        "H2O",
        "CO2",
        "CO",
        "CH4",
        "NH3",
        "HCN",
        "H2",
        "He",
    )
    metallicity_parameter_name: str = "metallicity"
    carbon_to_oxygen_parameter_name: str = "CtoO"
    constant_log10_vmr_parameters: Mapping[str, str] = field(default_factory=dict)
    name: str = "fastchem-equilibrium"
    convention: str = "volume_mixing_ratio"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = Path(self.fastchem_path).expanduser()
        if not self.name:
            raise RobertValidationError("chemistry name must not be empty")
        if self.convention != "volume_mixing_ratio":
            raise RobertValidationError("FastChem chemistry currently returns volume_mixing_ratio")
        if len(self.fastchem_species) != len(self.labels):
            raise RobertValidationError("fastchem_species and labels must have the same length")
        if not self.fastchem_species:
            raise RobertValidationError("FastChem chemistry requires at least one species")
        labels = tuple(_validate_species_name(label) for label in self.labels)
        if len(set(labels)) != len(labels):
            raise RobertValidationError("FastChem labels must be unique")
        if not self.metallicity_parameter_name or not self.carbon_to_oxygen_parameter_name:
            raise RobertValidationError("FastChem parameter names must not be empty")
        overrides = {
            _validate_species_name(species): str(parameter)
            for species, parameter in self.constant_log10_vmr_parameters.items()
        }
        unknown_overrides = sorted(set(overrides) - set(labels))
        if unknown_overrides:
            raise RobertValidationError(
                "constant FastChem overrides are not configured species: "
                + ", ".join(unknown_overrides)
            )
        if any(not parameter for parameter in overrides.values()):
            raise RobertValidationError(
                "constant FastChem override parameter names must not be empty"
            )
        if len(set(overrides.values())) != len(overrides):
            raise RobertValidationError(
                "constant FastChem override parameter names must be unique"
            )
        object.__setattr__(self, "fastchem_path", path)
        object.__setattr__(self, "fastchem_species", tuple(str(item) for item in self.fastchem_species))
        object.__setattr__(self, "labels", labels)
        object.__setattr__(
            self,
            "constant_log10_vmr_parameters",
            immutable_mapping(overrides),
        )
        object.__setattr__(
            self,
            "metadata",
            immutable_mapping(
                {
                    "fastchem_path": str(path),
                    "metallicity_parameter": self.metallicity_parameter_name,
                    "carbon_to_oxygen_parameter": self.carbon_to_oxygen_parameter_name,
                    "constant_log10_vmr_parameters": ",".join(
                        f"{species}:{parameter}"
                        for species, parameter in overrides.items()
                    ),
                    **dict(self.metadata),
                }
            ),
        )

    @property
    def species(self) -> tuple[str, ...]:
        """Species produced by this chemistry model."""

        return self.labels

    def required_parameters(self) -> tuple[str, ...]:
        """Return required FastChem parameter names."""

        return (
            self.metallicity_parameter_name,
            self.carbon_to_oxygen_parameter_name,
            *self.constant_log10_vmr_parameters.values(),
        )

    def evaluate(
        self,
        parameters: Mapping[str, float],
        pressure_grid: PressureGrid,
        temperature: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Return FastChem VMR profiles on the supplied P-T grid."""

        _validate_temperature_layers(temperature, pressure_grid)
        metallicity_dex = _parameter_value(
            parameters,
            self.metallicity_parameter_name,
            context="FastChem chemistry",
        )
        carbon_to_oxygen = _parameter_value(
            parameters,
            self.carbon_to_oxygen_parameter_name,
            context="FastChem chemistry",
        )
        if carbon_to_oxygen < 0.0:
            raise RobertValidationError("FastChem C/O must be non-negative")

        runtime = self._runtime
        element_abundances = np.array(runtime.solar_abundances, dtype=float, copy=True)
        metallicity_scale = 10.0**metallicity_dex
        for index in range(runtime.fastchem.getElementNumber()):
            symbol = runtime.fastchem.getElementSymbol(index)
            if symbol not in {"H", "He"}:
                element_abundances[index] *= metallicity_scale

        oxygen_index = runtime.fastchem.getElementIndex("O")
        carbon_index = runtime.fastchem.getElementIndex("C")
        element_abundances[carbon_index] = carbon_to_oxygen * element_abundances[oxygen_index]
        runtime.fastchem.setElementAbundances(element_abundances)

        pyfastchem = runtime.pyfastchem
        pressure_pa = _pressure_values_pa(pressure_grid)
        input_data = pyfastchem.FastChemInput()
        input_data.pressure = pressure_pa / 1.0e5
        input_data.temperature = np.asarray(temperature, dtype=float)

        output_data = pyfastchem.FastChemOutput()
        flag = runtime.fastchem.calcDensities(input_data, output_data)
        if int(flag) != 0:
            raise RobertValidationError(f"FastChem failed with status flag {flag}")

        k_b_cgs = 1.380649e-16
        gas_number_density = pressure_pa * 10.0 / (k_b_cgs * temperature)
        number_densities = np.asarray(output_data.number_densities, dtype=float)
        vmr = number_densities[:, runtime.species_indices] / gas_number_density[:, None]
        if vmr.shape != (pressure_grid.n_layers, len(self.labels)):
            raise RobertValidationError("FastChem returned an unexpected VMR shape")
        if not np.all(np.isfinite(vmr)) or np.any(vmr < 0.0):
            raise RobertValidationError("FastChem returned invalid VMR values")

        profiles: dict[str, NDArray[np.float64]] = {}
        for index, label in enumerate(self.labels):
            profiles[label] = _readonly_layer_array(vmr[:, index], f"{label} FastChem VMR", pressure_grid.n_layers)
        for label, parameter_name in self.constant_log10_vmr_parameters.items():
            log10_vmr = _parameter_value(
                parameters,
                parameter_name,
                context="FastChem constant abundance override",
            )
            if log10_vmr > 0.0:
                raise RobertValidationError(
                    f"{label} constant log10 VMR must not exceed zero"
                )
            ratio = float(10.0**log10_vmr)
            if not np.isfinite(ratio) or ratio <= 0.0:
                raise RobertValidationError(
                    f"{label} constant VMR must be finite and positive"
                )
            profiles[label] = _readonly_constant_profile(
                ratio, pressure_grid.n_layers
            )
        return profiles

    @cached_property
    def _runtime(self):
        try:
            import pyfastchem
        except Exception as exc:  # pragma: no cover - dependency availability is environment-specific.
            raise RobertConfigError("FastChem chemistry requires the optional pyfastchem package") from exc

        abundances_path = self.fastchem_path / "input" / "element_abundances" / "asplund_2009.dat"
        logk_path = self.fastchem_path / "input" / "logK" / "logK.dat"
        if not abundances_path.exists() or not logk_path.exists():
            raise RobertConfigError(
                "FastChem chemistry could not find asplund_2009.dat and logK.dat "
                f"under {self.fastchem_path}"
            )

        fastchem = pyfastchem.FastChem(str(abundances_path), str(logk_path), 0)
        species_indices: list[int] = []
        missing: list[str] = []
        for species in self.fastchem_species:
            index = fastchem.getGasSpeciesIndex(species)
            if index == pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
                missing.append(species)
            species_indices.append(int(index))
        if missing:
            raise RobertConfigError("FastChem does not know species: " + ", ".join(missing))

        return _FastChemRuntime(
            pyfastchem=pyfastchem,
            fastchem=fastchem,
            solar_abundances=np.array(fastchem.getElementAbundances(), dtype=float),
            species_indices=np.array(species_indices, dtype=int),
        )


@dataclass(frozen=True)
class ConstantChemistry:
    """Layer-constant trace composition.

    This model repeats declared mixing ratios through every atmospheric layer.
    It is a fixture-style chemistry model for wiring tests and examples, not an
    equilibrium or disequilibrium chemistry calculation.
    """

    mixing_ratios: Mapping[str, float]
    name: str = "constant-chemistry"
    convention: str = "volume_mixing_ratio"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mixing_ratios:
            raise RobertValidationError("mixing_ratios must contain at least one species")
        if not self.name:
            raise RobertValidationError("chemistry name must not be empty")
        if not self.convention:
            raise RobertValidationError("composition convention must not be empty")

        ratios: dict[str, float] = {}
        for species, value in self.mixing_ratios.items():
            name = _validate_species_name(species)
            ratio = float(value)
            if not np.isfinite(ratio) or ratio < 0.0:
                raise RobertValidationError("mixing ratios must be finite and non-negative")
            ratios[name] = ratio

        if self.convention == "volume_mixing_ratio" and sum(ratios.values()) > 1.0:
            raise RobertValidationError("volume mixing ratios must sum to no more than one")

        object.__setattr__(self, "mixing_ratios", ratios)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def species(self) -> tuple[str, ...]:
        """Species produced by this chemistry model."""

        return tuple(self.mixing_ratios)

    def required_parameters(self) -> tuple[str, ...]:
        """Return required parameter names."""

        return ()

    def evaluate(
        self,
        parameters: Mapping[str, float],
        pressure_grid: PressureGrid,
        temperature: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Return layer-constant composition profiles."""

        _validate_temperature_layers(temperature, pressure_grid)

        profiles: dict[str, NDArray[np.float64]] = {}
        for species, ratio in self.mixing_ratios.items():
            profiles[species] = _readonly_constant_profile(ratio, pressure_grid.n_layers)
        return profiles


@dataclass(frozen=True)
class FreeChemistry:
    """Free constant-with-altitude chemistry with optional background fill.

    Active gases are represented by one layer-constant VMR each. A species can
    either be fixed at construction time or read from the parameter mapping.
    If `fill_background` is true, the unused VMR budget is assigned to the
    declared background mixture.
    """

    active_species: tuple[str, ...]
    background: BackgroundGasMixture | None = field(default_factory=BackgroundGasMixture)
    fixed_mixing_ratios: Mapping[str, float] | None = None
    parameter_names: Mapping[str, str] | None = None
    parameter_mode: str = "linear"
    fill_background: bool = True
    excess_policy: str = "raise"
    name: str = "free-chemistry"
    convention: str = "volume_mixing_ratio"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("chemistry name must not be empty")
        if self.convention != "volume_mixing_ratio":
            raise RobertValidationError("FreeChemistry currently requires volume_mixing_ratio")
        if self.parameter_mode not in {"linear", "log10"}:
            raise RobertValidationError("parameter_mode must be 'linear' or 'log10'")
        if self.excess_policy not in {"raise", "normalize"}:
            raise RobertValidationError("excess_policy must be 'raise' or 'normalize'")
        if not self.active_species:
            raise RobertValidationError("active_species must contain at least one species")

        active_species = tuple(_validate_species_name(species) for species in self.active_species)
        if len(set(active_species)) != len(active_species):
            raise RobertValidationError("active_species must be unique")

        background = self.background
        if background is not None and not isinstance(background, BackgroundGasMixture):
            background = BackgroundGasMixture(background)
        if self.fill_background and background is None:
            raise RobertValidationError("fill_background=True requires a background mixture")
        if self.fill_background and background is not None:
            overlap = set(active_species).intersection(background.species)
            if overlap:
                species_list = ", ".join(sorted(overlap))
                raise RobertValidationError(
                    f"active species must not overlap background species: {species_list}"
                )

        fixed: dict[str, float] = {}
        for species, value in (self.fixed_mixing_ratios or {}).items():
            name = _validate_species_name(species)
            if name not in active_species:
                raise RobertValidationError(
                    f"fixed chemistry species is not active: {name}"
                )
            mixing_ratio = float(value)
            if not np.isfinite(mixing_ratio) or mixing_ratio < 0.0:
                raise RobertValidationError(
                    "fixed chemistry mixing ratios must be finite and non-negative"
                )
            fixed[name] = mixing_ratio

        parameter_names: dict[str, str] = {}
        for species in active_species:
            parameter_name = species
            if self.parameter_names and species in self.parameter_names:
                parameter_name = str(self.parameter_names[species])
            if not parameter_name:
                raise RobertValidationError("chemistry parameter names must not be empty")
            parameter_names[species] = parameter_name
        unknown_names = set(self.parameter_names or {}).difference(active_species)
        if unknown_names:
            species_list = ", ".join(sorted(unknown_names))
            raise RobertValidationError(
                f"parameter_names contains species that are not active: {species_list}"
            )
        required_names = [
            parameter_names[species]
            for species in active_species
            if species not in fixed
        ]
        if len(set(required_names)) != len(required_names):
            raise RobertValidationError("free chemistry parameter names must be unique")

        object.__setattr__(self, "active_species", active_species)
        object.__setattr__(self, "background", background)
        object.__setattr__(self, "fixed_mixing_ratios", fixed)
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def species(self) -> tuple[str, ...]:
        """Species produced by this chemistry model."""

        if self.fill_background and self.background is not None:
            return self.active_species + self.background.species
        return self.active_species

    def required_parameters(self) -> tuple[str, ...]:
        """Return required parameter names."""

        fixed_species = set(self.fixed_mixing_ratios or {})
        return tuple(
            self.parameter_names[species]
            for species in self.active_species
            if species not in fixed_species
        )

    def evaluate(
        self,
        parameters: Mapping[str, float],
        pressure_grid: PressureGrid,
        temperature: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Return layer-constant free-chemistry composition profiles."""

        _validate_temperature_layers(temperature, pressure_grid)

        active_values: dict[str, float] = {}
        for species in self.active_species:
            if species in self.fixed_mixing_ratios:
                mixing_ratio = self.fixed_mixing_ratios[species]
            else:
                parameter_value = _parameter_value(
                    parameters,
                    self.parameter_names[species],
                    context="free chemistry",
                )
                if self.parameter_mode == "log10":
                    mixing_ratio = 10.0**parameter_value
                else:
                    mixing_ratio = parameter_value

            if not np.isfinite(mixing_ratio) or mixing_ratio < 0.0:
                raise RobertValidationError("free chemistry VMRs must be finite and non-negative")
            active_values[species] = float(mixing_ratio)

        active_total = float(sum(active_values.values()))
        background_total = 0.0
        if active_total > 1.0 + 1.0e-12:
            if self.excess_policy == "raise":
                raise RobertValidationError("free chemistry active VMRs sum to more than one")
            active_values = {
                species: value / active_total
                for species, value in active_values.items()
            }
            active_total = 1.0
        if self.fill_background:
            background_total = max(0.0, 1.0 - active_total)

        profiles: dict[str, NDArray[np.float64]] = {}
        for species in self.active_species:
            profiles[species] = _readonly_constant_profile(
                active_values[species],
                pressure_grid.n_layers,
            )

        if self.fill_background and self.background is not None:
            for species, fraction in self.background.fractions.items():
                profiles[species] = _readonly_constant_profile(
                    background_total * fraction,
                    pressure_grid.n_layers,
                )

        return profiles
