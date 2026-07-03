"""Simple chemistry parameterizations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.core import PressureGrid, RobertConfigError, RobertValidationError


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

    def evaluate(
        self,
        composition: Mapping[str, NDArray[np.float64]],
        pressure_grid: PressureGrid,
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
    ) -> NDArray[np.float64]:
        """Return a fixed mean molecular weight for every layer."""

        return _readonly_constant_profile(self.value, pressure_grid.n_layers)


@dataclass(frozen=True)
class CompositionMeanMolecularWeight:
    """Mean molecular weight derived from VMR composition profiles.

    Molecular masses are in atomic mass units. With the default
    `normalization="require"` policy, each layer's VMR sum must be close to
    one so trace-only chemistry cannot accidentally define a bulk atmosphere.
    """

    molecular_masses: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MOLECULAR_MASSES)
    )
    normalization: str = "require"
    sum_tolerance: float = 1.0e-8
    unit: str = "amu"

    def __post_init__(self) -> None:
        if self.normalization not in {"require", "normalize"}:
            raise RobertValidationError("normalization must be 'require' or 'normalize'")
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

        object.__setattr__(self, "molecular_masses", masses)
        object.__setattr__(self, "sum_tolerance", tolerance)

    def evaluate(
        self,
        composition: Mapping[str, NDArray[np.float64]],
        pressure_grid: PressureGrid,
    ) -> NDArray[np.float64]:
        """Return composition-weighted mean molecular weight by layer."""

        if not composition:
            raise RobertValidationError("composition must contain at least one species")

        numerator = np.zeros(pressure_grid.n_layers, dtype=float)
        total = np.zeros(pressure_grid.n_layers, dtype=float)
        for species, values in composition.items():
            name = _validate_species_name(species)
            if name not in self.molecular_masses:
                raise RobertValidationError(f"missing molecular mass for species: {name}")
            profile = _readonly_layer_array(
                values,
                f"{name} composition",
                pressure_grid.n_layers,
            )
            if np.any(profile < 0.0):
                raise RobertValidationError("composition values must be non-negative")
            numerator += profile * self.molecular_masses[name]
            total += profile

        if np.any(total <= 0.0):
            raise RobertValidationError("composition VMR sums must be positive")
        if self.normalization == "require" and np.any(np.abs(total - 1.0) > self.sum_tolerance):
            raise RobertValidationError("composition VMRs must sum to one to derive mean molecular weight")

        mean_molecular_weight = numerator / total
        mean_molecular_weight.setflags(write=False)
        return mean_molecular_weight


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
        object.__setattr__(self, "metadata", dict(self.metadata))

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
        object.__setattr__(self, "metadata", dict(self.metadata))

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
