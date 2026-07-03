"""Factories that turn model-setup dictionaries into ROBERT components."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robert_exoplanets.atmosphere import (
    AtmosphereBuilder,
    BackgroundGasMixture,
    ChemistryModel,
    CompositionMeanMolecularWeight,
    FixedMeanMolecularWeight,
    FreeChemistry,
    IsothermalTemperatureProfile,
    MadhusudhanSeager2009TemperatureProfile,
    MeanMolecularWeightModel,
    ParmentierGuillot2014TemperatureProfile,
    SplineTemperatureProfile,
    TabulatedTemperatureProfile,
    TemperatureProfile,
)
from robert_exoplanets.bodies import Planet
from robert_exoplanets.core import PressureGrid, RobertConfigError


@dataclass(frozen=True)
class AtmosphereModelSetup:
    """Atmosphere components built from a model-setup configuration."""

    pressure_grid: PressureGrid
    temperature_profile: TemperatureProfile
    chemistry_model: ChemistryModel
    mean_molecular_weight_model: MeanMolecularWeightModel | None = None
    default_parameters: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parameters = {str(name): float(value) for name, value in self.default_parameters.items()}
        if any(not np.isfinite(value) for value in parameters.values()):
            raise RobertConfigError("default_parameters must contain only finite numeric values")
        object.__setattr__(self, "default_parameters", parameters)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def build_atmosphere_builder(self) -> AtmosphereBuilder:
        """Return an `AtmosphereBuilder` for this setup."""

        return AtmosphereBuilder(
            pressure_grid=self.pressure_grid,
            temperature_profile=self.temperature_profile,
            chemistry_model=self.chemistry_model,
            mean_molecular_weight_model=self.mean_molecular_weight_model,
        )


def build_atmosphere_setup(
    config: Mapping[str, Any],
    *,
    planet: Planet | None = None,
) -> AtmosphereModelSetup:
    """Build atmosphere components from a ROBERT or HAT-P-32b-style config mapping."""

    pressure_grid = build_pressure_grid_from_config(config)
    temperature_profile, temperature_parameters = build_temperature_profile_from_config(
        config,
        planet=planet,
    )
    chemistry_model, chemistry_parameters = build_chemistry_model_from_config(config)
    mean_molecular_weight_model = build_mean_molecular_weight_model_from_config(
        config,
        chemistry_model=chemistry_model,
    )
    default_parameters = _merge_parameter_defaults(
        temperature_parameters,
        chemistry_parameters,
    )
    return AtmosphereModelSetup(
        pressure_grid=pressure_grid,
        temperature_profile=temperature_profile,
        chemistry_model=chemistry_model,
        mean_molecular_weight_model=mean_molecular_weight_model,
        default_parameters=default_parameters,
        metadata={"source": "mapping"},
    )


def build_pressure_grid_from_config(config: Mapping[str, Any]) -> PressureGrid:
    """Build a pressure grid from `pressure_grid` config."""

    block = _mapping(config.get("pressure_grid", config), "pressure_grid")
    unit = str(block.get("unit", "bar"))
    if "p_top_bar" in block or "p_bot_bar" in block:
        min_pressure = _required_float(block, "p_top_bar", "pressure_grid")
        max_pressure = _required_float(block, "p_bot_bar", "pressure_grid")
        unit = "bar"
    else:
        min_pressure = _required_float(block, "min_pressure", "pressure_grid")
        max_pressure = _required_float(block, "max_pressure", "pressure_grid")
    n_layers = int(_required_float(block, "n_layers", "pressure_grid"))
    name = block.get("name")
    return PressureGrid.logspace(
        min_pressure=min_pressure,
        max_pressure=max_pressure,
        n_layers=n_layers,
        unit=unit,
        name=None if name is None else str(name),
    )


def build_temperature_profile_from_config(
    config: Mapping[str, Any],
    *,
    planet: Planet | None = None,
) -> tuple[TemperatureProfile, dict[str, float]]:
    """Build a temperature profile and default parameters from config."""

    block = _mapping(config.get("temperature_profile", config), "temperature_profile")
    profile_type = str(block.get("type", "isothermal")).strip().lower()
    values = _numeric_mapping(block.get("values", {}), "temperature_profile.values")
    pressure_unit = str(block.get("pressure_unit", "bar"))

    if profile_type in {"isothermal", "iso"}:
        parameter_name = str(block.get("parameter_name", "T_iso"))
        profile = IsothermalTemperatureProfile(parameter_name=parameter_name)
        return profile, _defaults_for_required(profile.required_parameters(), values)

    if profile_type in {"tabulated", "table", "input"}:
        csv_path = block.get("path") or block.get("csv_path") or block.get("pt_profile_csv")
        if csv_path is None:
            emission = _mapping(config.get("emission", {}), "emission", allow_empty=True)
            csv_path = emission.get("pt_profile_csv")
            pressure_column = emission.get("pt_pressure_col", "pressure_bar")
            temperature_column = emission.get("pt_temperature_col", "temperature_K")
            pressure_unit = str(emission.get("pt_pressure_unit", pressure_unit))
        else:
            pressure_column = block.get("pressure_column", "pressure_bar")
            temperature_column = block.get("temperature_column", "temperature_K")
        if csv_path is None:
            raise RobertConfigError("tabulated temperature profile requires a CSV path")
        profile = TabulatedTemperatureProfile.from_csv(
            Path(str(csv_path)).expanduser(),
            pressure_column=str(pressure_column),
            temperature_column=str(temperature_column),
            pressure_unit=pressure_unit,
            name=str(block.get("name", "tabulated")),
            extrapolation=str(block.get("extrapolation", "raise")),
        )
        return profile, {}

    if profile_type in {"madhu", "madhusudhan", "madhusudhan_seager_2009", "ms09"}:
        profile = MadhusudhanSeager2009TemperatureProfile(
            pressure_unit=pressure_unit,
            reference_pressure=_optional_float(block.get("reference_pressure")),
        )
        return profile, _defaults_for_required(profile.required_parameters(), values)

    if profile_type in {"guillot14", "pg14", "parmentier_guillot_2014"}:
        gravity = _optional_float(block.get("gravity"))
        if gravity is None and planet is not None:
            gravity = planet.gravity_m_s2
        internal_temperature = _optional_float(values.get("T_int", block.get("T_int", 0.0)))
        profile = ParmentierGuillot2014TemperatureProfile(
            gravity=gravity,
            internal_temperature=internal_temperature,
        )
        return profile, _defaults_for_required(profile.required_parameters(), values)

    if profile_type in {"spline", "cubic_spline"}:
        knot_pressure = _required_array(block, "knot_pressure", "temperature_profile")
        knot_temperature = block.get("knot_temperature")
        parameter_names = block.get("parameter_names")
        if knot_temperature is not None:
            knot_temperature_array = _array(knot_temperature, "temperature_profile.knot_temperature")
            profile = SplineTemperatureProfile(
                knot_pressure=knot_pressure,
                knot_temperature=knot_temperature_array,
                pressure_unit=pressure_unit,
                extrapolation=str(block.get("extrapolation", "raise")),
            )
            return profile, {}
        profile = SplineTemperatureProfile(
            knot_pressure=knot_pressure,
            parameter_names=None if parameter_names is None else tuple(str(name) for name in parameter_names),
            pressure_unit=pressure_unit,
            extrapolation=str(block.get("extrapolation", "raise")),
        )
        return profile, _defaults_for_required(profile.required_parameters(), values)

    if profile_type == "guillot":
        raise RobertConfigError(
            "temperature_profile.type='guillot' is not implemented yet; use 'guillot14' or tabulated"
        )
    raise RobertConfigError(f"unsupported temperature profile type: {profile_type}")


def build_chemistry_model_from_config(
    config: Mapping[str, Any],
) -> tuple[ChemistryModel, dict[str, float]]:
    """Build a chemistry model and default parameters from config."""

    molecules = _mapping(config.get("molecules", {}), "molecules", allow_empty=True)
    free = _mapping(molecules.get("free", config.get("free_chemistry", {})), "molecules.free")
    active_species = tuple(str(name) for name in free.get("names", ()))
    if not active_species:
        raise RobertConfigError("molecules.free.names must contain at least one active species")

    values = _numeric_mapping(free.get("values", {}), "molecules.free.values")
    fixed = _numeric_mapping(free.get("fixed", {}), "molecules.free.fixed")
    parameter_names = free.get("parameter_names")
    parameter_mode = str(free.get("parameter_mode", "linear")).lower()
    background = _background_from_free_config(free)
    fill_background = bool(free.get("fill_background", background is not None))
    chemistry = FreeChemistry(
        active_species=active_species,
        background=background,
        fixed_mixing_ratios=fixed,
        parameter_names=None if parameter_names is None else dict(parameter_names),
        parameter_mode=parameter_mode,
        fill_background=fill_background,
        excess_policy=str(free.get("excess_policy", "raise")),
        metadata={
            "source_log_prior": str(bool(free.get("log", False))),
        },
    )
    defaults = _defaults_for_required(chemistry.required_parameters(), values)
    return chemistry, defaults


def build_mean_molecular_weight_model_from_config(
    config: Mapping[str, Any],
    *,
    chemistry_model: ChemistryModel,
) -> MeanMolecularWeightModel | None:
    """Build a mean-molecular-weight model from config."""

    block = config.get("mean_molecular_weight", config.get("mmw"))
    if block is None:
        if isinstance(chemistry_model, FreeChemistry) and chemistry_model.fill_background:
            return CompositionMeanMolecularWeight()
        return None
    if isinstance(block, (int, float)):
        return FixedMeanMolecularWeight(float(block))

    mapping = _mapping(block, "mean_molecular_weight")
    mmw_type = str(mapping.get("type", "composition")).strip().lower()
    if mmw_type in {"none", "fixed-builder"}:
        return None
    if mmw_type == "fixed":
        return FixedMeanMolecularWeight(_required_float(mapping, "value", "mean_molecular_weight"))
    if mmw_type in {"composition", "derived"}:
        masses = mapping.get("molecular_masses")
        if masses is None:
            return CompositionMeanMolecularWeight(
                normalization=str(mapping.get("normalization", "require")),
                sum_tolerance=float(mapping.get("sum_tolerance", 1.0e-8)),
            )
        return CompositionMeanMolecularWeight(
            molecular_masses=dict(masses),
            normalization=str(mapping.get("normalization", "require")),
            sum_tolerance=float(mapping.get("sum_tolerance", 1.0e-8)),
        )
    raise RobertConfigError(f"unsupported mean_molecular_weight type: {mmw_type}")


def _background_from_free_config(free: Mapping[str, Any]) -> BackgroundGasMixture | None:
    inactive = free.get("inactive", {"names": ["H2", "He"]})
    if inactive is False or inactive is None:
        return None
    if isinstance(inactive, Mapping):
        names = tuple(str(name) for name in inactive.get("names", ()))
        fractions = inactive.get("fractions")
    else:
        names = tuple(str(name) for name in inactive)
        fractions = None
    if not names:
        return None
    if fractions is not None:
        return BackgroundGasMixture(dict(fractions))
    if set(names) == {"H2", "He"}:
        return BackgroundGasMixture.hydrogen_helium()
    equal_fraction = 1.0 / len(names)
    return BackgroundGasMixture({name: equal_fraction for name in names})


def _merge_parameter_defaults(*mappings: Mapping[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for mapping in mappings:
        for name, value in mapping.items():
            if name in merged and merged[name] != value:
                raise RobertConfigError(f"conflicting default parameter value for: {name}")
            merged[str(name)] = float(value)
    return merged


def _defaults_for_required(
    required_parameters: tuple[str, ...],
    values: Mapping[str, float],
) -> dict[str, float]:
    return {
        name: float(values[name])
        for name in required_parameters
        if name in values
    }


def _mapping(
    value: Any,
    name: str,
    *,
    allow_empty: bool = False,
) -> Mapping[str, Any]:
    if value is None and allow_empty:
        return {}
    if not isinstance(value, Mapping):
        raise RobertConfigError(f"{name} must be a mapping")
    return value


def _numeric_mapping(value: Any, name: str) -> dict[str, float]:
    mapping = _mapping(value, name, allow_empty=True)
    output: dict[str, float] = {}
    for key, raw_value in mapping.items():
        try:
            number = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise RobertConfigError(f"{name}.{key} must be numeric") from exc
        if not np.isfinite(number):
            raise RobertConfigError(f"{name}.{key} must be finite")
        output[str(key)] = number
    return output


def _required_float(mapping: Mapping[str, Any], key: str, context: str) -> float:
    if key not in mapping:
        raise RobertConfigError(f"{context} requires {key}")
    return _float(mapping[key], f"{context}.{key}")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _float(value, "optional float")


def _float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RobertConfigError(f"{name} must be numeric") from exc
    if not np.isfinite(number):
        raise RobertConfigError(f"{name} must be finite")
    return number


def _required_array(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
) -> np.ndarray:
    if key not in mapping:
        raise RobertConfigError(f"{context} requires {key}")
    return _array(mapping[key], f"{context}.{key}")


def _array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise RobertConfigError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise RobertConfigError(f"{name} must contain only finite values")
    return array
