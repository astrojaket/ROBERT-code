"""Physical cloud profiles for posterior diagnostics.

The helpers in this module inspect one already selected forward-model leaf.
They do not evaluate a forward model or run radiative transfer.  Returned
profiles are one value per atmospheric layer and can therefore be plotted on
the same pressure grid as temperature and composition diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.core import (
    PressureGrid,
    RobertError,
    RobertValidationError,
    SpectralGrid,
)
from robert_exoplanets.forward.clouds import (
    ParameterizedDeckHazeCloudModel,
    ParameterizedMieCloudModel,
    pressure_slab_layer_fractions,
)
from robert_exoplanets.opacity import pressure_values_in_unit
from robert_exoplanets.rt import (
    grey_cloud_deck,
    power_law_haze_from_mass_extinction,
)


class CloudProfile(TypedDict):
    """One named layer profile returned by :func:`cloud_profiles`."""

    values: NDArray[np.float64]
    unit: str


CloudProfileMapping = dict[str, CloudProfile]


class CloudProfileUnavailableError(RobertError):
    """Raised when a cloud model has no physical profile diagnostic."""


@dataclass(frozen=True)
class _HazeGrid:
    """Small input view required by the existing hydrostatic haze equation."""

    pressure_grid: PressureGrid
    spectral_grid: SpectralGrid
    layer_pressure_thickness_pa: NDArray[np.float64]
    gravity_m_s2: NDArray[np.float64]


_MISSING = object()


def cloud_profiles(
    model: object,
    atmosphere: AtmosphereState,
    parameters: Mapping[str, float],
) -> CloudProfileMapping:
    """Return physical cloud profiles for one forward-model leaf.

    The mapping is empty for a cloud-free model.  A shared Mie cloud returns
    ``condensate_mass_fraction`` and ``particle_radius``.  A deck plus haze
    returns layer extinction optical depths at the model's haze reference
    wavelength; that wavelength is part of each profile name.  Unsupported
    custom cloud implementations raise :class:`CloudProfileUnavailableError`
    so a collector can record an explicit unavailable diagnostic.
    """

    cloud_model, owner = _resolve_cloud_model(model)
    if cloud_model is None:
        return {}

    if isinstance(cloud_model, ParameterizedMieCloudModel):
        return _mie_profiles(cloud_model, atmosphere, parameters)
    if isinstance(cloud_model, ParameterizedDeckHazeCloudModel):
        return _deck_haze_profiles(cloud_model, owner, atmosphere, parameters)

    raise CloudProfileUnavailableError(
        "cloud profiles are unavailable for unsupported cloud model "
        f"{type(cloud_model).__qualname__}; no physical profile adapter is defined"
    )


def _resolve_cloud_model(model: object) -> tuple[object | None, object]:
    """Return a leaf cloud model and its forward-model owner."""

    if model is None:
        return None, model
    if isinstance(model, (ParameterizedMieCloudModel, ParameterizedDeckHazeCloudModel)):
        return model, model

    cloud_model = getattr(model, "cloud_model", _MISSING)
    if cloud_model is not _MISSING:
        if cloud_model is None and hasattr(model, "cloud"):
            raise CloudProfileUnavailableError(
                "cloud profiles are unavailable for legacy cloud configuration "
                f"{type(model).__qualname__}; use a modern shared cloud model"
            )
        return cast(object | None, cloud_model), model

    if hasattr(model, "cloud"):
        raise CloudProfileUnavailableError(
            "cloud profiles are unavailable for legacy cloud configuration "
            f"{type(model).__qualname__}; use a modern shared cloud model"
        )

    # These classes are imported lazily.  ``diagnostics`` is imported before
    # ``forward`` by the top-level package, so eager imports would create an
    # avoidable package-import cycle.
    from robert_exoplanets.forward.emission import EmissionForwardModel
    from robert_exoplanets.forward.transmission import (
        ParameterizedTransmissionForwardModel,
    )

    if isinstance(
        model,
        (EmissionForwardModel, ParameterizedTransmissionForwardModel),
    ):
        return None, model

    raise CloudProfileUnavailableError(
        "cloud profiles are unavailable for unsupported forward model "
        f"{type(model).__qualname__}; the collector must select a supported leaf"
    )


def _mie_profiles(
    cloud_model: Any,
    atmosphere: AtmosphereState,
    parameters: Mapping[str, float],
) -> CloudProfileMapping:
    """Return layer averaged condensate mass and effective radius profiles."""

    mass_fraction_parameter = str(cloud_model.log10_condensate_mass_fraction_parameter)
    mass_fraction_value = _ten_power(
        parameters,
        mass_fraction_parameter,
        "condensate mass fraction",
    )
    top_pressure = _optional_ten_power(
        parameters,
        getattr(cloud_model, "log10_cloud_top_pressure_bar_parameter", None),
        "cloud top pressure",
    )
    base_pressure = _optional_ten_power(
        parameters,
        getattr(cloud_model, "log10_cloud_base_pressure_bar_parameter", None),
        "cloud base pressure",
    )
    fractions = pressure_slab_layer_fractions(
        atmosphere.pressure_grid,
        top_pressure_bar=top_pressure,
        base_pressure_bar=base_pressure,
    )
    mass_fraction = _readonly_profile(fractions * mass_fraction_value)

    radius = _ten_power(
        parameters,
        str(cloud_model.log10_effective_radius_micron_parameter),
        "effective particle radius",
    )
    radius_profile = _readonly_profile(
        np.full(atmosphere.n_layers, radius, dtype=np.float64)
    )
    return {
        "condensate_mass_fraction": {
            "values": mass_fraction,
            "unit": "kg/kg",
        },
        "particle_radius": {
            "values": radius_profile,
            "unit": "micron",
        },
    }


def _deck_haze_profiles(
    cloud_model: ParameterizedDeckHazeCloudModel,
    owner: object,
    atmosphere: AtmosphereState,
    parameters: Mapping[str, float],
) -> CloudProfileMapping:
    """Evaluate deck and haze layer optical depths without running full RT."""

    cloud_top_pressure = _ten_power(
        parameters,
        cloud_model.log10_cloud_top_pressure_bar_parameter,
        "cloud top pressure",
    )
    deck_optical_depth = _ten_power(
        parameters,
        cloud_model.log10_cloud_optical_depth_parameter,
        "cloud optical depth",
    )
    haze_mass_extinction = _ten_power(
        parameters,
        cloud_model.log10_haze_mass_extinction_parameter,
        "haze mass extinction",
    )
    haze_slope = _parameter_value(parameters, cloud_model.haze_slope_parameter)
    reference_wavelength = float(cloud_model.haze_reference_wavelength_micron)

    # The deck is grey, but evaluate it at the same physical reference point
    # used for the haze so both diagnostics carry one clear wavelength label.
    reference_grid = SpectralGrid.from_array(
        [reference_wavelength],
        unit="micron",
        role="diagnostic",
    )
    deck = grey_cloud_deck(
        atmosphere.pressure_grid,
        reference_grid,
        cloud_top_pressure=cloud_top_pressure,
        cloud_top_pressure_unit="bar",
        optical_depth=deck_optical_depth,
        single_scattering_albedo=cloud_model.deck_single_scattering_albedo,
        asymmetry_factor=cloud_model.deck_asymmetry_factor,
    )

    gravity = _gravity_profile(owner, atmosphere, parameters)
    pressure_edges_pa = pressure_values_in_unit(
        atmosphere.pressure_grid.edges,
        atmosphere.pressure_grid.unit,
        "pa",
    )
    haze_input = _HazeGrid(
        pressure_grid=atmosphere.pressure_grid,
        spectral_grid=reference_grid,
        layer_pressure_thickness_pa=np.abs(np.diff(pressure_edges_pa)),
        gravity_m_s2=gravity,
    )
    # ``power_law_haze_from_mass_extinction`` only consumes this small grid
    # view.  No gas opacity or RT solver is evaluated for this diagnostic.
    haze = power_law_haze_from_mass_extinction(
        cast(Any, haze_input),
        mass_extinction_at_reference_cm2_g=haze_mass_extinction,
        reference_wavelength_micron=reference_wavelength,
        slope=haze_slope,
        single_scattering_albedo=cloud_model.haze_single_scattering_albedo,
        asymmetry_factor=cloud_model.haze_asymmetry_factor,
    )

    suffix = _wavelength_suffix(reference_wavelength)
    return {
        f"deck_extinction_optical_depth_at_{suffix}_micron": {
            "values": _readonly_profile(deck.extinction_tau[:, 0]),
            "unit": deck.unit,
        },
        f"haze_extinction_optical_depth_at_{suffix}_micron": {
            "values": _readonly_profile(haze.extinction_tau[:, 0]),
            "unit": haze.unit,
        },
    }


def _gravity_profile(
    owner: object,
    atmosphere: AtmosphereState,
    parameters: Mapping[str, float],
) -> NDArray[np.float64]:
    """Return the gravity used by the owning leaf for hydrostatic cloud tau."""

    value = getattr(owner, "gravity_m_s2", _MISSING)
    if value is not _MISSING:
        return _positive_profile(value, atmosphere.n_layers)

    # Transmission computes gravity through its hydrostatic path geometry.
    # Reuse that inexpensive geometry calculation so haze tau follows the
    # exact same layer gravity convention as the forward model.
    planet = getattr(owner, "planet", _MISSING)
    config = getattr(owner, "config", _MISSING)
    reference_gravity_method = getattr(owner, "_reference_gravity", None)
    if (
        planet is _MISSING
        or config is _MISSING
        or reference_gravity_method is None
        or getattr(planet, "radius_m", None) is None
    ):
        raise CloudProfileUnavailableError(
            "haze cloud profiles require the owning forward model's physical "
            "gravity profile"
        )

    radius_scale = 1.0
    radius_parameter = getattr(config, "radius_scale_parameter", None)
    if radius_parameter is not None:
        radius_scale = _parameter_value(parameters, radius_parameter)
        if radius_scale <= 0.0:
            raise RobertValidationError("radius scale must be positive")
    reference_radius = float(planet.radius_m) * radius_scale
    reference_gravity = float(reference_gravity_method(reference_radius))

    from robert_exoplanets.rt import (
        hydrostatic_path_geometry,
        inverse_square_hydrostatic_path_geometry,
    )

    if getattr(config, "gravity_model", "inverse_square") == "inverse_square":
        geometry = inverse_square_hydrostatic_path_geometry(
            atmosphere,
            reference_radius_m=reference_radius,
            reference_pressure=float(config.reference_pressure_bar),
            reference_gravity_m_s2=reference_gravity,
            reference_pressure_unit="bar",
        )
    else:
        geometry = hydrostatic_path_geometry(
            atmosphere,
            gravity_m_s2=reference_gravity,
            reference_radius_m=reference_radius,
            reference_pressure=float(config.reference_pressure_bar),
            reference_pressure_unit="bar",
        )
    return _positive_profile(geometry.gravity_m_s2, atmosphere.n_layers)


def _parameter_value(parameters: Mapping[str, float], name: str) -> float:
    if name not in parameters:
        raise RobertValidationError(f"cloud parameter is missing: {name}")
    value = float(parameters[name])
    if not np.isfinite(value):
        raise RobertValidationError(f"cloud parameter {name!r} must be finite")
    return value


def _ten_power(
    parameters: Mapping[str, float],
    name: str,
    description: str,
) -> float:
    value = _parameter_value(parameters, name)
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.power(10.0, value))
    if not np.isfinite(result) or result <= 0.0:
        raise RobertValidationError(f"{description} parameter overflowed")
    return result


def _optional_ten_power(
    parameters: Mapping[str, float],
    name: str | None,
    description: str,
) -> float | None:
    return None if name is None else _ten_power(parameters, name, description)


def _positive_profile(values: Any, n_layers: int) -> NDArray[np.float64]:
    profile = np.asarray(values, dtype=np.float64)
    if profile.ndim == 0:
        profile = np.full(n_layers, float(profile), dtype=np.float64)
    if profile.shape != (n_layers,):
        raise CloudProfileUnavailableError(
            "cloud gravity profile does not match the atmosphere layer count"
        )
    if not np.all(np.isfinite(profile)) or np.any(profile <= 0.0):
        raise CloudProfileUnavailableError(
            "cloud gravity profile must be finite and positive"
        )
    return _readonly_profile(profile)


def _readonly_profile(values: Any) -> NDArray[np.float64]:
    profile = np.array(values, dtype=np.float64, copy=True)
    if profile.ndim != 1 or not np.all(np.isfinite(profile)):
        raise CloudProfileUnavailableError("cloud profile values must be finite layers")
    profile.setflags(write=False)
    return profile


def _wavelength_suffix(value: float) -> str:
    return f"{value:.12g}".replace("+", "")


__all__ = [
    "CloudProfile",
    "CloudProfileMapping",
    "CloudProfileUnavailableError",
    "cloud_profiles",
]
