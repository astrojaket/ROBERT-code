"""Disk-inhomogeneous emission model composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from robert_exoplanets.core import RobertValidationError, Spectrum

EmissionEvaluator = Callable[[Mapping[str, float]], Spectrum]
MultiDatasetEmissionEvaluator = Callable[
    [Mapping[str, float]], Mapping[str, Spectrum]
]


def _parameter(parameters: Mapping[str, float], name: str) -> float:
    if name not in parameters:
        raise RobertValidationError(f"missing inhomogeneous-disk parameter {name!r}")
    value = float(parameters[name])
    if not np.isfinite(value):
        raise RobertValidationError(f"inhomogeneous-disk parameter {name!r} must be finite")
    return value


def _validate_fraction(value: float, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise RobertValidationError(f"{name} must lie in [0, 1]")
    return value


def _validate_compatible(left: Spectrum, right: Spectrum) -> None:
    if left.spectral_grid.unit != right.spectral_grid.unit or not np.array_equal(
        left.spectral_grid.values,
        right.spectral_grid.values,
    ):
        raise RobertValidationError("regional spectra must share one spectral grid")
    if left.unit != right.unit or left.observable != right.observable:
        raise RobertValidationError("regional spectra must share units and observable")


def _mixed_spectrum(
    hot: Spectrum,
    cold: Spectrum,
    hot_fraction: float,
    *,
    hot_fraction_parameter: str,
) -> Spectrum:
    _validate_compatible(hot, cold)
    return Spectrum(
        spectral_grid=hot.spectral_grid,
        values=hot_fraction * hot.values + (1.0 - hot_fraction) * cold.values,
        unit=hot.unit,
        observable=hot.observable,
        metadata={
            "disk_model": "two_region_areal_mixture",
            "hot_fraction_parameter": hot_fraction_parameter,
            "hot_area_fraction": f"{hot_fraction:.17g}",
            "source_equation": "Schlawin_et_al_2024_equation_1",
        },
    )


def _diluted_spectrum(
    spectrum: Spectrum,
    dilution: float,
    *,
    dilution_parameter: str,
) -> Spectrum:
    return Spectrum(
        spectral_grid=spectrum.spectral_grid,
        values=dilution * spectrum.values,
        unit=spectrum.unit,
        observable=spectrum.observable,
        metadata={
            **dict(spectrum.metadata),
            "disk_model": "diluted_single_region",
            "dilution_parameter": dilution_parameter,
            "dayside_dilution": f"{dilution:.17g}",
            "dilution_definition": "fractional_projected_dayside_emitting_area",
        },
    )


@dataclass(frozen=True)
class TwoRegionEmissionModel:
    """Area-weight two independent dayside emission columns.

    This implements Schlawin et al. (2024), Eq. 1:
    ``F = x_hot F_hot + (1 - x_hot) F_cold``. Each regional evaluator may
    carry its own temperature profile and cloud parameters while receiving the
    same full retrieval parameter mapping.
    """

    hot_model: EmissionEvaluator
    cold_model: EmissionEvaluator
    hot_fraction_parameter: str = "hot_area_fraction"

    def __post_init__(self) -> None:
        if not self.hot_fraction_parameter:
            raise RobertValidationError("hot_fraction_parameter must not be empty")

    def __call__(self, parameters: Mapping[str, float]) -> Spectrum:
        hot_fraction = _validate_fraction(
            _parameter(parameters, self.hot_fraction_parameter),
            self.hot_fraction_parameter,
        )
        hot = self.hot_model(parameters)
        cold = self.cold_model(parameters)
        return _mixed_spectrum(
            hot,
            cold,
            hot_fraction,
            hot_fraction_parameter=self.hot_fraction_parameter,
        )


@dataclass(frozen=True)
class DilutedEmissionModel:
    """Scale one emitting column by its fractional projected dayside area.

    Dilution is applied to the final planet observable, not to temperature,
    opacity, radius, or stellar flux. It is the one-emitting-region limit of an
    inhomogeneous disk when the complementary region contributes negligibly.
    """

    emission_model: EmissionEvaluator
    dilution_parameter: str = "dayside_dilution"

    def __post_init__(self) -> None:
        if not self.dilution_parameter:
            raise RobertValidationError("dilution_parameter must not be empty")

    def __call__(self, parameters: Mapping[str, float]) -> Spectrum:
        dilution = _validate_fraction(
            _parameter(parameters, self.dilution_parameter),
            self.dilution_parameter,
        )
        spectrum = self.emission_model(parameters)
        return _diluted_spectrum(
            spectrum,
            dilution,
            dilution_parameter=self.dilution_parameter,
        )


@dataclass(frozen=True)
class MultiDatasetTwoRegionEmissionModel:
    """Area-weight two regional models that each return named datasets."""

    hot_model: MultiDatasetEmissionEvaluator
    cold_model: MultiDatasetEmissionEvaluator
    hot_fraction_parameter: str = "hot_area_fraction"

    def __post_init__(self) -> None:
        if not self.hot_fraction_parameter:
            raise RobertValidationError("hot_fraction_parameter must not be empty")

    def __call__(self, parameters: Mapping[str, float]) -> Mapping[str, Spectrum]:
        hot_fraction = _validate_fraction(
            _parameter(parameters, self.hot_fraction_parameter),
            self.hot_fraction_parameter,
        )
        hot = dict(self.hot_model(parameters))
        cold = dict(self.cold_model(parameters))
        if not hot or set(hot) != set(cold):
            raise RobertValidationError(
                "regional multi-dataset spectra must have matching non-empty names"
            )
        return {
            name: _mixed_spectrum(
                hot[name],
                cold[name],
                hot_fraction,
                hot_fraction_parameter=self.hot_fraction_parameter,
            )
            for name in hot
        }


@dataclass(frozen=True)
class MultiDatasetDilutedEmissionModel:
    """Dilute every named spectrum from one multi-dataset emission model."""

    emission_model: MultiDatasetEmissionEvaluator
    dilution_parameter: str = "dayside_dilution"

    def __post_init__(self) -> None:
        if not self.dilution_parameter:
            raise RobertValidationError("dilution_parameter must not be empty")

    def __call__(self, parameters: Mapping[str, float]) -> Mapping[str, Spectrum]:
        dilution = _validate_fraction(
            _parameter(parameters, self.dilution_parameter),
            self.dilution_parameter,
        )
        spectra = dict(self.emission_model(parameters))
        if not spectra:
            raise RobertValidationError("multi-dataset emission model returned no spectra")
        return {
            name: _diluted_spectrum(
                spectrum,
                dilution,
                dilution_parameter=self.dilution_parameter,
            )
            for name, spectrum in spectra.items()
        }


@dataclass(frozen=True)
class DiskEmissionModelConfig:
    """Planet-independent selection of projected-dayside model geometry."""

    mode: str = "one_region"
    hot_fraction_parameter: str = "hot_area_fraction"
    dilution_parameter: str = "dayside_dilution"

    def __post_init__(self) -> None:
        mode = self.mode.strip().lower().replace("-", "_")
        aliases = {
            "one_region": "one_region",
            "homogeneous": "one_region",
            "diluted_one_region": "diluted_one_region",
            "diluted": "diluted_one_region",
            "two_region": "two_region",
            "2tp": "two_region",
        }
        if mode not in aliases:
            raise RobertValidationError(
                "disk emission mode must be 'one_region', 'diluted_one_region', or 'two_region'"
            )
        if not self.hot_fraction_parameter or not self.dilution_parameter:
            raise RobertValidationError("disk-model parameter names must not be empty")
        object.__setattr__(self, "mode", aliases[mode])


def build_disk_emission_model(
    config: DiskEmissionModelConfig,
    primary_model: EmissionEvaluator,
    *,
    secondary_model: EmissionEvaluator | None = None,
) -> EmissionEvaluator:
    """Compose regional ROBERT forward models using the selected disk geometry.

    Regional models remain responsible for atmosphere, chemistry, opacity,
    clouds, and RT. Consequently the same selector works for clear or cloudy
    columns and for any planet represented by ROBERT's domain objects.
    """

    if config.mode == "one_region":
        if secondary_model is not None:
            raise RobertValidationError("one-region mode does not accept a secondary model")
        return primary_model
    if config.mode == "diluted_one_region":
        if secondary_model is not None:
            raise RobertValidationError("diluted one-region mode does not accept a secondary model")
        return DilutedEmissionModel(
            primary_model,
            dilution_parameter=config.dilution_parameter,
        )
    if secondary_model is None:
        raise RobertValidationError("two-region mode requires a secondary regional model")
    return TwoRegionEmissionModel(
        primary_model,
        secondary_model,
        hot_fraction_parameter=config.hot_fraction_parameter,
    )


__all__ = [
    "DilutedEmissionModel",
    "DiskEmissionModelConfig",
    "EmissionEvaluator",
    "MultiDatasetDilutedEmissionModel",
    "MultiDatasetEmissionEvaluator",
    "MultiDatasetTwoRegionEmissionModel",
    "TwoRegionEmissionModel",
    "build_disk_emission_model",
]
