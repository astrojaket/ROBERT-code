"""Typed posterior model products for retrieval diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.core import RobertValidationError, Spectrum
from robert_exoplanets.core._immutability import immutable_mapping

FloatArray = NDArray[np.float64]
POSTERIOR_QUANTILE_PROBABILITIES: tuple[float, ...] = (
    0.02275013194817921,
    0.15865525393145707,
    0.5,
    0.8413447460685429,
    0.9772498680518208,
)
POSTERIOR_QUANTILE_LABELS: tuple[str, ...] = (
    "lower_2sigma",
    "lower_1sigma",
    "median",
    "upper_1sigma",
    "upper_2sigma",
)


class PosteriorProductUnavailableError(RobertValidationError):
    """Raised when a retrieval cannot provide a posterior diagnostic product."""


@dataclass(frozen=True)
class PosteriorSpectrumSummary:
    """Five quantiles on one model grid, including configured Gaussian offset."""

    wavelength: FloatArray
    wavelength_unit: str
    values: FloatArray
    unit: str
    observable: str
    bin_edges: FloatArray | None = None
    mask: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        wavelength = _array(self.wavelength, "posterior wavelength")
        values = _summary_quantiles(
            self.values, wavelength.size, "posterior spectrum values"
        )
        edges = (
            None
            if self.bin_edges is None
            else _array(self.bin_edges, "posterior bin edges")
        )
        if edges is not None and edges.size != wavelength.size + 1:
            raise RobertValidationError(
                "posterior bin edges must have len(wavelength) + 1 values"
            )
        mask = None if self.mask is None else np.array(self.mask, dtype=bool, copy=True)
        if mask is not None:
            if mask.ndim != 1 or mask.shape != wavelength.shape:
                raise RobertValidationError(
                    "posterior mask must match wavelength shape"
                )
            mask.setflags(write=False)
        for value, name in (
            (self.wavelength_unit, "wavelength unit"),
            (self.unit, "spectrum unit"),
            (self.observable, "spectrum observable"),
        ):
            _text(value, f"posterior {name}")
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "bin_edges", edges)
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True)
class PosteriorProfileSummary:
    """Five quantiles for one temperature, VMR, or cloud profile."""

    coordinate: FloatArray
    coordinate_unit: str
    values: FloatArray
    value_unit: str

    def __post_init__(self) -> None:
        coordinate = _array(self.coordinate, "posterior profile coordinate")
        values = _summary_quantiles(
            self.values, coordinate.size, "posterior profile values"
        )
        _text(self.coordinate_unit, "posterior coordinate unit")
        _text(self.value_unit, "posterior value unit")
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class PosteriorModelCollection:
    """Products calculated from one shared set of selected posterior vectors."""

    draw_vectors: FloatArray
    spectra: Mapping[str, PosteriorSpectrumSummary]
    temperature_profiles: Mapping[str, PosteriorProfileSummary]
    vmr_profiles: Mapping[str, Mapping[str, PosteriorProfileSummary]]
    cloud_profiles: Mapping[str, Mapping[str, PosteriorProfileSummary]]
    unsupported_cloud_profiles: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        draws = np.array(self.draw_vectors, dtype=float, copy=True)
        if draws.ndim != 2 or not draws.shape[0] or not draws.shape[1]:
            raise RobertValidationError(
                "posterior draw_vectors must be a non-empty 2D array"
            )
        if not np.all(np.isfinite(draws)):
            raise RobertValidationError("posterior draw_vectors must be finite")
        draws.setflags(write=False)
        object.__setattr__(self, "draw_vectors", draws)
        object.__setattr__(self, "spectra", immutable_mapping(self.spectra))
        object.__setattr__(
            self, "temperature_profiles", immutable_mapping(self.temperature_profiles)
        )
        object.__setattr__(self, "vmr_profiles", _nested(self.vmr_profiles))
        object.__setattr__(self, "cloud_profiles", _nested(self.cloud_profiles))
        object.__setattr__(
            self,
            "unsupported_cloud_profiles",
            _nested_reasons(self.unsupported_cloud_profiles),
        )


@dataclass(frozen=True)
class _Region:
    name: str
    model: object
    builder: object


def collect_posterior_models(
    problem: object, draw_vectors: ArrayLike
) -> PosteriorModelCollection:
    """Evaluate selected vectors once and return common five-quantile products.

    The caller normally supplies one hundred vectors after weighted posterior
    selection.  No resampling or weighting happens here.
    """

    draws = _draws(draw_vectors)
    mapper = getattr(problem, "parameter_mapping", None)
    if not callable(mapper):
        raise RobertValidationError("posterior problem must expose parameter_mapping")
    mapper = cast(Callable[[ArrayLike], Mapping[str, float]], mapper)
    observations = _observations(problem)
    expected_names = tuple(observations)
    regions = _named_regions(getattr(problem, "forward_model", None))
    spectrum_samples: dict[str, list[FloatArray]] = {}
    spectrum_reference: dict[str, Spectrum] = {}
    spectrum_masks: dict[str, NDArray[np.bool_] | None] = {}
    states: dict[str, object] = {}
    temperatures: dict[str, list[FloatArray]] = {r.name: [] for r in regions}
    vmrs: dict[str, dict[str, list[FloatArray]]] = {r.name: {} for r in regions}
    vmr_units: dict[str, str] = {}
    clouds: dict[str, dict[str, list[FloatArray]]] = {r.name: {} for r in regions}
    cloud_units: dict[tuple[str, str], str] = {}
    unsupported: dict[str, dict[str, str]] = {}

    for draw in draws:
        parameters = _parameters(mapper, draw)
        spectra = _spectra(_evaluate(problem, parameters))
        _check_names(spectra, expected_names, spectrum_samples)
        for name, spectrum in spectra.items():
            reference = spectrum_reference.get(name)
            if reference is None:
                spectrum_reference[name] = spectrum
                spectrum_samples[name] = []
                spectrum_masks[name] = _mask(
                    observations.get(name), spectrum.values.size
                )
            else:
                _same_grid(reference, spectrum, name)
            values = np.array(spectrum.values, dtype=float, copy=True)
            offset = _offset_parameter(problem, name)
            if offset and offset in parameters:
                values += float(parameters[offset])
            values.setflags(write=False)
            spectrum_samples[name].append(values)
        for region in regions:
            state = _build(region.builder, parameters, region.name)
            _state_values(region.name, state, states, temperatures, vmrs, vmr_units)
            _cloud_values(region, state, parameters, clouds, cloud_units, unsupported)

    _lengths(spectrum_samples, draws.shape[0], "spectrum")
    _lengths(temperatures, draws.shape[0], "temperature")
    for region_name, profiles in vmrs.items():
        _lengths(profiles, draws.shape[0], f"VMR {region_name!r}")
    for region_name, profiles in clouds.items():
        _lengths(profiles, draws.shape[0], f"cloud {region_name!r}")
    spectra_result = {
        name: PosteriorSpectrumSummary(
            wavelength=reference.spectral_grid.values,
            wavelength_unit=reference.spectral_grid.unit,
            values=_quantiles(spectrum_samples[name]),
            unit=reference.unit,
            observable=reference.observable,
            bin_edges=reference.spectral_grid.bin_edges,
            mask=spectrum_masks[name],
        )
        for name, reference in spectrum_reference.items()
    }
    temperature_result = {
        name: _profile(
            state, temperatures[name], str(getattr(state, "temperature_unit", "K"))
        )
        for name, state in states.items()
    }
    vmr_result = _profile_maps(vmrs, cast(Mapping[Any, str], vmr_units), states)
    cloud_result = _profile_maps(
        clouds, cast(Mapping[Any, str], cloud_units), states, cloud=True
    )
    return PosteriorModelCollection(
        draw_vectors=draws,
        spectra=spectra_result,
        temperature_profiles=temperature_result,
        vmr_profiles=vmr_result,
        cloud_profiles=cloud_result,
        unsupported_cloud_profiles=unsupported,
    )


def _draws(values: ArrayLike) -> FloatArray:
    draws = np.array(values, dtype=float, copy=True)
    if draws.ndim == 1:
        draws = draws.reshape(1, -1)
    if (
        draws.ndim != 2
        or not draws.shape[0]
        or not draws.shape[1]
        or not np.all(np.isfinite(draws))
    ):
        raise RobertValidationError(
            "posterior draw_vectors must be finite and non-empty 2D"
        )
    draws.setflags(write=False)
    return draws


def _parameters(
    mapper: Callable[[ArrayLike], Mapping[str, float]], draw: FloatArray
) -> Mapping[str, float]:
    values = mapper(draw)
    if not isinstance(values, Mapping):
        raise RobertValidationError("posterior parameter_mapping must return a mapping")
    try:
        result = {str(name): float(value) for name, value in values.items()}
    except (TypeError, ValueError, OverflowError) as error:
        raise RobertValidationError(
            "posterior parameter_mapping must return numeric values"
        ) from error
    if not all(np.isfinite(value) for value in result.values()):
        raise RobertValidationError(
            "posterior parameter_mapping must return finite values"
        )
    return immutable_mapping(result)


def _evaluate(problem: object, parameters: Mapping[str, float]) -> object:
    for method_name in ("model_spectra", "model_spectrum", "predict"):
        method = getattr(problem, method_name, None)
        if callable(method):
            return method(parameters)
    raise PosteriorProductUnavailableError(
        "posterior problem must expose model_spectra, model_spectrum, or predict"
    )


def _spectra(prediction: object) -> Mapping[str, Spectrum]:
    if isinstance(prediction, Spectrum):
        return {"primary": prediction}
    values = (
        prediction
        if isinstance(prediction, Mapping)
        else getattr(prediction, "spectra", None)
    )
    if not isinstance(values, Mapping) or not values:
        raise PosteriorProductUnavailableError(
            "posterior prediction must be a Spectrum or named Spectrum mapping"
        )
    output: dict[str, Spectrum] = {}
    for key, value in values.items():
        name = str(key).strip()
        if not name or name in output or not isinstance(value, Spectrum):
            raise PosteriorProductUnavailableError(
                "posterior prediction must contain named Spectrum values"
            )
        output[name] = value
    return output


def _check_names(
    spectra: Mapping[str, Spectrum],
    expected: tuple[str, ...],
    previous: Mapping[str, object],
) -> None:
    if previous and set(spectra) != set(previous):
        raise RobertValidationError(
            "posterior prediction dataset names changed across draws"
        )
    if expected and set(spectra) != set(expected):
        raise RobertValidationError(
            "posterior prediction names must match observations"
        )


def _same_grid(reference: Spectrum, current: Spectrum, name: str) -> None:
    left, right = reference.spectral_grid, current.spectral_grid
    left_edges, right_edges = left.bin_edges, right.bin_edges
    if (
        left.unit != right.unit
        or not np.array_equal(left.values, right.values)
        or (left_edges is None) != (right_edges is None)
        or (
            left_edges is not None
            and right_edges is not None
            and not np.array_equal(left_edges, right_edges)
        )
        or reference.unit != current.unit
        or reference.observable != current.observable
    ):
        raise RobertValidationError(
            f"posterior spectrum {name!r} changed grid or units"
        )


def _observations(problem: object) -> Mapping[str, object]:
    collection = getattr(problem, "observations", None)
    datasets = getattr(collection, "datasets", None)
    if datasets is not None:
        return {str(item.name): item.observation for item in datasets}
    observation = getattr(problem, "observation", None)
    return {} if observation is None else {"primary": observation}


def _mask(observation: object | None, size: int) -> NDArray[np.bool_] | None:
    value = None if observation is None else getattr(observation, "mask", None)
    if value is None:
        return None
    mask = np.array(value, dtype=bool, copy=True)
    if mask.shape != (size,):
        raise RobertValidationError("observation mask must match model grid")
    mask.setflags(write=False)
    return mask


def _offset_parameter(problem: object, name: str) -> str | None:
    collection = getattr(problem, "observations", None)
    datasets = getattr(collection, "datasets", None)
    if datasets is not None:
        for item in datasets:
            if str(item.name) == name:
                value = getattr(item, "offset_parameter", None)
                return None if value is None else str(value)
    value = getattr(getattr(problem, "likelihood", None), "offset_parameter", None)
    return None if value is None else str(value)


def _named_regions(model: object, *, prefix: str = "primary") -> tuple[_Region, ...]:
    if model is None:
        return ()
    hot, cold = getattr(model, "hot_model", None), getattr(model, "cold_model", None)
    if hot is not None and cold is not None:
        return _named_regions(hot, prefix="hot") + _named_regions(cold, prefix="cold")
    nested = getattr(model, "emission_model", None)
    if nested is not None:
        return _named_regions(nested, prefix=prefix)
    builder = getattr(model, "atmosphere_builder", None)
    if builder is not None and callable(getattr(builder, "build", None)):
        models = getattr(model, "models", None)
        owner = (
            next(iter(models.values()))
            if isinstance(models, Mapping) and models
            else model
        )
        return (_Region(prefix, owner, builder),)
    models = getattr(model, "models", None)
    if isinstance(models, Mapping) and models:
        return _named_regions(next(iter(models.values())), prefix=prefix)
    return ()


def _build(builder: object, parameters: Mapping[str, float], name: str) -> object:
    method = getattr(builder, "build", None)
    if not callable(method):
        raise RobertValidationError(
            f"atmosphere builder for region {name!r} is not callable"
        )
    state = method(parameters)
    if state is None:
        raise RobertValidationError(
            f"atmosphere builder for region {name!r} returned no state"
        )
    return state


def _state_values(
    name: str,
    state: object,
    references: dict[str, object],
    temperatures: dict[str, list[FloatArray]],
    vmrs: dict[str, dict[str, list[FloatArray]]],
    units: dict[str, str],
) -> None:
    grid = getattr(state, "pressure_grid", None)
    composition = getattr(state, "composition", None)
    if grid is None or not isinstance(composition, Mapping) or not composition:
        raise RobertValidationError(
            f"atmosphere state for region {name!r} is incomplete"
        )
    pressure = _array(getattr(grid, "centers", None), f"{name} pressure centers")
    _text(getattr(grid, "unit", ""), f"{name} pressure unit")
    temperature = _array(getattr(state, "temperature", None), f"{name} temperature")
    species = tuple(str(item) for item in composition)
    convention = str(getattr(state, "composition_convention", "volume_mixing_ratio"))
    if temperature.shape != pressure.shape:
        raise RobertValidationError(
            f"temperature for region {name!r} must match pressure layers"
        )
    if name not in references:
        references[name], temperatures[name], vmrs[name], units[name] = (
            state,
            [],
            {item: [] for item in species},
            convention,
        )
    else:
        _same_pressure_grid(references[name], state, name)
        if species != tuple(vmrs[name]) or convention != units[name]:
            raise RobertValidationError(
                f"atmosphere composition changed for region {name!r}"
            )
    temperatures[name].append(temperature)
    for item in species:
        profile = _array(composition[item], f"{name} {item} VMR")
        if profile.shape != pressure.shape:
            raise RobertValidationError(
                f"VMR {item!r} for region {name!r} must match pressure layers"
            )
        vmrs[name][item].append(profile)


def _same_pressure_grid(reference: object, current: object, name: str) -> None:
    left, right = (
        getattr(reference, "pressure_grid", None),
        getattr(current, "pressure_grid", None),
    )
    if (
        left is None
        or right is None
        or left.unit != right.unit
        or not np.array_equal(left.centers, right.centers)
        or not np.array_equal(left.edges, right.edges)
    ):
        raise RobertValidationError(f"pressure grid changed for region {name!r}")


def _cloud_values(
    region: _Region,
    state: object,
    parameters: Mapping[str, float],
    values: dict[str, dict[str, list[FloatArray]]],
    units: dict[tuple[str, str], str],
    unsupported: dict[str, dict[str, str]],
) -> None:
    if (
        not hasattr(region.model, "cloud_model")
        or getattr(region.model, "cloud_model") is None
    ):
        return
    from .cloud_profiles import CloudProfileUnavailableError, cloud_profiles

    try:
        profiles = cloud_profiles(
            region.model, cast(AtmosphereState, state), parameters
        )
    except CloudProfileUnavailableError as error:
        unsupported.setdefault(region.name, {})["cloud_profiles"] = (
            str(error) or "unsupported"
        )
        return
    if not isinstance(profiles, Mapping):
        raise RobertValidationError(
            "cloud_profiles must return a named profile mapping"
        )
    if not profiles:
        unsupported.setdefault(region.name, {})["cloud_profiles"] = (
            "no profile diagnostics returned"
        )
        return
    pressure = getattr(state, "pressure_grid").centers
    for raw_name, profile in profiles.items():
        name = str(raw_name)
        if (
            not isinstance(profile, Mapping)
            or "values" not in profile
            or "unit" not in profile
        ):
            raise RobertValidationError(
                f"cloud profile {name!r} must contain values and unit"
            )
        profile_values = _array(profile["values"], f"{region.name} cloud {name}")
        if profile_values.shape != pressure.shape:
            raise RobertValidationError(
                f"cloud profile {name!r} must match pressure layers"
            )
        unit = str(profile["unit"])
        _text(unit, f"cloud profile {name!r} unit")
        key = (region.name, name)
        if key in units and units[key] != unit:
            raise RobertValidationError(
                f"cloud profile {name!r} unit changed across draws"
            )
        units[key] = unit
        values[region.name].setdefault(name, []).append(profile_values)


def _profile(
    state: object, values: list[FloatArray], unit: str
) -> PosteriorProfileSummary:
    grid = getattr(state, "pressure_grid")
    return PosteriorProfileSummary(
        grid.centers, str(grid.unit), _quantiles(values), unit
    )


def _profile_maps(
    samples: Mapping[str, Mapping[str, list[FloatArray]]],
    units: Mapping[object, str],
    states: Mapping[str, object],
    *,
    cloud: bool = False,
) -> dict[str, dict[str, PosteriorProfileSummary]]:
    output: dict[str, dict[str, PosteriorProfileSummary]] = {}
    for region, profiles in samples.items():
        if not profiles:
            continue
        grid = getattr(states[region], "pressure_grid")
        output[region] = {}
        for name, values in profiles.items():
            key: object = (region, name) if cloud else region
            output[region][name] = PosteriorProfileSummary(
                grid.centers,
                str(grid.unit),
                _quantiles(values),
                units[key],
            )
    return output


def _lengths(values: Mapping[str, object], expected: int, label: str) -> None:
    for name, profiles in values.items():
        if isinstance(profiles, Mapping):
            _lengths(profiles, expected, f"{label} {name!r}")
        elif len(cast(list[FloatArray], profiles)) != expected:
            raise RobertValidationError(f"{label} {name!r} is missing selected draws")


def _array(values: object, name: str) -> FloatArray:
    if values is None:
        raise RobertValidationError(f"{name} must be provided")
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must be finite and one-dimensional")
    array.setflags(write=False)
    return array


def _quantiles(
    values: object, size: int | None = None, name: str = "posterior values"
) -> FloatArray:
    if isinstance(values, list):
        if not values:
            raise RobertValidationError(f"{name} has no samples")
        data = np.stack(values)
    else:
        data = np.array(values, dtype=float, copy=False)
    result = np.asarray(
        np.quantile(data, POSTERIOR_QUANTILE_PROBABILITIES, axis=0), dtype=float
    )
    if (
        result.ndim != 2
        or (size is not None and result.shape != (5, size))
        or not np.all(np.isfinite(result))
    ):
        raise RobertValidationError(f"{name} must produce five finite quantile rows")
    result.setflags(write=False)
    return result


def _summary_quantiles(values: object, size: int, name: str) -> FloatArray:
    result = np.array(values, dtype=float, copy=True)
    if result.shape != (5, size) or not np.all(np.isfinite(result)):
        raise RobertValidationError(f"{name} must have finite shape (5, {size})")
    result.setflags(write=False)
    return result


def _text(value: object, name: str) -> None:
    if not str(value).strip():
        raise RobertValidationError(f"{name} must not be empty")


def _nested(
    values: Mapping[str, Mapping[str, PosteriorProfileSummary]],
) -> Mapping[str, Mapping[str, PosteriorProfileSummary]]:
    return immutable_mapping(
        {
            str(region): immutable_mapping(
                {str(name): profile for name, profile in profiles.items()}
            )
            for region, profiles in values.items()
        }
    )


def _nested_reasons(
    values: Mapping[str, Mapping[str, str]],
) -> Mapping[str, Mapping[str, str]]:
    return immutable_mapping(
        {
            str(region): immutable_mapping(
                {str(name): str(reason) for name, reason in reasons.items()}
            )
            for region, reasons in values.items()
        }
    )


__all__ = [
    "POSTERIOR_QUANTILE_LABELS",
    "POSTERIOR_QUANTILE_PROBABILITIES",
    "PosteriorModelCollection",
    "PosteriorProfileSummary",
    "PosteriorProductUnavailableError",
    "PosteriorSpectrumSummary",
    "collect_posterior_models",
]
