"""Retrieval-facing parameterized high-resolution response orchestration."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Callable, Mapping, Protocol

import numpy as np

from robert_exoplanets.core import RobertValidationError, SpectralGrid, Spectrum
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.instruments import (
    GaussianHighResolutionResponse,
    HighResolutionResponseChain,
    Observation,
    PreparedHighResolutionResponseChain,
    RelativisticDopplerResponse,
    RotationalBroadeningResponse,
)


class PreparedSpectrumResponse(Protocol):
    """Prepared fixed response contract."""

    def observe(self, spectrum: Spectrum) -> Spectrum: ...


class RuntimeSpectrumResponse(Protocol):
    """Response that reads one or more retrieval parameters."""

    required_parameters: tuple[str, ...]

    def observe(
        self,
        spectrum: Spectrum,
        parameters: Mapping[str, float],
    ) -> Spectrum: ...


NativeMultiDatasetEvaluator = Callable[[Mapping[str, float]], Mapping[str, Spectrum]]
GridCacheKey = tuple[str, bytes, bytes | None]


def _grid_cache_key(spectrum: Spectrum) -> GridCacheKey:
    """Return the physical grid state used by prepared response caches."""

    grid = spectrum.spectral_grid
    values_key = np.asarray(grid.values, dtype=float).tobytes()
    edges_key = (
        None
        if grid.bin_edges is None
        else np.asarray(grid.bin_edges, dtype=float).tobytes()
    )
    return grid.unit, values_key, edges_key


@dataclass(frozen=True)
class VelocityParameterizedHighResolutionResponse:
    """Prepare and apply an exact response chain for a trial velocity.

    Velocity changes the response coordinates, so the Doppler operator cannot
    be one fixed matrix for the complete retrieval. The expensive atmospheric
    spectrum stays separate. A small bounded LRU keeps repeated velocity
    states without allowing response matrices to grow without limit. Setting
    ``max_cached_native_grids`` to zero disables both caches. A cached dynamic
    response retains its native-grid broadening prefix, so retaining it while
    the native-prefix cache is disabled would defeat that setting.
    """

    observation: Observation
    velocity_parameter: str = "radial_velocity_km_s"
    resolving_power: float = 100_000.0
    projected_rotation_km_s: float = 0.0
    limb_darkening: float = 0.0
    kernel_support_sigma: float = 4.0
    max_cached_responses: int = 2
    max_cached_native_grids: int = 1
    name: str = "velocity-parameterized-high-resolution-response"
    _cache: OrderedDict[
        tuple[GridCacheKey, float],
        PreparedHighResolutionResponseChain,
    ] = field(init=False, repr=False, compare=False)
    _native_cache: OrderedDict[
        GridCacheKey,
        tuple[tuple[object, ...], SpectralGrid],
    ] = field(init=False, repr=False, compare=False)
    _cache_lock: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        parameter = str(self.velocity_parameter).strip()
        if not parameter:
            raise RobertValidationError("velocity_parameter must not be empty")
        resolving_power = float(self.resolving_power)
        projected_rotation = float(self.projected_rotation_km_s)
        limb_darkening = float(self.limb_darkening)
        support = float(self.kernel_support_sigma)
        if not np.isfinite(resolving_power) or resolving_power <= 0.0:
            raise RobertValidationError("resolving_power must be finite and positive")
        if not np.isfinite(projected_rotation) or projected_rotation < 0.0:
            raise RobertValidationError(
                "projected_rotation_km_s must be finite and non-negative"
            )
        if not np.isfinite(limb_darkening) or not 0.0 <= limb_darkening <= 1.0:
            raise RobertValidationError("limb_darkening must be between zero and one")
        if not np.isfinite(support) or support <= 0.0:
            raise RobertValidationError(
                "kernel_support_sigma must be finite and positive"
            )
        if (
            isinstance(self.max_cached_responses, bool)
            or int(self.max_cached_responses) != self.max_cached_responses
            or int(self.max_cached_responses) < 0
        ):
            raise RobertValidationError(
                "max_cached_responses must be a non-negative integer"
            )
        if (
            isinstance(self.max_cached_native_grids, bool)
            or int(self.max_cached_native_grids) != self.max_cached_native_grids
            or int(self.max_cached_native_grids) < 0
        ):
            raise RobertValidationError(
                "max_cached_native_grids must be a non-negative integer"
            )
        if not self.name:
            raise RobertValidationError("response name must not be empty")
        object.__setattr__(self, "velocity_parameter", parameter)
        object.__setattr__(self, "resolving_power", resolving_power)
        object.__setattr__(self, "projected_rotation_km_s", projected_rotation)
        object.__setattr__(self, "limb_darkening", limb_darkening)
        object.__setattr__(self, "kernel_support_sigma", support)
        object.__setattr__(self, "max_cached_responses", int(self.max_cached_responses))
        object.__setattr__(
            self,
            "max_cached_native_grids",
            int(self.max_cached_native_grids),
        )
        object.__setattr__(self, "_cache", OrderedDict())
        object.__setattr__(self, "_native_cache", OrderedDict())
        object.__setattr__(self, "_cache_lock", RLock())

    @property
    def required_parameters(self) -> tuple[str, ...]:
        return (self.velocity_parameter,)

    @property
    def cached_response_count(self) -> int:
        """Return the number of velocity response chains in the bounded LRU."""

        with self._cache_lock:  # type: ignore[attr-defined]
            return len(self._cache)

    @property
    def cached_native_grid_count(self) -> int:
        """Return the number of prepared velocity-invariant broadening grids."""

        with self._cache_lock:  # type: ignore[attr-defined]
            return len(self._native_cache)

    def clear_cache(self) -> None:
        """Release all prepared velocity response chains."""

        with self._cache_lock:  # type: ignore[attr-defined]
            self._cache.clear()
            self._native_cache.clear()

    def _invariant_operators(
        self,
        spectrum: Spectrum,
    ) -> tuple[tuple[object, ...], SpectralGrid]:
        """Prepare rotation and constant-R LSF once on the rest-frame grid.

        Both kernels are scale-equivariant. Rotation is translation-invariant
        in log wavelength, and the Gaussian width is proportional to its
        central wavelength. They therefore commute with a multiplicative
        Doppler wavelength shift. Pixel integration does not commute and stays
        after the trial shift.
        """

        key = _grid_cache_key(spectrum)
        with self._cache_lock:  # type: ignore[attr-defined]
            cached = self._native_cache.get(key)
            if cached is not None:
                self._native_cache.move_to_end(key)
                return cached

        working_grid = spectrum.spectral_grid
        operators: list[object] = []
        if self.projected_rotation_km_s > 0.0:
            rotation = RotationalBroadeningResponse(
                projected_velocity_km_s=self.projected_rotation_km_s,
                limb_darkening=self.limb_darkening,
            ).prepare_on_grid(working_grid)
            operators.append(rotation)
            working_grid = rotation.target_grid
        lsf = GaussianHighResolutionResponse(
            resolving_power=self.resolving_power,
            kernel_support=self.kernel_support_sigma,
        ).prepare_on_grid(working_grid)
        operators.append(lsf)
        working_grid = lsf.target_grid
        prepared = (tuple(operators), working_grid)

        if self.max_cached_native_grids > 0:
            with self._cache_lock:  # type: ignore[attr-defined]
                # A concurrent first call can finish the same prefix while
                # this call prepares outside the lock. Reuse that prefix so
                # only one prepared operator set remains reachable.
                cached = self._native_cache.get(key)
                if cached is not None:
                    self._native_cache.move_to_end(key)
                    return cached
                while len(self._native_cache) >= self.max_cached_native_grids:
                    evicted_key, _ = self._native_cache.popitem(last=False)
                    self._evict_dynamic_responses(evicted_key)
                self._native_cache[key] = prepared
        return prepared

    def _evict_dynamic_responses(self, grid_key: GridCacheKey) -> None:
        """Remove dynamic responses that retain one evicted native prefix.

        The caller must hold ``_cache_lock``. Dynamic response chains contain
        references to the invariant broadening operators, so native-grid LRU
        eviction must remove those chains as well.
        """

        stale_keys = [
            cache_key
            for cache_key in self._cache
            if cache_key[0] == grid_key
        ]
        for cache_key in stale_keys:
            del self._cache[cache_key]

    def _prepared(
        self,
        spectrum: Spectrum,
        velocity_km_s: float,
    ) -> PreparedHighResolutionResponseChain:
        grid_key = _grid_cache_key(spectrum)
        key = (grid_key, velocity_km_s)
        with self._cache_lock:  # type: ignore[attr-defined]
            prepared = self._cache.get(key)
            if prepared is not None:
                self._cache.move_to_end(key)
                return prepared

        invariant_operators, broadened_grid = self._invariant_operators(spectrum)
        shifted = HighResolutionResponseChain(
            (RelativisticDopplerResponse(velocity_km_s),)
        ).prepare(
            self.observation,
            broadened_grid,
        )
        prepared = PreparedHighResolutionResponseChain(
            observation=self.observation,
            native_grid=spectrum.spectral_grid,
            operators=(*invariant_operators, *shifted.operators),
            name=self.name,
        )
        # A dynamic chain retains the invariant broadening prefix. Do not
        # retain it when native-prefix caching is disabled, otherwise
        # max_cached_native_grids=0 would still keep prepared prefixes alive.
        if self.max_cached_native_grids > 0 and self.max_cached_responses > 0:
            with self._cache_lock:  # type: ignore[attr-defined]
                # Another thread can evict this native prefix while the
                # velocity-dependent mapping is being prepared. In that case
                # leave the dynamic chain uncached; its result remains valid
                # for this call but must not extend the evicted prefix lifetime.
                cached_prefix = self._native_cache.get(grid_key)
                if (
                    cached_prefix is None
                    or cached_prefix[0] is not invariant_operators
                    or cached_prefix[1] is not broadened_grid
                ):
                    return prepared
                while len(self._cache) >= self.max_cached_responses:
                    self._cache.popitem(last=False)
                self._cache[key] = prepared
        return prepared

    def observe(
        self,
        spectrum: Spectrum,
        parameters: Mapping[str, float],
    ) -> Spectrum:
        """Apply the exact chain for the current radial velocity."""

        try:
            velocity = float(parameters[self.velocity_parameter])
        except KeyError as error:
            raise RobertValidationError(
                f"missing response parameter: {self.velocity_parameter}"
            ) from error
        if not np.isfinite(velocity):
            raise RobertValidationError("radial velocity must be finite")
        observed = self._prepared(spectrum, velocity).observe(spectrum)
        physical_stages = ["relativistic-doppler"]
        if self.projected_rotation_km_s > 0.0:
            physical_stages.append("rotational-broadening")
        physical_stages.extend(
            ("gaussian-high-resolution", "pixel-or-observation-mapping")
        )
        metadata = dict(observed.metadata)
        metadata.update(
            {
                "physical_response_order": ",".join(physical_stages),
                "response_optimization": (
                    "scale-equivariant rotation and constant-R LSF prepared "
                    "before trial Doppler coordinate scaling"
                ),
            }
        )
        return Spectrum(
            spectral_grid=observed.spectral_grid,
            values=observed.values,
            unit=observed.unit,
            observable=observed.observable,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ParameterizedMultiDatasetResponseForwardModel:
    """Apply fixed and runtime responses to named native model spectra."""

    native_model: NativeMultiDatasetEvaluator
    fixed_responses: Mapping[str, PreparedSpectrumResponse] = field(default_factory=dict)
    runtime_responses: Mapping[str, RuntimeSpectrumResponse] = field(default_factory=dict)
    required_parameters: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not callable(self.native_model):
            raise RobertValidationError("native_model must be callable")
        fixed = {str(name).strip(): response for name, response in self.fixed_responses.items()}
        runtime = {
            str(name).strip(): response for name, response in self.runtime_responses.items()
        }
        if any(not name for name in (*fixed, *runtime)):
            raise RobertValidationError("response dataset names must not be empty")
        overlap = set(fixed).intersection(runtime)
        if overlap:
            raise RobertValidationError(
                "a dataset cannot have both fixed and runtime responses"
            )
        if any(not callable(getattr(response, "observe", None)) for response in fixed.values()):
            raise RobertValidationError("fixed responses must implement observe")
        if any(not callable(getattr(response, "observe", None)) for response in runtime.values()):
            raise RobertValidationError("runtime responses must implement observe")
        parameters = list(getattr(self.native_model, "required_parameters", ()))
        for response in runtime.values():
            parameters.extend(response.required_parameters)
        if len(set(parameters)) != len(parameters):
            raise RobertValidationError(
                "native and response required parameter names must be unique"
            )
        object.__setattr__(self, "fixed_responses", immutable_mapping(fixed))
        object.__setattr__(self, "runtime_responses", immutable_mapping(runtime))
        object.__setattr__(self, "required_parameters", tuple(parameters))

    def __call__(self, parameters: Mapping[str, float]) -> Mapping[str, Spectrum]:
        spectra = self.native_model(parameters)
        if not isinstance(spectra, Mapping) or any(
            not isinstance(value, Spectrum) for value in spectra.values()
        ):
            raise RobertValidationError(
                "native multi-dataset model must return named Spectrum values"
            )
        unknown = (set(self.fixed_responses) | set(self.runtime_responses)) - set(spectra)
        if unknown:
            raise RobertValidationError(
                "response dataset names must exist in the native prediction"
            )
        output: dict[str, Spectrum] = {}
        for name, spectrum in spectra.items():
            if name in self.fixed_responses:
                output[name] = self.fixed_responses[name].observe(spectrum)
            elif name in self.runtime_responses:
                output[name] = self.runtime_responses[name].observe(
                    spectrum,
                    parameters,
                )
            else:
                output[name] = spectrum
        return immutable_mapping(output)


__all__ = [
    "ParameterizedMultiDatasetResponseForwardModel",
    "RuntimeSpectrumResponse",
    "VelocityParameterizedHighResolutionResponse",
]
