"""Smith/Brogi-Line likelihood for time-resolved high-resolution spectra.

Smith et al. (2024) apply principal-component removal to each IGRINS order
and use the Brogi & Line (2019) likelihood on the filtered spectra.  This
module implements that contract for an immutable
``TimeResolvedHighResolutionObservation``.  It intentionally keeps the
calculation on the CPU and uses deterministic ``numpy.linalg.svd``; a
retrieval caller can therefore cap BLAS threads through the normal ROBERT
runtime policy.

The input detector cube is ``(order, frame, pixel)``.  For each order the
first ``n_components`` left/right singular-vector products form the low-rank
scaling matrix ``data_scale``.  A trial model is injected as
``(1 + Fp/Fs) * data_scale``, reprocessed with the same number of principal
components, and then compared with the similarly filtered data.  The
per-order/per-frame statistic is

``-n_active / 2 * log((m2 + f2 - 2 * R) / n_active)``.  The data residual is
formed once as ``data - data_scale``.  Smith's order-level three-sigma clip
zero-fills outliers but does not remove their pixels from ``n_active`` or the
model norm.

Positive velocity is a redshift.  The Smith default samples the source grid at
``wavelength * (1 - beta)``.  A relativistic mapping is available only as a
named diagnostic.  All ROBERT chemistry interfaces remain
volume-mixing-ratio based; this likelihood only uses a supplied
planet/stellar flux template.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertDataError, RobertValidationError
from robert_exoplanets.instruments.time_resolved_high_resolution import (
    HighResolutionEmissionTemplate,
    TimeResolvedHighResolutionObservation,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
_SPEED_OF_LIGHT_KM_S = 299_792.458


def _readonly_float_array(values: ArrayLike, name: str, *, ndim: int | None = None) -> FloatArray:
    """Copy and freeze one finite floating-point array."""

    array = np.array(values, dtype=float, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise RobertValidationError(f"{name} must be {ndim}-dimensional")
    if array.size == 0:
        raise RobertValidationError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _readonly_bool_array(values: ArrayLike, name: str, shape: tuple[int, ...]) -> BoolArray:
    """Copy and freeze one boolean array with an exact shape."""

    array = np.array(values, dtype=bool, copy=True)
    if array.shape != shape:
        raise RobertValidationError(f"{name} must have shape {shape}")
    array.setflags(write=False)
    return array


def _finite_parameter(parameters: Mapping[str, float], name: str, default: float) -> float:
    """Read one finite runtime parameter."""

    value = default if name not in parameters else parameters[name]
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RobertValidationError(f"{name} must be finite") from error
    if not np.isfinite(result):
        raise RobertValidationError(f"{name} must be finite")
    return result


def _positive_integer(value: int, name: str) -> int:
    """Validate one non-negative integer option."""

    if isinstance(value, (bool, np.bool_)):
        raise RobertValidationError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise RobertValidationError(f"{name} must be a non-negative integer") from error
    if result != value or result < 0:
        raise RobertValidationError(f"{name} must be a non-negative integer")
    return result


def _positive_float(value: float, name: str) -> float:
    """Validate one finite positive scalar."""

    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RobertValidationError(f"{name} must be finite and positive") from error
    if not np.isfinite(result) or result <= 0.0:
        raise RobertValidationError(f"{name} must be finite and positive")
    return result


def _velocity(value: float, name: str) -> float:
    """Validate a velocity in km/s."""

    result = _finite_parameter({name: value}, name, 0.0)
    if abs(result) >= _SPEED_OF_LIGHT_KM_S:
        raise RobertValidationError(f"{name} must be below the speed of light")
    return result


def _template(value: HighResolutionEmissionTemplate | Any) -> HighResolutionEmissionTemplate:
    """Validate a high-resolution emission template."""

    if not isinstance(value, HighResolutionEmissionTemplate):
        raise RobertValidationError(
            "prediction must be a HighResolutionEmissionTemplate"
        )
    return value


def _low_rank_reconstruction(
    matrix: FloatArray,
    n_components: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return low-rank reconstruction and the exact deterministic SVD."""

    if matrix.ndim != 2:
        raise RobertValidationError("PCA input must be two-dimensional")
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = min(int(n_components), singular_values.size)
    if rank == 0:
        reconstruction = np.zeros_like(matrix)
    else:
        reconstruction = (u[:, :rank] * singular_values[:rank]) @ vt[:rank, :]
    return (
        np.asarray(reconstruction, dtype=float),
        np.asarray(singular_values, dtype=float),
        np.asarray(u[:, :rank] if rank else u[:, :0], dtype=float),
    )


def _freeze_in_place(array: np.ndarray) -> FloatArray:
    """Set a temporary NumPy array read-only and return it."""

    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TimeResolvedHighResolutionModel:
    """Diagnostic arrays for one reprocessed trial model."""

    velocity_km_s: FloatArray
    flux_ratio: FloatArray
    injected_model: FloatArray
    scaling_matrix: FloatArray
    filtered_model: FloatArray
    centered_model: FloatArray
    scale: float

    def __post_init__(self) -> None:
        velocity = _readonly_float_array(self.velocity_km_s, "velocity_km_s", ndim=1)
        ratio = _readonly_float_array(self.flux_ratio, "flux_ratio", ndim=3)
        injected = _readonly_float_array(self.injected_model, "injected_model", ndim=3)
        scaling = _readonly_float_array(self.scaling_matrix, "scaling_matrix", ndim=3)
        filtered = _readonly_float_array(self.filtered_model, "filtered_model", ndim=3)
        centered = _readonly_float_array(self.centered_model, "centered_model", ndim=3)
        if injected.shape != scaling.shape or injected.shape != filtered.shape:
            raise RobertValidationError("model diagnostic cube shapes must match")
        if centered.shape != injected.shape or ratio.shape != injected.shape:
            raise RobertValidationError("model flux-ratio and diagnostic cube shapes must match")
        if velocity.size != injected.shape[1]:
            raise RobertValidationError("model velocity length must match frame dimension")
        scale = _positive_float(self.scale, "model scale")
        for name, value in (
            ("velocity_km_s", velocity),
            ("flux_ratio", ratio),
            ("injected_model", injected),
            ("scaling_matrix", scaling),
            ("filtered_model", filtered),
            ("centered_model", centered),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "scale", scale)


@dataclass(frozen=True)
class PreparedTimeResolvedHighResolutionLikelihood:
    """PCA-prepared data and exact Smith/Brogi-Line statistic."""

    observation: TimeResolvedHighResolutionObservation
    n_components: int = 4
    sigma_clip: float = 3.0
    kp_parameter: str = "Kp"
    dVsys_parameter: str = "dVsys"
    dphi_parameter: str = "dphi"
    scale_parameter: str = "log10_a"
    scale_is_log10: bool = True
    doppler_mode: str = "smith_nonrelativistic"
    flux_ratio_scale: float = 1.0
    invalid_model_loglike: float = float("-inf")
    name: str = "prepared-smith2024-pca-likelihood"
    data_scale: FloatArray = field(init=False)
    pca_scaling: FloatArray = field(init=False)
    data_residual: FloatArray = field(init=False)
    data_centered: FloatArray = field(init=False)
    clipping_mask: BoolArray = field(init=False)
    observation_mask: BoolArray = field(init=False)
    f2_by_order_frame: FloatArray = field(init=False)
    n_active_by_order_frame: NDArray[np.int64] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, TimeResolvedHighResolutionObservation):
            raise RobertValidationError(
                "observation must be a TimeResolvedHighResolutionObservation"
            )
        n_components = _positive_integer(self.n_components, "n_components")
        sigma_clip = _positive_float(self.sigma_clip, "sigma_clip")
        if n_components >= min(self.observation.n_frames, self.observation.n_pixels):
            raise RobertValidationError(
                "n_components must be smaller than both frame and pixel dimensions"
            )
        if not isinstance(self.scale_is_log10, (bool, np.bool_)):
            raise RobertValidationError("scale_is_log10 must be boolean")
        doppler_mode = str(self.doppler_mode).strip().lower()
        if doppler_mode not in {
            "smith",
            "smith_nonrelativistic",
            "nonrelativistic",
            "relativistic",
            "relativistic_doppler",
        }:
            raise RobertValidationError(
                "doppler_mode must be smith_nonrelativistic or relativistic"
            )
        flux_ratio_scale = _positive_float(self.flux_ratio_scale, "flux_ratio_scale")
        invalid = float(self.invalid_model_loglike)
        if np.isnan(invalid) or invalid == np.inf:
            raise RobertValidationError("invalid_model_loglike must be finite or -inf")
        for parameter_name, label in (
            (self.kp_parameter, "kp_parameter"),
            (self.dVsys_parameter, "dVsys_parameter"),
            (self.dphi_parameter, "dphi_parameter"),
            (self.scale_parameter, "scale_parameter"),
        ):
            if not str(parameter_name).strip():
                raise RobertValidationError(f"{label} must not be empty")
        if not str(self.name).strip():
            raise RobertValidationError("name must not be empty")

        raw_data = np.asarray(self.observation.flux, dtype=float)
        pca_scaling = np.zeros_like(raw_data)
        data_residual = np.zeros_like(raw_data)
        observation_mask = np.array(self.observation.mask, dtype=bool, copy=True)
        clipping_mask = np.zeros_like(observation_mask)

        for order in range(self.observation.n_orders):
            order_data = raw_data[order]
            scaling, _, _ = _low_rank_reconstruction(order_data, n_components)
            residual = order_data - scaling
            pca_scaling[order] = scaling
            # Smith/POSEIDON supplies this PCA residual directly to the
            # likelihood.  The clipping statistics cover one whole order,
            # rather than one frame at a time.  Clipped data are zero-filled,
            # while their observational pixels remain active in the model
            # norm and in ``n_active``.
            values = residual[observation_mask[order]]
            if values.size:
                median = float(np.median(values))
                spread = float(np.std(values))
                if spread > 0.0 and np.isfinite(spread):
                    clipped = observation_mask[order] & (
                        np.abs(residual - median) > sigma_clip * spread
                    )
                    clipping_mask[order] = clipped
                    residual[clipped] = 0.0
            data_residual[order] = residual

        if not np.any(observation_mask):
            raise RobertValidationError("observation mask excludes every data point")
        f2 = np.zeros((self.observation.n_orders, self.observation.n_frames), dtype=float)
        active = np.zeros(
            (self.observation.n_orders, self.observation.n_frames),
            dtype=np.int64,
        )
        for order in range(self.observation.n_orders):
            for frame in range(self.observation.n_frames):
                valid = observation_mask[order, frame]
                n_valid = int(np.count_nonzero(valid))
                active[order, frame] = n_valid
                if n_valid == 0:
                    continue
                values = data_residual[order, frame, valid]
                f2[order, frame] = float(np.dot(values, values))

        for array in (
            pca_scaling,
            data_residual,
            f2,
            active,
            clipping_mask,
            observation_mask,
        ):
            array.setflags(write=False)
        object.__setattr__(self, "n_components", n_components)
        object.__setattr__(self, "sigma_clip", sigma_clip)
        object.__setattr__(self, "invalid_model_loglike", invalid)
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "doppler_mode", doppler_mode)
        object.__setattr__(self, "flux_ratio_scale", flux_ratio_scale)
        object.__setattr__(self, "data_scale", pca_scaling)
        object.__setattr__(self, "pca_scaling", pca_scaling)
        object.__setattr__(self, "data_residual", data_residual)
        # Compatibility alias retained for callers that used the earlier
        # name. Smith's residual is not re-centred in the likelihood loop.
        object.__setattr__(self, "data_centered", data_residual)
        object.__setattr__(self, "clipping_mask", clipping_mask)
        object.__setattr__(self, "observation_mask", observation_mask)
        object.__setattr__(self, "f2_by_order_frame", f2)
        object.__setattr__(self, "n_active_by_order_frame", active)

    @property
    def data(self) -> FloatArray:
        """Return the low-rank data scale used for model injection."""

        return self.data_scale

    @property
    def clipped_mask(self) -> BoolArray:
        """Return the outlier mask from the order-level sigma clip."""

        return self.clipping_mask

    @property
    def valid_mask(self) -> BoolArray:
        """Return the observational mask used in the likelihood."""

        return self.observation_mask

    @property
    def effective_residual_rank(self) -> int:
        """Return the number of retained order/frame/pixel samples."""

        return int(np.sum(self.n_active_by_order_frame))

    def _velocity(self, parameters: Mapping[str, float]) -> FloatArray:
        """Build the Smith velocity expression for every frame."""

        kp = _velocity(
            _finite_parameter(parameters, self.kp_parameter, 0.0),
            self.kp_parameter,
        )
        dVsys = _velocity(
            _finite_parameter(parameters, self.dVsys_parameter, 0.0),
            self.dVsys_parameter,
        )
        dphi = _finite_parameter(parameters, self.dphi_parameter, 0.0)
        velocity = (
            dVsys
            + self.observation.fixed_velocity_km_s
            + kp * np.sin(2.0 * np.pi * (self.observation.phase + dphi))
        )
        if np.any(np.abs(velocity) >= _SPEED_OF_LIGHT_KM_S):
            raise RobertValidationError("trial velocity is above the speed of light")
        velocity = np.array(velocity, dtype=float, copy=True)
        velocity.setflags(write=False)
        return velocity

    def velocity(self, parameters: Mapping[str, float] | None = None) -> FloatArray:
        """Return the fixed-plus-orbital per-frame velocity expression."""

        return self._velocity({} if parameters is None else parameters)

    def _scale(self, parameters: Mapping[str, float]) -> float:
        """Read the model scale, supporting a log10 nuisance parameter."""

        values = parameters
        if self.scale_parameter in values:
            raw = _finite_parameter(values, self.scale_parameter, 0.0)
            scale = 10.0**raw if self.scale_is_log10 else raw
        elif self.scale_is_log10 and "a" in values:
            # ``a`` is a useful compatibility spelling for Smith's log(a_i)
            # parameter; it remains interpreted as log10 when this prepared
            # likelihood is configured with the default log scale.
            raw = _finite_parameter(values, "a", 0.0)
            scale = 10.0**raw
        else:
            scale = 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            raise RobertValidationError("model scale must be finite and positive")
        return float(scale)

    def _model_flux_ratio(
        self,
        template: HighResolutionEmissionTemplate,
        velocity: FloatArray,
    ) -> FloatArray:
        """Interpolate one planet/star ratio for every order and frame."""

        ratios = np.empty(
            (
                self.observation.n_orders,
                self.observation.n_frames,
                self.observation.n_pixels,
            ),
            dtype=float,
        )
        for order in range(self.observation.n_orders):
            wavelengths = self.observation.order_wavelengths[order]
            for frame in range(self.observation.n_frames):
                ratios[order, frame] = template.interpolate_flux_ratio(
                    wavelengths,
                    float(velocity[frame]),
                    doppler_mode=self.doppler_mode,
                )
        if not np.all(np.isfinite(ratios)):
            raise RobertValidationError(
                "trial velocity moves part of the observation outside template coverage"
            )
        ratios *= self.flux_ratio_scale * template.flux_ratio_scale
        ratios.setflags(write=False)
        return ratios

    def _model_flux_ratio_order(
        self,
        template: HighResolutionEmissionTemplate,
        velocity: FloatArray,
        order: int,
    ) -> FloatArray:
        """Interpolate one order without allocating a complete model cube."""

        ratios = np.empty(
            (self.observation.n_frames, self.observation.n_pixels),
            dtype=float,
        )
        wavelengths = self.observation.order_wavelengths[order]
        for frame in range(self.observation.n_frames):
            ratios[frame] = template.interpolate_flux_ratio(
                wavelengths,
                float(velocity[frame]),
                doppler_mode=self.doppler_mode,
            )
        if not np.all(np.isfinite(ratios)):
            raise RobertValidationError(
                "trial velocity moves part of the observation outside template coverage"
            )
        ratios *= self.flux_ratio_scale * template.flux_ratio_scale
        return ratios

    def evaluate_model(
        self,
        template: HighResolutionEmissionTemplate,
        parameters: Mapping[str, float] | None = None,
    ) -> TimeResolvedHighResolutionModel:
        """Inject and reprocess one trial template with exact NumPy SVD."""

        template = _template(template)
        if template.wavelength_unit != self.observation.wavelength_unit:
            raise RobertValidationError("template and observation wavelength units must match")
        runtime = {} if parameters is None else parameters
        velocity = self._velocity(runtime)
        ratio = self._model_flux_ratio(template, velocity)
        injected = (1.0 + ratio) * self.data_scale
        scaling = np.zeros_like(injected)
        filtered = np.zeros_like(injected)
        centered = np.zeros_like(injected)
        scale = self._scale(runtime)
        for order in range(self.observation.n_orders):
            model_scaling, _, _ = _low_rank_reconstruction(
                injected[order],
                self.n_components,
            )
            model_filtered = injected[order] - model_scaling
            scaling[order] = model_scaling
            filtered[order] = model_filtered
            for frame in range(self.observation.n_frames):
                valid = self.observation_mask[order, frame]
                if np.any(valid):
                    centred = model_filtered[frame, valid] - np.mean(model_filtered[frame, valid])
                    centered[order, frame, valid] = scale * centred
        return TimeResolvedHighResolutionModel(
            velocity_km_s=velocity,
            flux_ratio=ratio,
            injected_model=injected,
            scaling_matrix=scaling,
            filtered_model=filtered,
            centered_model=centered,
            scale=scale,
        )

    def _statistics(
        self,
        model: TimeResolvedHighResolutionModel,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Return per-order/frame ``m2``, ``f2``, ``R``, and loglike terms."""

        m2 = np.zeros_like(self.f2_by_order_frame)
        f2 = np.array(self.f2_by_order_frame, copy=True)
        cross = np.zeros_like(self.f2_by_order_frame)
        terms = np.full_like(self.f2_by_order_frame, self.invalid_model_loglike)
        for order in range(self.observation.n_orders):
            for frame in range(self.observation.n_frames):
                valid = self.observation_mask[order, frame]
                n_valid = int(self.n_active_by_order_frame[order, frame])
                if n_valid == 0:
                    continue
                data_values = self.data_centered[order, frame, valid]
                model_values = model.centered_model[order, frame, valid]
                model_norm = float(np.dot(model_values, model_values))
                cross_value = float(np.dot(data_values, model_values))
                argument = (model_norm + f2[order, frame] - 2.0 * cross_value) / n_valid
                m2[order, frame] = model_norm
                cross[order, frame] = cross_value
                if not np.isfinite(argument) or argument <= 0.0:
                    continue
                terms[order, frame] = -0.5 * n_valid * np.log(argument)
        for array in (m2, f2, cross, terms):
            array.setflags(write=False)
        return m2, f2, cross, terms

    def _stream_statistics(
        self,
        template: HighResolutionEmissionTemplate,
        parameters: Mapping[str, float] | None = None,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Accumulate the scalar statistic one order at a time.

        A full diagnostic model is useful for audits, but a nested sampler
        must not allocate complete ratio, injection, SVD, and centred-model
        cubes on every call.  This method retains the prepared data cubes and
        keeps only one order's temporary arrays alive.
        """

        template = _template(template)
        if template.wavelength_unit != self.observation.wavelength_unit:
            raise RobertValidationError("template and observation wavelength units must match")
        runtime = {} if parameters is None else parameters
        velocity = self._velocity(runtime)
        scale = self._scale(runtime)
        m2 = np.zeros_like(self.f2_by_order_frame)
        f2 = np.array(self.f2_by_order_frame, copy=True)
        cross = np.zeros_like(self.f2_by_order_frame)
        terms = np.full_like(self.f2_by_order_frame, self.invalid_model_loglike)
        for order in range(self.observation.n_orders):
            ratio = self._model_flux_ratio_order(template, velocity, order)
            injected = (1.0 + ratio) * self.data_scale[order]
            model_scaling, _, _ = _low_rank_reconstruction(
                injected,
                self.n_components,
            )
            model_filtered = injected - model_scaling
            for frame in range(self.observation.n_frames):
                valid = self.observation_mask[order, frame]
                n_valid = int(self.n_active_by_order_frame[order, frame])
                if n_valid == 0:
                    continue
                model_values = model_filtered[frame, valid]
                model_values = scale * (model_values - np.mean(model_values))
                data_values = self.data_centered[order, frame, valid]
                model_norm = float(np.dot(model_values, model_values))
                cross_value = float(np.dot(data_values, model_values))
                argument = (model_norm + f2[order, frame] - 2.0 * cross_value) / n_valid
                m2[order, frame] = model_norm
                cross[order, frame] = cross_value
                if np.isfinite(argument) and argument > 0.0:
                    terms[order, frame] = -0.5 * n_valid * np.log(argument)
        for array in (m2, f2, cross, terms):
            array.setflags(write=False)
        return m2, f2, cross, terms

    def loglike(
        self,
        template: HighResolutionEmissionTemplate,
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        """Evaluate the exact Smith/Brogi-Line PCA log likelihood."""

        try:
            _, _, _, terms = self._stream_statistics(template, parameters)
        except (RobertDataError, RobertValidationError, ValueError, FloatingPointError):
            return float(self.invalid_model_loglike)
        active = self.n_active_by_order_frame > 0
        if not np.any(active) or not np.all(np.isfinite(terms[active])):
            return float(self.invalid_model_loglike)
        result = float(np.sum(terms[active]))
        return result if np.isfinite(result) else float(self.invalid_model_loglike)

    def loglike_terms(
        self,
        template: HighResolutionEmissionTemplate,
        parameters: Mapping[str, float] | None = None,
    ) -> FloatArray:
        """Return one Brogi-Line term per order and frame."""

        return self._stream_statistics(template, parameters)[3]

    def cross_correlation(
        self,
        template: HighResolutionEmissionTemplate,
        parameters: Mapping[str, float] | None = None,
    ) -> FloatArray:
        """Return normalized CCF values per order and frame."""

        m2, f2, cross, _ = self._stream_statistics(template, parameters)
        denominator = np.sqrt(m2 * f2)
        output = np.full(cross.shape, np.nan, dtype=float)
        valid = denominator > 0.0
        output[valid] = cross[valid] / denominator[valid]
        output.setflags(write=False)
        return output

    # Common CCF terminology aliases.
    ccf = cross_correlation
    ccf_diagnostic = cross_correlation

    def ccf_map(
        self,
        template: HighResolutionEmissionTemplate,
        kp_values_km_s: ArrayLike,
        dVsys_values_km_s: ArrayLike,
        *,
        parameters: Mapping[str, float] | None = None,
    ) -> FloatArray:
        """Build a deterministic summed CCF map over ``Kp`` and ``dVsys``."""

        kp_values = _readonly_float_array(kp_values_km_s, "kp_values_km_s", ndim=1)
        dVsys_values = _readonly_float_array(
            dVsys_values_km_s,
            "dVsys_values_km_s",
            ndim=1,
        )
        base = {} if parameters is None else dict(parameters)
        output = np.empty((kp_values.size, dVsys_values.size), dtype=float)
        for i, kp in enumerate(kp_values):
            for j, dVsys in enumerate(dVsys_values):
                trial = dict(base)
                trial[self.kp_parameter] = float(kp)
                trial[self.dVsys_parameter] = float(dVsys)
                try:
                    values = self.cross_correlation(template, trial)
                except (RobertDataError, RobertValidationError, ValueError, FloatingPointError):
                    output[i, j] = np.nan
                else:
                    output[i, j] = float(np.nansum(values))
        output.setflags(write=False)
        return output


@dataclass(frozen=True)
class TimeResolvedHighResolutionLikelihood:
    """Configuration for the Smith et al. (2024) HRS likelihood."""

    n_components: int = 4
    sigma_clip: float = 3.0
    kp_parameter: str = "Kp"
    dVsys_parameter: str = "dVsys"
    dphi_parameter: str = "dphi"
    scale_parameter: str = "log10_a"
    scale_is_log10: bool = True
    doppler_mode: str = "smith_nonrelativistic"
    flux_ratio_scale: float = 1.0
    invalid_model_loglike: float = float("-inf")
    name: str = "smith2024-pca-likelihood"

    def __post_init__(self) -> None:
        n_components = _positive_integer(self.n_components, "n_components")
        if not np.isfinite(self.sigma_clip) or self.sigma_clip <= 0.0:
            raise RobertValidationError("sigma_clip must be finite and positive")
        invalid = float(self.invalid_model_loglike)
        if np.isnan(invalid) or invalid == np.inf:
            raise RobertValidationError("invalid_model_loglike must be finite or -inf")
        for value, label in (
            (self.kp_parameter, "kp_parameter"),
            (self.dVsys_parameter, "dVsys_parameter"),
            (self.dphi_parameter, "dphi_parameter"),
            (self.scale_parameter, "scale_parameter"),
            (self.name, "name"),
        ):
            if not str(value).strip():
                raise RobertValidationError(f"{label} must not be empty")
        if not isinstance(self.scale_is_log10, (bool, np.bool_)):
            raise RobertValidationError("scale_is_log10 must be boolean")
        doppler_mode = str(self.doppler_mode).strip().lower()
        if doppler_mode not in {
            "smith",
            "smith_nonrelativistic",
            "nonrelativistic",
            "relativistic",
            "relativistic_doppler",
        }:
            raise RobertValidationError(
                "doppler_mode must be smith_nonrelativistic or relativistic"
            )
        flux_ratio_scale = _positive_float(self.flux_ratio_scale, "flux_ratio_scale")
        object.__setattr__(self, "n_components", n_components)
        object.__setattr__(self, "sigma_clip", float(self.sigma_clip))
        object.__setattr__(self, "invalid_model_loglike", invalid)
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "doppler_mode", doppler_mode)
        object.__setattr__(self, "flux_ratio_scale", flux_ratio_scale)

    def prepare(
        self,
        observation: TimeResolvedHighResolutionObservation,
    ) -> PreparedTimeResolvedHighResolutionLikelihood:
        """Compute data PCA and clipping once for repeated model evaluations."""

        return PreparedTimeResolvedHighResolutionLikelihood(
            observation=observation,
            n_components=self.n_components,
            sigma_clip=self.sigma_clip,
            kp_parameter=self.kp_parameter,
            dVsys_parameter=self.dVsys_parameter,
            dphi_parameter=self.dphi_parameter,
            scale_parameter=self.scale_parameter,
            scale_is_log10=self.scale_is_log10,
            doppler_mode=self.doppler_mode,
            flux_ratio_scale=self.flux_ratio_scale,
            invalid_model_loglike=self.invalid_model_loglike,
            name=f"prepared-{self.name}",
        )

    def loglike(
        self,
        template: HighResolutionEmissionTemplate,
        observation: TimeResolvedHighResolutionObservation,
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        """Prepare a temporary likelihood and evaluate one template."""

        return self.prepare(observation).loglike(template, parameters)


# Names used by the Smith and Brogi-Line literature.
Smith2024HighResolutionLikelihood = TimeResolvedHighResolutionLikelihood
Smith2024PcaLikelihood = TimeResolvedHighResolutionLikelihood
PreparedSmith2024HighResolutionLikelihood = PreparedTimeResolvedHighResolutionLikelihood
PreparedSmith2024PcaLikelihood = PreparedTimeResolvedHighResolutionLikelihood
SmithPcaLikelihood = TimeResolvedHighResolutionLikelihood
PreparedSmithPcaLikelihood = PreparedTimeResolvedHighResolutionLikelihood
BrogiLineLikelihood = TimeResolvedHighResolutionLikelihood
PreparedBrogiLineLikelihood = PreparedTimeResolvedHighResolutionLikelihood


__all__ = [
    "BrogiLineLikelihood",
    "PreparedBrogiLineLikelihood",
    "PreparedSmith2024HighResolutionLikelihood",
    "PreparedSmith2024PcaLikelihood",
    "PreparedSmithPcaLikelihood",
    "PreparedTimeResolvedHighResolutionLikelihood",
    "Smith2024HighResolutionLikelihood",
    "Smith2024PcaLikelihood",
    "SmithPcaLikelihood",
    "TimeResolvedHighResolutionLikelihood",
    "TimeResolvedHighResolutionModel",
]
