"""Device-resident instrument projection and Gaussian likelihoods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robert_exoplanets.core import RobertValidationError


@dataclass(frozen=True)
class MetalLinearProjection:
    """A fixed dense response matrix retained as a JAX device array."""

    matrix: Any

    @classmethod
    def from_matrix(cls, matrix: Any, runtime) -> "MetalLinearProjection":
        array = np.asarray(matrix, dtype=np.float32)
        if array.ndim != 2 or not np.all(np.isfinite(array)):
            raise RobertValidationError(
                "projection matrix must be finite and two-dimensional"
            )
        return cls(matrix=runtime.put(array))

    def project(self, native_values: Any):
        """Project one native spectrum, returning a device array."""

        return self.matrix @ native_values

    @property
    def native_size(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def output_size(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def device_bytes(self) -> int:
        return int(self.matrix.size * self.matrix.dtype.itemsize)


@dataclass(frozen=True)
class MetalIdentityProjection:
    """Zero-storage identity projection for native-grid observations."""

    size: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.size, int)
            or isinstance(self.size, bool)
            or self.size < 1
        ):
            raise RobertValidationError("identity projection size must be positive")

    @property
    def native_size(self) -> int:
        return self.size

    @property
    def output_size(self) -> int:
        return self.size

    @property
    def device_bytes(self) -> int:
        return 0

    def project(self, native_values: Any):
        """Return the native device array without allocating a dense matrix."""

        if native_values.shape != (self.size,):
            raise RobertValidationError(
                "native values must match identity projection size"
            )
        return native_values


@dataclass(frozen=True)
class MetalBinnedProjection:
    """Contiguous native-bin averages using prefix sums, without a matrix."""

    starts: Any
    stops: Any
    counts: Any
    size: int
    jnp: Any

    @classmethod
    def from_edges(
        cls, edges: Any, native_size: int, runtime: Any
    ) -> "MetalBinnedProjection":
        boundaries = np.asarray(edges, dtype=np.int32)
        if (
            boundaries.ndim != 1
            or boundaries.size < 2
            or boundaries[0] != 0
            or boundaries[-1] != native_size
            or np.any(np.diff(boundaries) <= 0)
        ):
            raise RobertValidationError(
                "bin edges must increase from zero through native_size"
            )
        starts = boundaries[:-1]
        stops = boundaries[1:]
        return cls(
            starts=runtime.jax.device_put(starts, runtime.device),
            stops=runtime.jax.device_put(stops, runtime.device),
            counts=runtime.put(stops - starts),
            size=int(native_size),
            jnp=runtime.jnp,
        )

    @property
    def native_size(self) -> int:
        return self.size

    @property
    def output_size(self) -> int:
        return int(self.starts.shape[0])

    @property
    def device_bytes(self) -> int:
        return int(
            self.starts.size * self.starts.dtype.itemsize
            + self.stops.size * self.stops.dtype.itemsize
            + self.counts.size * self.counts.dtype.itemsize
        )

    def project(self, native_values: Any):
        """Average consecutive native samples into configured bins."""

        if native_values.shape != (self.size,):
            raise RobertValidationError(
                "native values must match binned projection size"
            )
        cumulative = self.jnp.concatenate(
            (
                self.jnp.zeros((1,), dtype=native_values.dtype),
                self.jnp.cumsum(native_values),
            )
        )
        return (cumulative[self.stops] - cumulative[self.starts]) / self.counts


@dataclass(frozen=True)
class MetalLinearGaussianLikelihood:
    """Independent Gaussian likelihood with CPU-compatible nuisance semantics."""

    observation: Any
    uncertainty: Any
    mask: Any
    projection: MetalLinearProjection | MetalIdentityProjection | MetalBinnedProjection
    jnp: Any
    include_normalization: bool = False
    uncertainty_scale: float = 1.0
    invalid_model_loglike: float = float("-inf")
    name: str = "metal-independent-gaussian"
    coordinate_rtol: float = 0.0
    coordinate_atol: float = 0.0
    offset_parameter: str | None = None
    jitter_parameter: str | None = None

    @classmethod
    def from_arrays(
        cls,
        observation: Any,
        uncertainty: Any,
        projection: MetalLinearProjection
        | MetalIdentityProjection
        | MetalBinnedProjection,
        runtime,
        *,
        mask: Any | None = None,
        include_normalization: bool = False,
        uncertainty_scale: float = 1.0,
        invalid_model_loglike: float = float("-inf"),
    ) -> "MetalLinearGaussianLikelihood":
        observed = np.asarray(observation, dtype=np.float32)
        sigma = np.asarray(uncertainty, dtype=np.float32)
        active = (
            np.ones(observed.shape, dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool)
        )
        if (
            observed.ndim != 1
            or sigma.shape != observed.shape
            or active.shape != observed.shape
        ):
            raise RobertValidationError(
                "observation, uncertainty, and mask must be equal-length vectors"
            )
        if (
            not np.all(np.isfinite(observed))
            or not np.all(np.isfinite(sigma))
            or np.any(sigma <= 0.0)
        ):
            raise RobertValidationError(
                "observation must be finite and uncertainty positive"
            )
        if not np.any(active):
            raise RobertValidationError(
                "likelihood mask excludes all observation points"
            )
        if not np.isfinite(uncertainty_scale) or uncertainty_scale <= 0.0:
            raise RobertValidationError("uncertainty_scale must be finite and positive")
        if projection.output_size != observed.size:
            raise RobertValidationError("projection rows must match observation length")
        return cls(
            runtime.put(observed),
            runtime.put(sigma),
            runtime.jax.device_put(active, runtime.device),
            projection,
            runtime.jnp,
            bool(include_normalization),
            float(uncertainty_scale),
            float(invalid_model_loglike),
        )

    def loglike(
        self,
        native_values: Any,
        *,
        offset: Any = 0.0,
        jitter: Any = 0.0,
        uncertainty_scale_parameter: Any = 1.0,
    ):
        """Return a device scalar using the same equation as the CPU likelihood.

        Invalid *dynamic* nuisance values produce ``invalid_model_loglike`` on
        device.  Static observations, masks, and base scales are validated by
        :meth:`from_arrays` before compilation.
        """

        dtype = self.observation.dtype
        offset_value = self.jnp.asarray(offset, dtype=dtype)
        jitter_value = self.jnp.asarray(jitter, dtype=dtype)
        dynamic_scale = self.jnp.asarray(uncertainty_scale_parameter, dtype=dtype)
        scale = self.jnp.asarray(self.uncertainty_scale, dtype=dtype) * dynamic_scale
        predicted = self.projection.project(native_values) + offset_value
        effective_uncertainty = self.uncertainty * scale
        variance = effective_uncertainty**2 + jitter_value**2
        residual = self.observation - predicted
        terms = -self.jnp.asarray(0.5, dtype=dtype) * residual**2 / variance
        if self.include_normalization:
            terms = terms - self.jnp.asarray(0.5, dtype=dtype) * self.jnp.log(
                self.jnp.asarray(2.0, dtype=dtype) * self.jnp.pi * variance
            )
        value = self.jnp.sum(self.jnp.where(self.mask, terms, 0.0))
        valid = (
            self.jnp.all(self.jnp.isfinite(predicted))
            & self.jnp.isfinite(offset_value)
            & self.jnp.isfinite(jitter_value)
            & (jitter_value >= 0.0)
            & self.jnp.isfinite(scale)
            & (scale > 0.0)
            & self.jnp.isfinite(value)
        )
        return self.jnp.where(
            valid,
            value,
            self.jnp.asarray(self.invalid_model_loglike, dtype=dtype),
        )


@dataclass(frozen=True)
class MetalMultiDatasetGaussianLikelihood:
    """Ordered sum of independently projected device-resident datasets."""

    datasets: tuple[MetalLinearGaussianLikelihood, ...]

    def __post_init__(self) -> None:
        if not self.datasets:
            raise RobertValidationError("at least one Metal likelihood is required")

    def loglike(
        self,
        native_values_by_dataset: Any,
        *,
        offsets: Any,
        jitters: Any,
        uncertainty_scales: Any,
    ):
        """Sum distinct dataset spectra without leaving the device graph."""

        jnp = self.datasets[0].jnp
        if not isinstance(native_values_by_dataset, (tuple, list)) or len(
            native_values_by_dataset
        ) != len(self.datasets):
            raise RobertValidationError(
                "native_values_by_dataset must provide one spectrum per dataset"
            )
        offset_values = jnp.asarray(offsets)
        jitter_values = jnp.asarray(jitters)
        scale_values = jnp.asarray(uncertainty_scales)
        expected = (len(self.datasets),)
        if (
            offset_values.shape != expected
            or jitter_values.shape != expected
            or scale_values.shape != expected
        ):
            raise RobertValidationError(
                "dataset nuisance arrays must match the number of datasets"
            )
        terms = [
            dataset.loglike(
                native_values_by_dataset[index],
                offset=offset_values[index],
                jitter=jitter_values[index],
                uncertainty_scale_parameter=scale_values[index],
            )
            for index, dataset in enumerate(self.datasets)
        ]
        return jnp.sum(jnp.stack(terms))


__all__ = [
    "MetalLinearGaussianLikelihood",
    "MetalIdentityProjection",
    "MetalBinnedProjection",
    "MetalLinearProjection",
    "MetalMultiDatasetGaussianLikelihood",
]
