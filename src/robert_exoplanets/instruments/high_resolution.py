"""High-resolution line-spread-function responses.

The response in this module is deliberately prepared with a native spectral
grid.  Preparation builds a fixed, sparse convolution operator.  This keeps
grid searches, coverage checks, and quadrature outside a retrieval likelihood
loop.  The operator uses a Gaussian with a wavelength-dependent width, so the
full width at half maximum is ``wavelength / resolving_power`` at every output
wavelength.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.core import RobertCoverageError, RobertValidationError, SpectralGrid, Spectrum
from robert_exoplanets.core._immutability import immutable_mapping

from .observation import Observation


_GAUSSIAN_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
_QUADRATURE_ORDER = 8
_SPEED_OF_LIGHT_KM_S = 299_792.458


def _positive_float(value: float, name: str) -> float:
    """Validate and normalise one positive finite scalar."""

    if isinstance(value, (bool, np.bool_)):
        raise RobertValidationError(f"{name} must be finite and positive")
    try:
        normalised = float(value)
    except (TypeError, ValueError) as error:
        raise RobertValidationError(f"{name} must be finite and positive") from error
    if not np.isfinite(normalised) or normalised <= 0.0:
        raise RobertValidationError(f"{name} must be finite and positive")
    return normalised


def _finite_float(value: float, name: str) -> float:
    """Validate and normalise one finite scalar."""

    if isinstance(value, (bool, np.bool_)):
        raise RobertValidationError(f"{name} must be finite")
    try:
        normalised = float(value)
    except (TypeError, ValueError) as error:
        raise RobertValidationError(f"{name} must be finite") from error
    if not np.isfinite(normalised):
        raise RobertValidationError(f"{name} must be finite")
    return normalised


def _nonnegative_float(value: float, name: str) -> float:
    """Validate and normalise one finite non-negative scalar."""

    normalised = _finite_float(value, name)
    if normalised < 0.0:
        raise RobertValidationError(f"{name} must be finite and non-negative")
    return normalised


def _grid_orientation(values: NDArray[np.float64]) -> bool:
    """Return ``True`` for a descending grid."""

    return bool(values.size > 1 and values[0] > values[-1])


def _ascending_values(grid: SpectralGrid) -> NDArray[np.float64]:
    """Return a read-only spectral grid coordinate in increasing order."""

    values = np.asarray(grid.values, dtype=float)
    return values[::-1] if _grid_orientation(values) else values


def _grid_matches(left: SpectralGrid, right: SpectralGrid) -> bool:
    """Compare the physical coordinates and units of two spectral grids."""

    if left.unit != right.unit or not np.array_equal(left.values, right.values):
        return False
    if left.bin_edges is None or right.bin_edges is None:
        return left.bin_edges is None and right.bin_edges is None
    return np.array_equal(left.bin_edges, right.bin_edges)


def _coverage_tolerance(minimum: float, maximum: float) -> float:
    """Return a small coordinate tolerance for boundary round-off."""

    return 32.0 * np.finfo(float).eps * max(1.0, abs(minimum), abs(maximum))


def _require_positive_grid(grid: SpectralGrid, name: str) -> None:
    """Validate a positive spectral grid, including optional bin edges."""

    if np.any(grid.values <= 0.0):  # SpectralGrid already checks this; keep the contract explicit.
        raise RobertValidationError(f"{name} wavelengths must be positive")
    if grid.bin_edges is not None and np.any(grid.bin_edges <= 0.0):
        raise RobertValidationError(f"{name} bin edges must be positive")


def _check_coordinate_coverage(
    source: NDArray[np.float64],
    requested: NDArray[np.float64],
    *,
    message: str,
) -> None:
    """Reject requests that require interpolation outside source coverage."""

    minimum = float(source[0])
    maximum = float(source[-1])
    tolerance = _coverage_tolerance(minimum, maximum)
    if requested[0] < minimum - tolerance or requested[-1] > maximum + tolerance:
        raise RobertCoverageError(message)


def _build_linear_interpolation_operator(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
    """Build a fixed piecewise-linear interpolation operator."""

    _check_coordinate_coverage(
        source,
        target,
        message="requested response grid extends outside native spectrum",
    )
    indices: list[int] = []
    weights: list[float] = []
    row_starts = [0]
    for coordinate in target:
        insertion = int(np.searchsorted(source, coordinate, side="left"))
        if insertion < source.size and coordinate == source[insertion]:
            indices.append(insertion)
            weights.append(1.0)
        elif insertion <= 0 or insertion >= source.size:
            raise RobertCoverageError("requested response grid extends outside native spectrum")
        else:
            left = insertion - 1
            right = insertion
            fraction = (coordinate - source[left]) / (source[right] - source[left])
            indices.extend((left, right))
            weights.extend((1.0 - float(fraction), float(fraction)))
        row_starts.append(len(indices))
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(weights, dtype=float),
        np.asarray(row_starts, dtype=np.int64),
    )


@dataclass(frozen=True)
class PreparedSpectralOperator:
    """Prepared sparse linear operator between two spectral grids.

    The operator stores all interpolation or convolution weights.  A model
    evaluation only performs the fixed sparse matrix-vector product.
    """

    source_grid: SpectralGrid
    target_grid: SpectralGrid
    indices: NDArray[np.int64]
    weights: NDArray[np.float64]
    row_starts: NDArray[np.int64]
    name: str = "prepared-spectral-operator"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_grid.unit != self.target_grid.unit:
            raise RobertValidationError("source and target spectral units must match")
        _require_positive_grid(self.source_grid, "source")
        _require_positive_grid(self.target_grid, "target")
        if not self.name:
            raise RobertValidationError("name must not be empty")
        indices = np.array(self.indices, dtype=np.int64, copy=True)
        weights = np.array(self.weights, dtype=float, copy=True)
        row_starts = np.array(self.row_starts, dtype=np.int64, copy=True)
        if indices.ndim != 1 or weights.ndim != 1 or row_starts.ndim != 1:
            raise RobertValidationError("prepared operator arrays must be one-dimensional")
        if indices.size != weights.size:
            raise RobertValidationError("operator indices and weights must have matching sizes")
        if row_starts.size != self.target_grid.size + 1:
            raise RobertValidationError("operator row_starts must contain n_target + 1 values")
        if (
            row_starts[0] != 0
            or row_starts[-1] != indices.size
            or np.any(np.diff(row_starts) <= 0)
        ):
            raise RobertValidationError("each prepared operator row must contain weights")
        if np.any(indices < 0) or np.any(indices >= self.source_grid.size):
            raise RobertValidationError("operator indices exceed the source grid")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise RobertValidationError("operator weights must be finite and positive")
        indices.setflags(write=False)
        weights.setflags(write=False)
        row_starts.setflags(write=False)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "row_starts", row_starts)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def native_grid(self) -> SpectralGrid:
        """Alias for the source grid."""

        return self.source_grid

    def observe(self, spectrum: Spectrum) -> Spectrum:
        """Apply the prepared operator to one model spectrum."""

        if spectrum.spectral_grid.unit != self.source_grid.unit:
            raise RobertValidationError("spectral units must match before response mapping")
        if not np.array_equal(spectrum.spectral_grid.values, self.source_grid.values):
            raise RobertValidationError("spectrum grid does not match the prepared source grid")
        source_values = np.asarray(spectrum.values, dtype=float)
        if _grid_orientation(self.source_grid.values):
            source_values = source_values[::-1]

        target_values = np.empty(self.target_grid.size, dtype=float)
        for row in range(self.target_grid.size):
            start = int(self.row_starts[row])
            stop = int(self.row_starts[row + 1])
            target_values[row] = np.dot(
                self.weights[start:stop],
                source_values[self.indices[start:stop]],
            )
        if _grid_orientation(self.target_grid.values):
            target_values = target_values[::-1]

        output_metadata = dict(spectrum.metadata)
        output_metadata.update(self.metadata)
        output_metadata["response"] = self.name
        return Spectrum(
            spectral_grid=self.target_grid,
            values=target_values,
            unit=spectrum.unit,
            observable=spectrum.observable,
            metadata=output_metadata,
        )

    apply = observe


def _build_gaussian_operator(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    resolving_power: float,
    kernel_support: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
    """Build a sparse operator for Gaussian convolution and interpolation.

    Each target row integrates a piecewise-linear source spectrum over the
    finite Gaussian support.  The row is normalised after integration.  This
    preserves a constant spectrum even when source points are not uniformly
    spaced, and it avoids evaluating an interpolator during retrieval.
    """

    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(
        _QUADRATURE_ORDER
    )
    source_size = source.size
    indices: list[int] = []
    weights: list[float] = []
    row_starts = [0]

    for wavelength in target:
        sigma = wavelength * _GAUSSIAN_FWHM_TO_SIGMA / resolving_power
        support_half_width = kernel_support * sigma
        lower = wavelength - support_half_width
        upper = wavelength + support_half_width

        # Coverage is checked by the caller.  Clipping the values here only
        # removes round-off at a native-grid boundary.
        lower = max(lower, float(source[0]))
        upper = min(upper, float(source[-1]))

        first_interval = max(
            0,
            int(np.searchsorted(source, lower, side="right")) - 1,
        )
        last_interval = min(
            source_size - 2,
            int(np.searchsorted(source, upper, side="left")),
        )
        row_coefficients: dict[int, float] = {}

        for interval in range(first_interval, last_interval + 1):
            interval_lower = max(float(source[interval]), lower)
            interval_upper = min(float(source[interval + 1]), upper)
            interval_width = interval_upper - interval_lower
            if interval_width <= 0.0:
                continue

            nodes = 0.5 * interval_width * quadrature_nodes + 0.5 * (
                interval_upper + interval_lower
            )
            integration_weights = 0.5 * interval_width * quadrature_weights
            kernel = np.exp(-0.5 * ((nodes - wavelength) / sigma) ** 2)
            source_interval_width = source[interval + 1] - source[interval]
            left_interpolation = (source[interval + 1] - nodes) / source_interval_width
            right_interpolation = (nodes - source[interval]) / source_interval_width
            contribution = integration_weights * kernel

            row_coefficients[interval] = row_coefficients.get(interval, 0.0) + float(
                np.sum(contribution * left_interpolation)
            )
            row_coefficients[interval + 1] = row_coefficients.get(interval + 1, 0.0) + float(
                np.sum(contribution * right_interpolation)
            )

        normalisation = float(sum(row_coefficients.values()))
        if not np.isfinite(normalisation) or normalisation <= 0.0:
            raise RobertCoverageError(
                "native high-resolution grid has no samples inside the Gaussian kernel support"
            )

        for index in sorted(row_coefficients):
            coefficient = row_coefficients[index] / normalisation
            if coefficient > 0.0:
                indices.append(index)
                weights.append(coefficient)
        row_starts.append(len(indices))

    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(weights, dtype=float),
        np.asarray(row_starts, dtype=np.int64),
    )


def _trim_gaussian_target_grid(
    source_grid: SpectralGrid,
    resolving_power: float,
    kernel_support: float,
) -> SpectralGrid:
    """Build a source-grid subset with complete Gaussian support."""

    source = _ascending_values(source_grid)
    source_min = float(source[0])
    source_max = float(source[-1])
    sigma = source * _GAUSSIAN_FWHM_TO_SIGMA / resolving_power
    half_width = kernel_support * sigma
    valid = (source - half_width >= source_min) & (source + half_width <= source_max)
    if not np.any(valid):
        raise RobertCoverageError(
            "native high-resolution grid has no points with complete Gaussian support"
        )
    target = source[valid]
    if _grid_orientation(source_grid.values):
        target = target[::-1]
    return SpectralGrid(
        values=target,
        unit=source_grid.unit,
        name=source_grid.name,
        role="high_resolution_lsf",
    )


def _prepare_gaussian_operator_on_grids(
    source_grid: SpectralGrid,
    target_grid: SpectralGrid,
    resolving_power: float,
    kernel_support: float,
    *,
    name: str,
) -> PreparedSpectralOperator:
    """Prepare a Gaussian operator between explicit grids."""

    if source_grid.unit != target_grid.unit:
        raise RobertValidationError("native and target spectral units must match")
    _require_positive_grid(source_grid, "source")
    _require_positive_grid(target_grid, "target")
    source = _ascending_values(source_grid)
    target = _ascending_values(target_grid)
    source_min = float(source[0])
    source_max = float(source[-1])
    tolerance = _coverage_tolerance(source_min, source_max)
    for wavelength in target:
        sigma = wavelength * _GAUSSIAN_FWHM_TO_SIGMA / resolving_power
        support_half_width = kernel_support * sigma
        if (
            wavelength - support_half_width < source_min - tolerance
            or wavelength + support_half_width > source_max + tolerance
        ):
            raise RobertCoverageError(
                "target wavelengths plus Gaussian kernel support extend outside native spectrum"
            )
    if target_grid.bin_edges is not None:
        edges = _ascending_values(
            SpectralGrid(
                values=target_grid.bin_edges,
                unit=target_grid.unit,
                role="target_bin_edges",
            )
        )
        if edges[0] < source_min - tolerance or edges[-1] > source_max + tolerance:
            raise RobertCoverageError("target bins extend outside native spectrum")
    indices, weights, row_starts = _build_gaussian_operator(
        source,
        target,
        resolving_power,
        kernel_support,
    )
    return PreparedSpectralOperator(
        source_grid=source_grid,
        target_grid=target_grid,
        indices=indices,
        weights=weights,
        row_starts=row_starts,
        name=name,
        metadata={
            "resolving_power": f"{resolving_power:g}",
            "kernel_support_sigma": f"{kernel_support:g}",
            "convolution": "piecewise_linear_gaussian",
        },
    )


def _build_log_kernel_operator(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    half_width: float,
    profile,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
    """Build a compact kernel operator in a log-wavelength coordinate.

    ``source`` and ``target`` are increasing log-wavelength coordinates.
    ``profile`` receives velocity-like offsets in the same units as
    ``half_width`` and returns a non-negative kernel density.  The source
    spectrum is treated as piecewise linear in log wavelength.
    """

    nodes, quadrature_weights = np.polynomial.legendre.leggauss(_QUADRATURE_ORDER)
    indices: list[int] = []
    weights: list[float] = []
    row_starts = [0]
    for coordinate in target:
        lower = coordinate - half_width
        upper = coordinate + half_width
        first_interval = max(0, int(np.searchsorted(source, lower, side="right")) - 1)
        last_interval = min(
            source.size - 2,
            int(np.searchsorted(source, upper, side="left")),
        )
        row_coefficients: dict[int, float] = {}
        for interval in range(first_interval, last_interval + 1):
            interval_lower = max(float(source[interval]), lower)
            interval_upper = min(float(source[interval + 1]), upper)
            interval_width = interval_upper - interval_lower
            if interval_width <= 0.0:
                continue
            sample = 0.5 * interval_width * nodes + 0.5 * (
                interval_upper + interval_lower
            )
            integration_weights = 0.5 * interval_width * quadrature_weights
            kernel = np.asarray(profile(sample - coordinate), dtype=float)
            if np.any(kernel < 0.0) or not np.all(np.isfinite(kernel)):
                raise RobertValidationError("rotational kernel values must be finite and non-negative")
            source_interval_width = source[interval + 1] - source[interval]
            left_interpolation = (source[interval + 1] - sample) / source_interval_width
            right_interpolation = (sample - source[interval]) / source_interval_width
            contribution = integration_weights * kernel
            row_coefficients[interval] = row_coefficients.get(interval, 0.0) + float(
                np.sum(contribution * left_interpolation)
            )
            row_coefficients[interval + 1] = row_coefficients.get(interval + 1, 0.0) + float(
                np.sum(contribution * right_interpolation)
            )
        normalisation = float(sum(row_coefficients.values()))
        if not np.isfinite(normalisation) or normalisation <= 0.0:
            raise RobertCoverageError(
                "native grid has no samples inside the compact velocity kernel support"
            )
        for index in sorted(row_coefficients):
            coefficient = row_coefficients[index] / normalisation
            if coefficient > 0.0:
                indices.append(index)
                weights.append(coefficient)
        row_starts.append(len(indices))
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(weights, dtype=float),
        np.asarray(row_starts, dtype=np.int64),
    )


def _build_bin_integration_operator(
    source: NDArray[np.float64],
    edges: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
    """Build a fixed, flux-conserving piecewise-linear bin operator."""

    nodes, quadrature_weights = np.polynomial.legendre.leggauss(_QUADRATURE_ORDER)
    indices: list[int] = []
    weights: list[float] = []
    row_starts = [0]
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        first_interval = max(0, int(np.searchsorted(source, lower, side="right")) - 1)
        last_interval = min(
            source.size - 2,
            int(np.searchsorted(source, upper, side="left")),
        )
        row_coefficients: dict[int, float] = {}
        for interval in range(first_interval, last_interval + 1):
            interval_lower = max(float(source[interval]), float(lower))
            interval_upper = min(float(source[interval + 1]), float(upper))
            interval_width = interval_upper - interval_lower
            if interval_width <= 0.0:
                continue
            sample = 0.5 * interval_width * nodes + 0.5 * (
                interval_upper + interval_lower
            )
            integration_weights = 0.5 * interval_width * quadrature_weights
            source_interval_width = source[interval + 1] - source[interval]
            left_interpolation = (source[interval + 1] - sample) / source_interval_width
            right_interpolation = (sample - source[interval]) / source_interval_width
            row_coefficients[interval] = row_coefficients.get(interval, 0.0) + float(
                np.sum(integration_weights * left_interpolation)
            )
            row_coefficients[interval + 1] = row_coefficients.get(interval + 1, 0.0) + float(
                np.sum(integration_weights * right_interpolation)
            )
        normalisation = float(sum(row_coefficients.values()))
        if not np.isfinite(normalisation) or normalisation <= 0.0:
            raise RobertCoverageError("native grid has no samples inside an observation bin")
        # The row is a bin average.  Normalisation is the numerically
        # integrated bin width, so a constant flux remains exactly constant.
        for index in sorted(row_coefficients):
            coefficient = row_coefficients[index] / normalisation
            if coefficient > 0.0:
                indices.append(index)
                weights.append(coefficient)
        row_starts.append(len(indices))
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(weights, dtype=float),
        np.asarray(row_starts, dtype=np.int64),
    )


def _rotational_profile(
    velocity_offset: NDArray[np.float64],
    projected_velocity: float,
    limb_darkening: float,
) -> NDArray[np.float64]:
    """Return the Gray rotational kernel with linear limb darkening.

    For ``|v| <= v_sini``, the profile is

    ``[2(1-eps)*sqrt(1-u^2) + pi*eps/2*(1-u^2)]``
    ``/ [pi*v_sini*(1-eps/3)]``, where ``u = v/v_sini``.
    """

    scaled = np.asarray(velocity_offset, dtype=float) / projected_velocity
    inside = np.abs(scaled) <= 1.0
    profile = np.zeros_like(scaled, dtype=float)
    one_minus_u2 = np.maximum(0.0, 1.0 - scaled[inside] ** 2)
    numerator = (
        2.0 * (1.0 - limb_darkening) * np.sqrt(one_minus_u2)
        + 0.5 * np.pi * limb_darkening * one_minus_u2
    )
    denominator = np.pi * projected_velocity * (1.0 - limb_darkening / 3.0)
    profile[inside] = numerator / denominator
    return profile


@dataclass(frozen=True)
class GaussianHighResolutionResponse:
    """Gaussian line-spread response with constant resolving power.

    ``resolving_power`` defines the Gaussian FWHM as
    ``wavelength / resolving_power``.  ``kernel_support`` is the number of
    Gaussian standard deviations retained on each side of an output point.
    The native grid must cover this complete support for every requested
    observation wavelength.
    """

    resolving_power: float
    kernel_support: float = 4.0
    name: str = "gaussian-high-resolution"

    def __post_init__(self) -> None:
        resolving_power = _positive_float(self.resolving_power, "resolving_power")
        kernel_support = _positive_float(self.kernel_support, "kernel_support")
        if not self.name:
            raise RobertValidationError("name must not be empty")
        object.__setattr__(self, "resolving_power", resolving_power)
        object.__setattr__(self, "kernel_support", kernel_support)

    def prepare_on_grid(
        self,
        native_grid: SpectralGrid,
        target_grid: SpectralGrid | None = None,
    ) -> PreparedSpectralOperator:
        """Prepare a Gaussian operator between explicit spectral grids.

        When ``target_grid`` is omitted, the response uses the subset of the
        native grid for which the complete finite Gaussian support is covered.
        This mode is used by a response chain before its final pixel mapping.
        """

        if native_grid.size < 2:
            raise RobertValidationError(
                "native high-resolution grid must contain at least two points"
            )
        _require_positive_grid(native_grid, "native")
        target = (
            _trim_gaussian_target_grid(
                native_grid,
                self.resolving_power,
                self.kernel_support,
            )
            if target_grid is None
            else target_grid
        )
        return _prepare_gaussian_operator_on_grids(
            native_grid,
            target,
            self.resolving_power,
            self.kernel_support,
            name=self.name,
        )

    def prepare(
        self,
        observation: Observation,
        native_grid: SpectralGrid | None = None,
        *,
        source_grid: SpectralGrid | None = None,
    ) -> "PreparedGaussianHighResolutionResponse":
        """Prepare a fixed response from a native grid to an observation.

        ``native_grid`` is required because the convolution operator depends
        on the source coordinates.  ``source_grid`` is accepted as a named
        alias for callers that use source terminology.  The returned object
        can be reused for every trial spectrum on that exact native grid.
        """

        if native_grid is None:
            native_grid = source_grid
        elif source_grid is not None and source_grid is not native_grid:
            raise RobertValidationError("native_grid and source_grid must not both be provided")
        if native_grid is None:
            raise RobertValidationError("native_grid is required when preparing a Gaussian response")
        if native_grid.unit != observation.wavelength_unit:
            raise RobertValidationError("native and observation spectral units must match")
        if native_grid.size < 2:
            raise RobertValidationError(
                "native high-resolution grid must contain at least two points"
            )

        # Accessing this property validates positive observed wavelengths and
        # positive, correctly oriented bin edges through SpectralGrid.
        observation_grid = observation.spectral_grid
        source = np.asarray(native_grid.values, dtype=float)
        target = np.asarray(observation_grid.values, dtype=float)
        source_descending = bool(source[0] > source[-1])
        target_descending = bool(target[0] > target[-1])
        source_ascending = source[::-1] if source_descending else source
        target_ascending = target[::-1] if target_descending else target

        source_min = float(source_ascending[0])
        source_max = float(source_ascending[-1])
        tolerance = 32.0 * np.finfo(float).eps * max(1.0, source_max)

        for wavelength in target_ascending:
            sigma = wavelength * _GAUSSIAN_FWHM_TO_SIGMA / self.resolving_power
            support_half_width = self.kernel_support * sigma
            lower = wavelength - support_half_width
            upper = wavelength + support_half_width
            if lower < source_min - tolerance or upper > source_max + tolerance:
                raise RobertCoverageError(
                    "observation wavelengths plus Gaussian kernel support extend outside native spectrum"
                )

        if observation_grid.bin_edges is not None:
            edges = np.asarray(observation_grid.bin_edges, dtype=float)
            if edges[0] > edges[-1]:
                edges = edges[::-1]
            if edges[0] < source_min - tolerance or edges[-1] > source_max + tolerance:
                raise RobertCoverageError("observation bins extend outside native spectrum")

        indices, weights, row_starts = _build_gaussian_operator(
            source_ascending,
            target_ascending,
            self.resolving_power,
            self.kernel_support,
        )
        return PreparedGaussianHighResolutionResponse(
            observation=observation,
            source_grid=native_grid,
            kernel_indices=indices,
            kernel_weights=weights,
            row_starts=row_starts,
            resolving_power=self.resolving_power,
            kernel_support=self.kernel_support,
            source_descending=source_descending,
            target_descending=target_descending,
            name=self.name,
        )


@dataclass(frozen=True)
class PreparedGaussianHighResolutionResponse:
    """Prepared Gaussian response that can be reused in a likelihood loop."""

    observation: Observation
    source_grid: SpectralGrid
    kernel_indices: NDArray[np.int64]
    kernel_weights: NDArray[np.float64]
    row_starts: NDArray[np.int64]
    resolving_power: float
    kernel_support: float
    source_descending: bool = False
    target_descending: bool = False
    name: str = "gaussian-high-resolution"

    def __post_init__(self) -> None:
        if self.source_grid.unit != self.observation.wavelength_unit:
            raise RobertValidationError("native and observation spectral units must match")
        if self.source_grid.size < 2:
            raise RobertValidationError(
                "native high-resolution grid must contain at least two points"
            )
        resolving_power = _positive_float(self.resolving_power, "resolving_power")
        kernel_support = _positive_float(self.kernel_support, "kernel_support")
        if not self.name:
            raise RobertValidationError("name must not be empty")

        indices = np.array(self.kernel_indices, dtype=np.int64, copy=True)
        weights = np.array(self.kernel_weights, dtype=float, copy=True)
        row_starts = np.array(self.row_starts, dtype=np.int64, copy=True)
        if indices.ndim != 1 or weights.ndim != 1 or row_starts.ndim != 1:
            raise RobertValidationError("prepared Gaussian operator arrays must be one-dimensional")
        if indices.size != weights.size:
            raise RobertValidationError("Gaussian operator indices and weights must have matching sizes")
        if row_starts.size != self.observation.n_points + 1:
            raise RobertValidationError(
                "Gaussian operator row_starts must contain n_points + 1 values"
            )
        if row_starts[0] != 0 or row_starts[-1] != indices.size or np.any(np.diff(row_starts) <= 0):
            raise RobertValidationError("Gaussian operator rows must contain at least one weight")
        if np.any(indices < 0) or np.any(indices >= self.source_grid.size):
            raise RobertValidationError("Gaussian operator indices exceed the native grid")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise RobertValidationError("Gaussian operator weights must be finite and positive")

        indices.setflags(write=False)
        weights.setflags(write=False)
        row_starts.setflags(write=False)
        object.__setattr__(self, "kernel_indices", indices)
        object.__setattr__(self, "kernel_weights", weights)
        object.__setattr__(self, "row_starts", row_starts)
        object.__setattr__(self, "resolving_power", resolving_power)
        object.__setattr__(self, "kernel_support", kernel_support)

    @property
    def native_grid(self) -> SpectralGrid:
        """Alias for the fixed native grid used by this prepared response."""

        return self.source_grid

    def observe(self, spectrum: Spectrum) -> Spectrum:
        """Convolve a native spectrum and map it to the observation grid."""

        if spectrum.spectral_grid.unit != self.source_grid.unit:
            raise RobertValidationError("spectral units must match before response mapping")
        if not np.array_equal(spectrum.spectral_grid.values, self.source_grid.values):
            raise RobertValidationError(
                "spectrum grid does not match the prepared native high-resolution grid"
            )
        if spectrum.unit != self.observation.flux_unit:
            raise RobertValidationError("spectrum and observation value units must match")
        if spectrum.observable != self.observation.observable:
            raise RobertValidationError("spectrum and observation observables must match")

        source_values = np.asarray(spectrum.values, dtype=float)
        source_values_ascending = source_values[::-1] if self.source_descending else source_values
        values_ascending = np.empty(self.observation.n_points, dtype=float)
        for row in range(self.observation.n_points):
            start = int(self.row_starts[row])
            stop = int(self.row_starts[row + 1])
            values_ascending[row] = np.dot(
                self.kernel_weights[start:stop],
                source_values_ascending[self.kernel_indices[start:stop]],
            )
        values = values_ascending[::-1] if self.target_descending else values_ascending

        return Spectrum(
            spectral_grid=self.observation.spectral_grid,
            values=values,
            unit=spectrum.unit,
            observable=spectrum.observable,
            metadata={
                "response": self.name,
                "instrument": self.observation.instrument or "",
                "resolving_power": f"{self.resolving_power:g}",
                "kernel_support_sigma": f"{self.kernel_support:g}",
                "convolution": "piecewise_linear_gaussian",
            },
        )


def _shifted_grid(source_grid: SpectralGrid, doppler_factor: float) -> SpectralGrid:
    """Return a point-sampled working grid after a Doppler shift.

    Native bin edges describe the input sampling. They are not output pixel
    edges and must not constrain later convolution stages. The final pixel
    response uses the explicit observation bin edges.
    """

    return SpectralGrid(
        values=np.asarray(source_grid.values, dtype=float) * doppler_factor,
        unit=source_grid.unit,
        name=source_grid.name,
        role="doppler_shifted",
        metadata={"doppler_factor": f"{doppler_factor:.16g}"},
    )


def _prepare_doppler_operator(
    source_grid: SpectralGrid,
    target_grid: SpectralGrid,
    doppler_factor: float,
    *,
    name: str,
) -> PreparedSpectralOperator:
    """Prepare Doppler interpolation from rest to observed wavelengths."""

    if source_grid.unit != target_grid.unit:
        raise RobertValidationError("source and target spectral units must match")
    _require_positive_grid(source_grid, "source")
    _require_positive_grid(target_grid, "target")
    shifted_source = _ascending_values(source_grid) * doppler_factor
    target = _ascending_values(target_grid)
    _check_coordinate_coverage(
        shifted_source,
        target,
        message="Doppler-shifted native spectrum does not cover the target grid",
    )
    if target_grid.bin_edges is not None:
        edges = np.asarray(target_grid.bin_edges, dtype=float)
        if edges[0] > edges[-1]:
            edges = edges[::-1]
        _check_coordinate_coverage(
            shifted_source,
            edges,
            message="Doppler-shifted native spectrum does not cover target bin edges",
        )
    indices, weights, row_starts = _build_linear_interpolation_operator(
        shifted_source,
        target,
    )
    return PreparedSpectralOperator(
        source_grid=source_grid,
        target_grid=target_grid,
        indices=indices,
        weights=weights,
        row_starts=row_starts,
        name=name,
        metadata={
            "doppler_factor": f"{doppler_factor:.16g}",
            "doppler_formula": "sqrt((1+beta)/(1-beta))",
        },
    )


@dataclass(frozen=True, init=False)
class RelativisticDopplerResponse:
    """Relativistic wavelength shift with velocity in km/s.

    Positive velocity means recession and redshift.  For
    ``beta = velocity / c`` the wavelength factor is
    ``D = sqrt((1 + beta) / (1 - beta))``.  The response evaluates the
    rest-frame spectrum at shifted wavelengths by fixed interpolation.
    """

    velocity_km_s: float
    name: str = "relativistic-doppler"

    def __init__(
        self,
        velocity_km_s: float | None = None,
        *,
        velocity: float | None = None,
        radial_velocity_km_s: float | None = None,
        name: str = "relativistic-doppler",
    ) -> None:
        supplied = [
            value
            for value in (velocity_km_s, velocity, radial_velocity_km_s)
            if value is not None
        ]
        if len(supplied) > 1:
            raise RobertValidationError(
                "provide only one of velocity_km_s, velocity, or radial_velocity_km_s"
            )
        value = 0.0 if not supplied else supplied[0]
        normalised_velocity = _finite_float(value, "velocity_km_s")
        if abs(normalised_velocity) >= _SPEED_OF_LIGHT_KM_S:
            raise RobertValidationError("absolute velocity must be smaller than the speed of light")
        if not name:
            raise RobertValidationError("name must not be empty")
        object.__setattr__(self, "velocity_km_s", normalised_velocity)
        object.__setattr__(self, "name", name)

    @property
    def velocity(self) -> float:
        """Velocity in km/s."""

        return self.velocity_km_s

    @property
    def beta(self) -> float:
        """Dimensionless velocity ``v/c``."""

        return self.velocity_km_s / _SPEED_OF_LIGHT_KM_S

    @property
    def doppler_factor(self) -> float:
        """Relativistic wavelength multiplier."""

        return float(np.sqrt((1.0 + self.beta) / (1.0 - self.beta)))

    def prepare_on_grid(
        self,
        native_grid: SpectralGrid,
        target_grid: SpectralGrid | None = None,
    ) -> "PreparedRelativisticDopplerResponse":
        """Prepare a shift to a target grid or to a shifted native grid."""

        if target_grid is None:
            target_grid = _shifted_grid(native_grid, self.doppler_factor)
        operator = _prepare_doppler_operator(
            native_grid,
            target_grid,
            self.doppler_factor,
            name=self.name,
        )
        return PreparedRelativisticDopplerResponse(
            source_grid=native_grid,
            target_grid=target_grid,
            operator=operator,
            velocity_km_s=self.velocity_km_s,
            doppler_factor=self.doppler_factor,
            name=self.name,
        )

    def prepare(
        self,
        observation_or_grid: Observation | SpectralGrid,
        native_grid: SpectralGrid | None = None,
    ) -> "PreparedRelativisticDopplerResponse":
        """Prepare a shift for an observation or a shifted native grid."""

        if isinstance(observation_or_grid, Observation):
            if native_grid is None:
                raise RobertValidationError("native_grid is required for an observation response")
            prepared = self.prepare_on_grid(native_grid, observation_or_grid.spectral_grid)
            object.__setattr__(prepared, "observation", observation_or_grid)
            return prepared
        if not isinstance(observation_or_grid, SpectralGrid) or native_grid is not None:
            raise RobertValidationError("prepare expects a SpectralGrid or an Observation plus native_grid")
        return self.prepare_on_grid(observation_or_grid)


@dataclass(frozen=True)
class PreparedRelativisticDopplerResponse:
    """Prepared relativistic Doppler operator."""

    source_grid: SpectralGrid
    target_grid: SpectralGrid
    operator: PreparedSpectralOperator
    velocity_km_s: float
    doppler_factor: float
    name: str = "relativistic-doppler"
    observation: Observation | None = None

    def __post_init__(self) -> None:
        if not _grid_matches(self.source_grid, self.operator.source_grid):
            raise RobertValidationError("Doppler operator source grid does not match response")
        if not _grid_matches(self.target_grid, self.operator.target_grid):
            raise RobertValidationError("Doppler operator target grid does not match response")
        velocity = _finite_float(self.velocity_km_s, "velocity_km_s")
        if abs(velocity) >= _SPEED_OF_LIGHT_KM_S:
            raise RobertValidationError("absolute velocity must be smaller than the speed of light")
        factor = _positive_float(self.doppler_factor, "doppler_factor")
        if not self.name:
            raise RobertValidationError("name must not be empty")
        expected = float(
            np.sqrt(
                (1.0 + velocity / _SPEED_OF_LIGHT_KM_S)
                / (1.0 - velocity / _SPEED_OF_LIGHT_KM_S)
            )
        )
        if not np.isclose(factor, expected, rtol=1.0e-12, atol=0.0):
            raise RobertValidationError("doppler_factor does not match velocity_km_s")
        object.__setattr__(self, "velocity_km_s", velocity)
        object.__setattr__(self, "doppler_factor", factor)
        if self.observation is not None:
            if self.observation.wavelength_unit != self.target_grid.unit:
                raise RobertValidationError("Doppler observation and target units must match")

    @property
    def native_grid(self) -> SpectralGrid:
        """Alias for the source grid."""

        return self.source_grid

    def observe(self, spectrum: Spectrum) -> Spectrum:
        """Apply the prepared Doppler shift."""

        if self.observation is not None:
            if spectrum.unit != self.observation.flux_unit:
                raise RobertValidationError("spectrum and observation value units must match")
            if spectrum.observable != self.observation.observable:
                raise RobertValidationError("spectrum and observation observables must match")
        return self.operator.observe(spectrum)

    apply = observe


def _trim_rotational_target_grid(
    source_grid: SpectralGrid,
    projected_velocity_km_s: float,
    velocity_step_km_s: float | None,
) -> SpectralGrid:
    """Build a constant-velocity grid with complete rotational support."""

    if projected_velocity_km_s == 0.0:
        return source_grid
    source = _ascending_values(source_grid)
    log_source = np.log(source)
    half_width = projected_velocity_km_s / _SPEED_OF_LIGHT_KM_S
    lower = float(log_source[0]) + half_width
    upper = float(log_source[-1]) - half_width
    if lower >= upper:
        raise RobertCoverageError(
            "native grid is too narrow for the requested rotational broadening"
        )
    if velocity_step_km_s is None:
        native_steps = np.diff(log_source) * _SPEED_OF_LIGHT_KM_S
        step = float(np.min(native_steps))
    else:
        step = velocity_step_km_s
    count = max(2, int(np.floor((upper - lower) / (step / _SPEED_OF_LIGHT_KM_S))) + 1)
    log_target = np.linspace(lower, upper, count)
    target = np.exp(log_target)
    if _grid_orientation(source_grid.values):
        target = target[::-1]
    return SpectralGrid(
        values=target,
        unit=source_grid.unit,
        name=source_grid.name,
        role="constant_velocity_rotational",
    )


def _prepare_rotational_operator(
    source_grid: SpectralGrid,
    target_grid: SpectralGrid,
    projected_velocity_km_s: float,
    limb_darkening: float,
    *,
    name: str,
) -> PreparedSpectralOperator:
    """Prepare rotational broadening using log-wavelength quadrature."""

    if source_grid.unit != target_grid.unit:
        raise RobertValidationError("source and target spectral units must match")
    _require_positive_grid(source_grid, "source")
    _require_positive_grid(target_grid, "target")
    source = _ascending_values(source_grid)
    target = _ascending_values(target_grid)
    source_log = np.log(source)
    target_log = np.log(target)
    half_width = projected_velocity_km_s / _SPEED_OF_LIGHT_KM_S
    if projected_velocity_km_s > 0.0:
        _check_coordinate_coverage(
            source_log,
            target_log - half_width,
            message="target wavelengths plus rotational broadening extend outside native spectrum",
        )
        _check_coordinate_coverage(
            source_log,
            target_log + half_width,
            message="target wavelengths plus rotational broadening extend outside native spectrum",
        )
        if target_grid.bin_edges is not None:
            edges = np.asarray(target_grid.bin_edges, dtype=float)
            if edges[0] > edges[-1]:
                edges = edges[::-1]
            log_edges = np.log(edges)
            _check_coordinate_coverage(
                source_log,
                log_edges - half_width,
                message="target bins plus rotational broadening extend outside native spectrum",
            )
            _check_coordinate_coverage(
                source_log,
                log_edges + half_width,
                message="target bins plus rotational broadening extend outside native spectrum",
            )
    if projected_velocity_km_s == 0.0:
        indices, weights, row_starts = _build_linear_interpolation_operator(
            source,
            target,
        )
    else:
        indices, weights, row_starts = _build_log_kernel_operator(
            source_log,
            target_log,
            half_width,
            lambda offset: _rotational_profile(
                _SPEED_OF_LIGHT_KM_S * np.asarray(offset, dtype=float),
                projected_velocity_km_s,
                limb_darkening,
            ),
        )
    return PreparedSpectralOperator(
        source_grid=source_grid,
        target_grid=target_grid,
        indices=indices,
        weights=weights,
        row_starts=row_starts,
        name=name,
        metadata={
            "projected_velocity_km_s": f"{projected_velocity_km_s:g}",
            "limb_darkening": f"{limb_darkening:g}",
            "rotation_kernel": "gray_linear_limb_darkening",
            "rotation_coordinate": "ln_wavelength_constant_velocity",
        },
    )


@dataclass(frozen=True, init=False)
class RotationalBroadeningResponse:
    """Rotational broadening on a constant-velocity log-wavelength grid.

    ``projected_velocity_km_s`` is the projected equatorial velocity
    ``v sin(i)``.  ``limb_darkening`` is the linear coefficient ``epsilon`` in
    the range [0, 1].  The normalized Gray kernel is

    ``G(v) = [2(1-epsilon)*sqrt(1-u^2) + pi*epsilon/2*(1-u^2)]``
    ``/ [pi*v_sini*(1-epsilon/3)]``, ``u = v/v_sini``,
    for ``|u| <= 1`` and zero otherwise.  Positive and negative velocity
    offsets are symmetric.  The prepared operator integrates in
    ``x = ln(wavelength)`` where ``dv = c dx``.  If no step is supplied, the
    smallest native velocity step sets the constant-velocity output grid.
    """

    projected_velocity_km_s: float
    limb_darkening: float = 0.6
    velocity_step_km_s: float | None = None
    name: str = "rotational-broadening"

    def __init__(
        self,
        projected_velocity_km_s: float | None = None,
        limb_darkening: float = 0.6,
        velocity_step_km_s: float | None = None,
        *,
        vsini_km_s: float | None = None,
        velocity_km_s: float | None = None,
        vsini: float | None = None,
        linear_limb_darkening: float | None = None,
        name: str = "rotational-broadening",
    ) -> None:
        velocity_values = [
            value
            for value in (
                projected_velocity_km_s,
                vsini_km_s,
                velocity_km_s,
                vsini,
            )
            if value is not None
        ]
        if len(velocity_values) > 1:
            raise RobertValidationError(
                "provide only one projected rotational velocity alias"
            )
        velocity = 0.0 if not velocity_values else velocity_values[0]
        projected_velocity = _nonnegative_float(velocity, "projected_velocity_km_s")
        if linear_limb_darkening is not None:
            if limb_darkening != 0.6:
                raise RobertValidationError(
                    "provide only one of limb_darkening and linear_limb_darkening"
                )
            limb_darkening = linear_limb_darkening
        epsilon = _finite_float(limb_darkening, "limb_darkening")
        if not 0.0 <= epsilon <= 1.0:
            raise RobertValidationError("limb_darkening must be between 0 and 1")
        if velocity_step_km_s is not None:
            velocity_step = _positive_float(velocity_step_km_s, "velocity_step_km_s")
        else:
            velocity_step = None
        if not name:
            raise RobertValidationError("name must not be empty")
        object.__setattr__(self, "projected_velocity_km_s", projected_velocity)
        object.__setattr__(self, "limb_darkening", epsilon)
        object.__setattr__(self, "velocity_step_km_s", velocity_step)
        object.__setattr__(self, "name", name)

    @property
    def vsini_km_s(self) -> float:
        """Projected rotational velocity in km/s."""

        return self.projected_velocity_km_s

    def prepare_on_grid(
        self,
        native_grid: SpectralGrid,
        target_grid: SpectralGrid | None = None,
    ) -> "PreparedRotationalBroadeningResponse":
        """Prepare rotation to an explicit or constant-velocity target grid."""

        if native_grid.size < 2:
            raise RobertValidationError("native rotational grid must contain at least two points")
        target = (
            _trim_rotational_target_grid(
                native_grid,
                self.projected_velocity_km_s,
                self.velocity_step_km_s,
            )
            if target_grid is None
            else target_grid
        )
        operator = _prepare_rotational_operator(
            native_grid,
            target,
            self.projected_velocity_km_s,
            self.limb_darkening,
            name=self.name,
        )
        return PreparedRotationalBroadeningResponse(
            source_grid=native_grid,
            target_grid=target,
            operator=operator,
            projected_velocity_km_s=self.projected_velocity_km_s,
            limb_darkening=self.limb_darkening,
            name=self.name,
        )

    def prepare(
        self,
        observation_or_grid: Observation | SpectralGrid,
        native_grid: SpectralGrid | None = None,
    ) -> "PreparedRotationalBroadeningResponse":
        """Prepare rotation for an observation or a native grid."""

        if isinstance(observation_or_grid, Observation):
            if native_grid is None:
                raise RobertValidationError("native_grid is required for an observation response")
            prepared = self.prepare_on_grid(native_grid, observation_or_grid.spectral_grid)
            object.__setattr__(prepared, "observation", observation_or_grid)
            return prepared
        if not isinstance(observation_or_grid, SpectralGrid) or native_grid is not None:
            raise RobertValidationError("prepare expects a SpectralGrid or an Observation plus native_grid")
        return self.prepare_on_grid(observation_or_grid)


@dataclass(frozen=True)
class PreparedRotationalBroadeningResponse:
    """Prepared Gray rotational broadening operator."""

    source_grid: SpectralGrid
    target_grid: SpectralGrid
    operator: PreparedSpectralOperator
    projected_velocity_km_s: float
    limb_darkening: float
    name: str = "rotational-broadening"
    observation: Observation | None = None

    def __post_init__(self) -> None:
        if not _grid_matches(self.source_grid, self.operator.source_grid):
            raise RobertValidationError("rotational operator source grid does not match response")
        if not _grid_matches(self.target_grid, self.operator.target_grid):
            raise RobertValidationError("rotational operator target grid does not match response")
        velocity = _nonnegative_float(self.projected_velocity_km_s, "projected_velocity_km_s")
        epsilon = _finite_float(self.limb_darkening, "limb_darkening")
        if not 0.0 <= epsilon <= 1.0:
            raise RobertValidationError("limb_darkening must be between 0 and 1")
        if not self.name:
            raise RobertValidationError("name must not be empty")
        object.__setattr__(self, "projected_velocity_km_s", velocity)
        object.__setattr__(self, "limb_darkening", epsilon)
        if self.observation is not None:
            if self.observation.wavelength_unit != self.target_grid.unit:
                raise RobertValidationError("rotational observation and target units must match")

    @property
    def native_grid(self) -> SpectralGrid:
        """Alias for the source grid."""

        return self.source_grid

    @property
    def vsini_km_s(self) -> float:
        """Projected rotational velocity in km/s."""

        return self.projected_velocity_km_s

    def observe(self, spectrum: Spectrum) -> Spectrum:
        """Apply the prepared rotational kernel."""

        if self.observation is not None:
            if spectrum.unit != self.observation.flux_unit:
                raise RobertValidationError("spectrum and observation value units must match")
            if spectrum.observable != self.observation.observable:
                raise RobertValidationError("spectrum and observation observables must match")
        return self.operator.observe(spectrum)

    apply = observe


def _prepare_pixel_operator(
    observation: Observation,
    source_grid: SpectralGrid,
    *,
    name: str,
) -> PreparedSpectralOperator:
    """Prepare a flux-conserving response for explicit observation bins."""

    if observation.wavelength_bin_edges is None:
        raise RobertValidationError(
            "pixel integration requires observation wavelength bin edges"
        )
    target_grid = observation.spectral_grid
    if source_grid.unit != target_grid.unit:
        raise RobertValidationError("source and observation spectral units must match")
    _require_positive_grid(source_grid, "source")
    _require_positive_grid(target_grid, "target")
    source = _ascending_values(source_grid)
    edges = np.asarray(target_grid.bin_edges, dtype=float)
    if edges[0] > edges[-1]:
        edges = edges[::-1]
    _check_coordinate_coverage(
        source,
        edges,
        message="observation bins extend outside native spectrum",
    )
    indices, weights, row_starts = _build_bin_integration_operator(source, edges)
    return PreparedSpectralOperator(
        source_grid=source_grid,
        target_grid=target_grid,
        indices=indices,
        weights=weights,
        row_starts=row_starts,
        name=name,
        metadata={
            "integration": "piecewise_linear_flux_conserving_bins",
        },
    )


@dataclass(frozen=True)
class PixelIntegrationResponse:
    """Flux-conserving integration into explicit observation pixel bins."""

    name: str = "pixel-bin-integration"

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("name must not be empty")

    def prepare_on_grid(
        self,
        observation: Observation,
        native_grid: SpectralGrid,
    ) -> "PreparedPixelIntegrationResponse":
        """Prepare fixed piecewise-linear bin integrals."""

        operator = _prepare_pixel_operator(
            observation,
            native_grid,
            name=self.name,
        )
        return PreparedPixelIntegrationResponse(
            observation=observation,
            source_grid=native_grid,
            target_grid=observation.spectral_grid,
            operator=operator,
            name=self.name,
        )

    def prepare(
        self,
        observation: Observation,
        native_grid: SpectralGrid | None = None,
        *,
        source_grid: SpectralGrid | None = None,
    ) -> "PreparedPixelIntegrationResponse":
        """Prepare integration from a native grid to observation bins."""

        if native_grid is None:
            native_grid = source_grid
        elif source_grid is not None and source_grid is not native_grid:
            raise RobertValidationError("native_grid and source_grid must not both be provided")
        if native_grid is None:
            raise RobertValidationError("native_grid is required for pixel integration")
        return self.prepare_on_grid(observation, native_grid)


@dataclass(frozen=True)
class PreparedPixelIntegrationResponse:
    """Prepared flux-conserving pixel/bin integration operator."""

    observation: Observation
    source_grid: SpectralGrid
    target_grid: SpectralGrid
    operator: PreparedSpectralOperator
    name: str = "pixel-bin-integration"

    def __post_init__(self) -> None:
        if not _grid_matches(self.source_grid, self.operator.source_grid):
            raise RobertValidationError("pixel operator source grid does not match response")
        if not _grid_matches(self.target_grid, self.operator.target_grid):
            raise RobertValidationError("pixel operator target grid does not match response")
        if self.target_grid.size != self.observation.n_points:
            raise RobertValidationError("pixel target grid must match observation points")
        if not self.name:
            raise RobertValidationError("name must not be empty")

    @property
    def native_grid(self) -> SpectralGrid:
        """Alias for the source grid."""

        return self.source_grid

    def observe(self, spectrum: Spectrum) -> Spectrum:
        """Apply the prepared pixel/bin integration."""

        if spectrum.unit != self.observation.flux_unit:
            raise RobertValidationError("spectrum and observation value units must match")
        if spectrum.observable != self.observation.observable:
            raise RobertValidationError("spectrum and observation observables must match")
        return self.operator.observe(spectrum)

    apply = observe


def _prepare_linear_grid_operator(
    source_grid: SpectralGrid,
    target_grid: SpectralGrid,
    *,
    name: str,
) -> PreparedSpectralOperator:
    """Prepare a strict, non-extrapolating linear grid mapping."""

    if source_grid.unit != target_grid.unit:
        raise RobertValidationError("source and target spectral units must match")
    _require_positive_grid(source_grid, "source")
    _require_positive_grid(target_grid, "target")
    source = _ascending_values(source_grid)
    target = _ascending_values(target_grid)
    _check_coordinate_coverage(
        source,
        target,
        message="target grid extends outside native spectrum",
    )
    if target_grid.bin_edges is not None:
        edges = np.asarray(target_grid.bin_edges, dtype=float)
        if edges[0] > edges[-1]:
            edges = edges[::-1]
        _check_coordinate_coverage(
            source,
            edges,
            message="target bin edges extend outside native spectrum",
        )
    indices, weights, row_starts = _build_linear_interpolation_operator(source, target)
    return PreparedSpectralOperator(
        source_grid=source_grid,
        target_grid=target_grid,
        indices=indices,
        weights=weights,
        row_starts=row_starts,
        name=name,
        metadata={"mapping": "piecewise_linear_no_extrapolation"},
    )


def _stage_rank(stage: object) -> int:
    """Return the required physical ordering rank for one response stage."""

    if isinstance(stage, RelativisticDopplerResponse):
        return 0
    if isinstance(stage, RotationalBroadeningResponse):
        return 1
    if isinstance(stage, GaussianHighResolutionResponse):
        return 2
    if isinstance(stage, PixelIntegrationResponse):
        return 3
    raise RobertValidationError(
        "response chain stages must be Doppler, rotational, Gaussian LSF, or pixel integration responses"
    )


@dataclass(frozen=True, init=False)
class HighResolutionResponseChain:
    """Ordered prepared response chain for high-resolution observations.

    The physical order is Doppler shift, rotational broadening, Gaussian LSF,
    then pixel/bin integration.  The final mapping is added automatically:
    explicit observation bins use flux-conserving integration, and observations
    without bins use strict piecewise-linear interpolation.  Continuum removal,
    detrending, and telluric corrections are intentionally not stages here.
    """

    stages: tuple[object, ...]
    name: str = "high-resolution-response-chain"

    def __init__(
        self,
        stages: Sequence[object] | None = None,
        *,
        responses: Sequence[object] | None = None,
        name: str = "high-resolution-response-chain",
    ) -> None:
        if stages is not None and responses is not None:
            raise RobertValidationError("provide only one of stages and responses")
        selected = stages if stages is not None else responses
        if selected is None:
            raise RobertValidationError("response chain requires at least one stage")
        normalised = tuple(selected)
        if not normalised:
            raise RobertValidationError("response chain requires at least one stage")
        ranks = [_stage_rank(stage) for stage in normalised]
        if ranks != sorted(ranks):
            raise RobertValidationError(
                "response chain stages must be ordered Doppler, rotational, Gaussian LSF, pixel integration"
            )
        if len(set(ranks)) != len(ranks):
            raise RobertValidationError(
                "response chain can contain at most one stage of each physical type"
            )
        if not name:
            raise RobertValidationError("name must not be empty")
        object.__setattr__(self, "stages", normalised)
        object.__setattr__(self, "name", name)

    @property
    def responses(self) -> tuple[object, ...]:
        """Alias for configured response stages."""

        return self.stages

    def prepare(
        self,
        observation: Observation,
        native_grid: SpectralGrid,
    ) -> "PreparedHighResolutionResponseChain":
        """Prepare all response operators once for one observation."""

        if native_grid.unit != observation.wavelength_unit:
            raise RobertValidationError("native and observation spectral units must match")
        _require_positive_grid(native_grid, "native")
        observation_grid = observation.spectral_grid
        working_grid = native_grid
        prepared: list[object] = []

        for stage in self.stages:
            if isinstance(stage, RelativisticDopplerResponse):
                prepared_stage = stage.prepare_on_grid(working_grid)
            elif isinstance(stage, RotationalBroadeningResponse):
                prepared_stage = stage.prepare_on_grid(working_grid)
            elif isinstance(stage, GaussianHighResolutionResponse):
                prepared_stage = stage.prepare_on_grid(working_grid)
            elif isinstance(stage, PixelIntegrationResponse):
                prepared_stage = stage.prepare_on_grid(observation, working_grid)
            else:  # pragma: no cover - validated by __init__
                raise RobertValidationError("unsupported high-resolution response stage")
            prepared.append(prepared_stage)
            working_grid = prepared_stage.target_grid

        if not _grid_matches(working_grid, observation_grid):
            if observation_grid.bin_edges is not None:
                final_stage = PixelIntegrationResponse().prepare_on_grid(
                    observation,
                    working_grid,
                )
            else:
                linear_indices, linear_weights, linear_row_starts = (
                    _build_linear_interpolation_operator(
                        _ascending_values(working_grid),
                        _ascending_values(observation_grid),
                    )
                )
                final_stage = PreparedSpectralOperator(
                    source_grid=working_grid,
                    target_grid=observation_grid,
                    indices=linear_indices,
                    weights=linear_weights,
                    row_starts=linear_row_starts,
                    name="linear-observation-grid",
                    metadata={"mapping": "piecewise_linear_no_extrapolation"},
                )
            prepared.append(final_stage)

        return PreparedHighResolutionResponseChain(
            observation=observation,
            native_grid=native_grid,
            operators=tuple(prepared),
            name=self.name,
        )


@dataclass(frozen=True)
class PreparedHighResolutionResponseChain:
    """Prepared ordered response operators for repeated model evaluations."""

    observation: Observation
    native_grid: SpectralGrid
    operators: tuple[object, ...]
    name: str = "high-resolution-response-chain"

    def __post_init__(self) -> None:
        operators = tuple(self.operators)
        if not operators:
            raise RobertValidationError("prepared response chain requires operators")
        if not self.name:
            raise RobertValidationError("name must not be empty")
        first_source = getattr(operators[0], "source_grid", None)
        last_target = getattr(operators[-1], "target_grid", None)
        if not isinstance(first_source, SpectralGrid) or not _grid_matches(
            first_source,
            self.native_grid,
        ):
            raise RobertValidationError("prepared response chain source grid does not match native_grid")
        if not isinstance(last_target, SpectralGrid) or not _grid_matches(
            last_target,
            self.observation.spectral_grid,
        ):
            raise RobertValidationError("prepared response chain does not terminate on observation grid")
        for left, right in zip(operators[:-1], operators[1:], strict=True):
            left_target = getattr(left, "target_grid", None)
            right_source = getattr(right, "source_grid", None)
            if not isinstance(left_target, SpectralGrid) or not isinstance(right_source, SpectralGrid):
                raise RobertValidationError("prepared response chain operators must expose source and target grids")
            if not _grid_matches(left_target, right_source):
                raise RobertValidationError("prepared response chain has a grid discontinuity")
        object.__setattr__(self, "operators", operators)

    @property
    def stages(self) -> tuple[object, ...]:
        """Alias for prepared operators."""

        return self.operators

    @property
    def prepared_operators(self) -> tuple[object, ...]:
        """Return the immutable prepared operator sequence."""

        return self.operators

    def observe(self, spectrum: Spectrum) -> Spectrum:
        """Apply every prepared stage in the configured order."""

        if not _grid_matches(spectrum.spectral_grid, self.native_grid):
            raise RobertValidationError(
                "spectrum grid does not match the prepared response chain native grid"
            )
        if spectrum.unit != self.observation.flux_unit:
            raise RobertValidationError("spectrum and observation value units must match")
        if spectrum.observable != self.observation.observable:
            raise RobertValidationError("spectrum and observation observables must match")
        current = spectrum
        for operator in self.operators:
            current = operator.observe(current)
        if not _grid_matches(current.spectral_grid, self.observation.spectral_grid):
            raise RobertValidationError("prepared response chain output grid does not match observation")
        metadata = dict(current.metadata)
        metadata["response_chain"] = self.name
        metadata["response_chain_stages"] = ",".join(
            str(getattr(operator, "name", type(operator).__name__))
            for operator in self.operators
        )
        return Spectrum(
            spectral_grid=current.spectral_grid,
            values=current.values,
            unit=current.unit,
            observable=current.observable,
            metadata=metadata,
        )

    apply = observe


# These aliases make the response discoverable under common instrument-model
# terminology while keeping one implementation and one prepared type.
GaussianLineSpreadResponse = GaussianHighResolutionResponse
PreparedGaussianLineSpreadResponse = PreparedGaussianHighResolutionResponse
GaussianLSFResponse = GaussianHighResolutionResponse
PreparedGaussianLSFResponse = PreparedGaussianHighResolutionResponse
RelativisticDopplerShiftResponse = RelativisticDopplerResponse
PreparedRelativisticDopplerShiftResponse = PreparedRelativisticDopplerResponse
DopplerShiftResponse = RelativisticDopplerResponse
PreparedDopplerShiftResponse = PreparedRelativisticDopplerResponse
RotationalResponse = RotationalBroadeningResponse
PreparedRotationalResponse = PreparedRotationalBroadeningResponse
FluxConservingPixelResponse = PixelIntegrationResponse
PreparedFluxConservingPixelResponse = PreparedPixelIntegrationResponse
ResponseChain = HighResolutionResponseChain
PreparedResponseChain = PreparedHighResolutionResponseChain


__all__ = [
    "DopplerShiftResponse",
    "FluxConservingPixelResponse",
    "GaussianHighResolutionResponse",
    "PreparedGaussianHighResolutionResponse",
    "GaussianLineSpreadResponse",
    "PreparedGaussianLineSpreadResponse",
    "GaussianLSFResponse",
    "PreparedGaussianLSFResponse",
    "HighResolutionResponseChain",
    "PreparedHighResolutionResponseChain",
    "PreparedSpectralOperator",
    "PixelIntegrationResponse",
    "PreparedPixelIntegrationResponse",
    "PreparedDopplerShiftResponse",
    "PreparedFluxConservingPixelResponse",
    "PreparedRelativisticDopplerResponse",
    "PreparedRelativisticDopplerShiftResponse",
    "PreparedResponseChain",
    "PreparedRotationalBroadeningResponse",
    "PreparedRotationalResponse",
    "RelativisticDopplerResponse",
    "RelativisticDopplerShiftResponse",
    "ResponseChain",
    "RotationalBroadeningResponse",
    "RotationalResponse",
]
