"""Correlated-noise likelihood and covariance interfaces.

The classes here evaluate the likelihood of one already prepared spectrum.
Instrument response and data filtering belong to
``robert_exoplanets.instruments.high_resolution_observation``.  A caller can
therefore prepare a real order once, apply the same linear filter to data and
model, and pass the resulting covariance to this module.

This module evaluates one observation at a time.  Summing separate
per-order likelihoods assumes that the orders have independent noise; it does
not represent covariance between orders.  Cross-order covariance needs one
joint covariance operator and one joint observation vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError, Spectrum
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.instruments import Observation


class CovarianceOperator(Protocol):
    """Interface for a covariance used by a Gaussian likelihood.

    Implementations can be positive definite or positive semidefinite.  A
    semidefinite implementation must expose ``effective_rank`` and
    ``supports`` so that a likelihood can use the density on the supported
    subspace and reject residuals outside that subspace.
    """

    size: int

    def solve(self, vector: ArrayLike) -> NDArray[np.float64]:
        """Solve ``C x = vector`` without exposing an inverse."""

    def log_determinant(self) -> float:
        """Return the supported-space log determinant."""

    def diagonal(self) -> NDArray[np.float64]:
        """Return the covariance diagonal."""

    def subset(self, mask: ArrayLike) -> "CovarianceOperator":
        """Return the covariance selected by a boolean mask."""

    effective_rank: int

    def supports(self, vector: ArrayLike) -> bool:
        """Return whether a vector lies in the covariance support."""

    def pseudo_log_determinant(self) -> float:
        """Return the log determinant on the supported subspace."""


def _readonly_vector(values: ArrayLike, name: str, size: int) -> NDArray[np.float64]:
    """Validate one finite vector."""

    vector = np.array(values, dtype=float, copy=True)
    if vector.ndim != 1 or vector.size != size:
        raise RobertValidationError(f"{name} must be one-dimensional with length {size}")
    if not np.all(np.isfinite(vector)):
        raise RobertValidationError(f"{name} must contain only finite values")
    vector.setflags(write=False)
    return vector


def _readonly_mask(values: ArrayLike, name: str, size: int) -> NDArray[np.bool_]:
    """Validate one boolean mask."""

    mask = np.array(values, dtype=bool, copy=True)
    if mask.ndim != 1 or mask.size != size:
        raise RobertValidationError(f"{name} must be one-dimensional with length {size}")
    if not np.any(mask):
        raise RobertValidationError("likelihood mask excludes all observation points")
    mask.setflags(write=False)
    return mask


def _nonnegative_finite_float(value: float, name: str) -> float:
    """Validate one finite non-negative floating-point tolerance."""

    if isinstance(value, (bool, np.bool_)):
        raise RobertValidationError(f"{name} must be finite and non-negative")
    try:
        normalised = float(value)
    except (TypeError, ValueError) as error:
        raise RobertValidationError(f"{name} must be finite and non-negative") from error
    if not np.isfinite(normalised) or normalised < 0.0:
        raise RobertValidationError(f"{name} must be finite and non-negative")
    return normalised


def _covariance_effective_rank(covariance: CovarianceOperator) -> int:
    """Return and validate the supported covariance rank."""

    value = getattr(covariance, "effective_rank", covariance.size)
    if callable(value):
        value = value()
    try:
        rank = int(value)
    except (TypeError, ValueError) as error:
        raise RobertValidationError(
            "covariance effective_rank must be a non-negative integer"
        ) from error
    if rank != value or rank < 0 or rank > covariance.size:
        raise RobertValidationError(
            "covariance effective_rank must be between zero and covariance size"
        )
    return rank


def _covariance_pseudo_log_determinant(covariance: CovarianceOperator) -> float:
    """Return a finite supported-space log determinant."""

    method = getattr(covariance, "pseudo_log_determinant", None)
    value = covariance.log_determinant() if not callable(method) else method()
    try:
        normalised = float(value)
    except (TypeError, ValueError) as error:
        raise RobertValidationError(
            "covariance pseudo-log-determinant must be finite"
        ) from error
    if not np.isfinite(normalised):
        raise RobertValidationError(
            "covariance pseudo-log-determinant must be finite"
        )
    return normalised


def _covariance_supports(
    covariance: CovarianceOperator,
    vector: ArrayLike,
) -> bool:
    """Check a residual against an optional semidefinite support."""

    method = getattr(covariance, "supports", None)
    if not callable(method):
        # Older custom positive-definite operators have full support by
        # contract.  They remain valid without the optional PSD method.
        return True
    return bool(method(vector))


def _require_positive_effective_rank(covariance: CovarianceOperator) -> int:
    """Require at least one stochastic residual dimension."""

    rank = _covariance_effective_rank(covariance)
    if rank < 1:
        raise RobertValidationError(
            "covariance effective_rank must be positive for a likelihood"
        )
    return rank


@dataclass(frozen=True)
class DenseCovariance:
    """Validated dense positive-definite covariance operator.

    The Cholesky factor is kept once.  Repeated likelihood calls use triangular
    solves and never form a covariance inverse.  This class is intended for
    small or moderate order windows.  Larger data sets can provide the same
    :class:`CovarianceOperator` interface with a structured or sparse backend.
    """

    matrix: ArrayLike
    name: str = "dense-covariance"
    _cholesky: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        matrix = np.array(self.matrix, dtype=float, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
            raise RobertValidationError("covariance matrix must be non-empty and square")
        if not np.all(np.isfinite(matrix)):
            raise RobertValidationError("covariance matrix must contain only finite values")
        if not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-14):
            raise RobertValidationError("covariance matrix must be symmetric")
        try:
            cholesky = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise RobertValidationError(
                "covariance matrix must be positive definite"
            ) from error
        if not str(self.name).strip():
            raise RobertValidationError("covariance name must not be empty")
        matrix.setflags(write=False)
        cholesky.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "_cholesky", cholesky)
        object.__setattr__(self, "name", str(self.name).strip())

    @property
    def size(self) -> int:
        """Number of correlated data points."""

        return int(self.matrix.shape[0])

    @property
    def effective_rank(self) -> int:
        """Number of supported dimensions for this full-rank covariance."""

        return self.size

    @property
    def cholesky(self) -> NDArray[np.float64]:
        """Return the read-only lower Cholesky factor."""

        return self._cholesky

    def solve(self, vector: ArrayLike) -> NDArray[np.float64]:
        """Solve the covariance system with two triangular solves."""

        values = _readonly_vector(vector, "covariance solve vector", self.size)
        forward = np.linalg.solve(self._cholesky, values)
        result = np.linalg.solve(self._cholesky.T, forward)
        result = np.asarray(result, dtype=float)
        result.setflags(write=False)
        return result

    def whiten(self, vector: ArrayLike) -> NDArray[np.float64]:
        """Return the Cholesky-whitened vector ``L^-1 vector``."""

        values = _readonly_vector(vector, "covariance whiten vector", self.size)
        result = np.asarray(np.linalg.solve(self._cholesky, values), dtype=float)
        result.setflags(write=False)
        return result

    def log_determinant(self) -> float:
        """Return the stable Cholesky log determinant."""

        return float(2.0 * np.sum(np.log(np.diag(self._cholesky))))

    def pseudo_log_determinant(self) -> float:
        """Return the supported-space log determinant.

        A dense positive-definite covariance has no null space, so its
        pseudo-log-determinant is its ordinary log determinant.
        """

        return self.log_determinant()

    pseudo_logdet = pseudo_log_determinant

    def supports(self, vector: ArrayLike) -> bool:
        """Return ``True`` for every finite vector of the correct size."""

        _readonly_vector(vector, "covariance support vector", self.size)
        return True

    def diagonal(self) -> NDArray[np.float64]:
        """Return a read-only copy of the covariance diagonal."""

        result = np.array(np.diag(self.matrix), dtype=float, copy=True)
        result.setflags(write=False)
        return result

    def subset(self, mask: ArrayLike) -> "DenseCovariance":
        """Return a positive-definite covariance for selected data points."""

        selected = _readonly_mask(mask, "covariance subset mask", self.size)
        return DenseCovariance(
            self.matrix[np.ix_(selected, selected)],
            name=f"{self.name}-masked",
        )


@dataclass(frozen=True)
class PositiveSemidefiniteCovariance:
    """Validated covariance on a possibly lower-dimensional support.

    This operator is intended for fixed PCA/SysRem projections and other
    linear filters that produce ``F C F.T`` with dependent output rows.  The
    eigendecomposition defines a numerical support: eigenvalues greater than
    ``max(rank_atol, rank_rtol * largest_eigenvalue)`` are retained.  Small
    negative eigenvalues within this threshold are treated as round-off; a
    larger negative eigenvalue is rejected.

    ``solve`` is a Moore-Penrose solve on the retained support.  A residual
    with a component outside that support is not a valid Gaussian residual and
    raises :class:`RobertValidationError`.  ``log_determinant`` and
    ``pseudo_log_determinant`` return the log pseudo-determinant, and
    ``effective_rank`` gives the supported dimension.  ``support_rtol`` and
    ``support_atol`` control the residual support test in the units of the
    supplied data.
    """

    matrix: ArrayLike
    name: str = "positive-semidefinite-covariance"
    rank_rtol: float = 1.0e-12
    rank_atol: float = 0.0
    support_rtol: float = 1.0e-10
    support_atol: float = 0.0
    _eigenvalues: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _eigenvectors: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _supported_eigenvalues: NDArray[np.float64] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _supported_eigenvectors: NDArray[np.float64] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _null_eigenvectors: NDArray[np.float64] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        matrix = np.array(self.matrix, dtype=float, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
            raise RobertValidationError(
                "covariance matrix must be non-empty and square"
            )
        if not np.all(np.isfinite(matrix)):
            raise RobertValidationError("covariance matrix must contain only finite values")
        if not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-14):
            raise RobertValidationError("covariance matrix must be symmetric")
        if not str(self.name).strip():
            raise RobertValidationError("covariance name must not be empty")

        rank_rtol = _nonnegative_finite_float(self.rank_rtol, "rank_rtol")
        rank_atol = _nonnegative_finite_float(self.rank_atol, "rank_atol")
        support_rtol = _nonnegative_finite_float(self.support_rtol, "support_rtol")
        support_atol = _nonnegative_finite_float(self.support_atol, "support_atol")
        if rank_rtol == 0.0 and rank_atol == 0.0:
            raise RobertValidationError(
                "rank_rtol and rank_atol cannot both be zero"
            )

        # Symmetrising after the explicit symmetry check removes only the
        # small representational asymmetry that can otherwise perturb eigh.
        symmetric = 0.5 * (matrix + matrix.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        except np.linalg.LinAlgError as error:
            raise RobertValidationError(
                "covariance eigendecomposition failed"
            ) from error
        scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
        threshold = max(rank_atol, rank_rtol * scale)
        if np.any(eigenvalues < -threshold):
            raise RobertValidationError(
                "covariance matrix must be positive semidefinite"
            )

        supported = eigenvalues > threshold
        clipped = np.maximum(eigenvalues, 0.0)
        supported_eigenvalues = np.asarray(clipped[supported], dtype=float)
        supported_eigenvectors = np.asarray(eigenvectors[:, supported], dtype=float)
        null_eigenvectors = np.asarray(eigenvectors[:, ~supported], dtype=float)
        for array in (
            symmetric,
            eigenvalues,
            eigenvectors,
            supported_eigenvalues,
            supported_eigenvectors,
            null_eigenvectors,
        ):
            array.setflags(write=False)

        object.__setattr__(self, "matrix", symmetric)
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "rank_rtol", rank_rtol)
        object.__setattr__(self, "rank_atol", rank_atol)
        object.__setattr__(self, "support_rtol", support_rtol)
        object.__setattr__(self, "support_atol", support_atol)
        object.__setattr__(self, "_eigenvalues", eigenvalues)
        object.__setattr__(self, "_eigenvectors", eigenvectors)
        object.__setattr__(self, "_supported_eigenvalues", supported_eigenvalues)
        object.__setattr__(self, "_supported_eigenvectors", supported_eigenvectors)
        object.__setattr__(self, "_null_eigenvectors", null_eigenvectors)

    @property
    def size(self) -> int:
        """Number of output coordinates."""

        return int(self.matrix.shape[0])

    @property
    def effective_rank(self) -> int:
        """Number of eigen-directions retained by the tolerance."""

        return int(self._supported_eigenvalues.size)

    @property
    def rank(self) -> int:
        """Alias for :attr:`effective_rank`."""

        return self.effective_rank

    @property
    def eigenvalues(self) -> NDArray[np.float64]:
        """Return all eigenvalues, including numerical null directions."""

        return self._eigenvalues

    @property
    def eigenvectors(self) -> NDArray[np.float64]:
        """Return the orthonormal eigenvector matrix."""

        return self._eigenvectors

    def _validated_support_vector(self, vector: ArrayLike) -> NDArray[np.float64]:
        return _readonly_vector(vector, "covariance support vector", self.size)

    def support_residual(self, vector: ArrayLike) -> float:
        """Return the norm of the component outside the retained support."""

        values = self._validated_support_vector(vector)
        if self._null_eigenvectors.shape[1] == 0:
            return 0.0
        residual = float(np.linalg.norm(self._null_eigenvectors.T @ values))
        return residual if np.isfinite(residual) else float("inf")

    def supports(self, vector: ArrayLike) -> bool:
        """Return whether a vector lies in the numerical covariance support."""

        values = self._validated_support_vector(vector)
        vector_norm = float(np.linalg.norm(values))
        if not np.isfinite(vector_norm):
            return False
        residual = self.support_residual(values)
        tolerance = max(
            self.support_atol,
            self.support_rtol * vector_norm,
        )
        return bool(residual <= tolerance)

    # The longer spelling is useful when the support test is read in code.
    in_support = supports

    def solve(self, vector: ArrayLike) -> NDArray[np.float64]:
        """Return the Moore-Penrose solve on the supported subspace."""

        values = self._validated_support_vector(vector)
        if not self.supports(values):
            raise RobertValidationError(
                "covariance solve vector lies outside the positive-semidefinite support"
            )
        if self.effective_rank == 0:
            result = np.zeros(self.size, dtype=float)
        else:
            coefficients = self._supported_eigenvectors.T @ values
            result = self._supported_eigenvectors @ (
                coefficients / self._supported_eigenvalues
            )
        result = np.asarray(result, dtype=float)
        if not np.all(np.isfinite(result)):
            raise RobertValidationError("covariance pseudo-solve produced non-finite values")
        result.setflags(write=False)
        return result

    def whiten(self, vector: ArrayLike) -> NDArray[np.float64]:
        """Return supported-space whitened coordinates."""

        values = self._validated_support_vector(vector)
        if not self.supports(values):
            raise RobertValidationError(
                "covariance whiten vector lies outside the positive-semidefinite support"
            )
        if self.effective_rank == 0:
            result = np.empty(0, dtype=float)
        else:
            result = (
                self._supported_eigenvectors.T @ values
            ) / np.sqrt(self._supported_eigenvalues)
        result = np.asarray(result, dtype=float)
        if not np.all(np.isfinite(result)):
            raise RobertValidationError("covariance whitening produced non-finite values")
        result.setflags(write=False)
        return result

    def pseudo_log_determinant(self) -> float:
        """Return the log pseudo-determinant on the supported subspace."""

        if self.effective_rank == 0:
            return 0.0
        result = float(np.sum(np.log(self._supported_eigenvalues)))
        if not np.isfinite(result):
            raise RobertValidationError(
                "covariance pseudo-log-determinant must be finite"
            )
        return result

    pseudo_logdet = pseudo_log_determinant

    def log_determinant(self) -> float:
        """Return the supported-space log determinant for protocol use."""

        return self.pseudo_log_determinant()

    def diagonal(self) -> NDArray[np.float64]:
        """Return a read-only copy of the covariance diagonal."""

        result = np.array(np.diag(self.matrix), dtype=float, copy=True)
        result.setflags(write=False)
        return result

    def subset(self, mask: ArrayLike) -> "PositiveSemidefiniteCovariance":
        """Return the selected principal covariance, retaining PSD support."""

        selected = _readonly_mask(mask, "covariance subset mask", self.size)
        return PositiveSemidefiniteCovariance(
            self.matrix[np.ix_(selected, selected)],
            name=f"{self.name}-masked",
            rank_rtol=self.rank_rtol,
            rank_atol=self.rank_atol,
            support_rtol=self.support_rtol,
            support_atol=self.support_atol,
        )


def _prediction_spectrum(prediction: Spectrum | Any) -> Spectrum:
    """Extract a spectrum from a direct spectrum or forward prediction."""

    if isinstance(prediction, Spectrum):
        return prediction
    observed_spectrum = getattr(prediction, "observed_spectrum", None)
    if isinstance(observed_spectrum, Spectrum):
        return observed_spectrum
    raise RobertValidationError(
        "prediction must be a Spectrum or expose observed_spectrum"
    )


@dataclass(frozen=True)
class CorrelatedGaussianLikelihood:
    """Gaussian likelihood with a full covariance operator.

    The covariance is fixed during a likelihood evaluation.  Observation masks
    select the corresponding covariance rows and columns.  The optional
    ``offset_parameter`` is an additive model nuisance term.  No stellar or
    telluric fitting is performed here.

    A :class:`PositiveSemidefiniteCovariance` uses a density on its supported
    subspace.  Residuals outside that subspace receive ``invalid_model_loglike``
    and the normalization uses the covariance effective rank.  Separate
    instances for separate spectral orders do not model cross-order
    covariance.

    ``effective_inputs`` returns the diagonal standard deviation for
    compatibility with the existing independent-likelihood interface.  It is
    only a diagnostic convenience; use :meth:`effective_covariance` for the
    full correlated noise model.
    """

    covariance: CovarianceOperator | ArrayLike
    name: str = "correlated-gaussian"
    include_normalization: bool = False
    offset_parameter: str | None = None
    invalid_model_loglike: float = float("-inf")
    coordinate_rtol: float = 1.0e-12
    coordinate_atol: float = 0.0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise RobertValidationError("likelihood name must not be empty")
        rtol = float(self.coordinate_rtol)
        atol = float(self.coordinate_atol)
        if not np.isfinite(rtol) or rtol < 0.0:
            raise RobertValidationError("coordinate_rtol must be finite and non-negative")
        if not np.isfinite(atol) or atol < 0.0:
            raise RobertValidationError("coordinate_atol must be finite and non-negative")
        invalid = float(self.invalid_model_loglike)
        if np.isnan(invalid) or invalid == np.inf:
            raise RobertValidationError(
                "invalid_model_loglike must be finite or negative infinity"
            )
        covariance = self.covariance
        if not all(
            callable(getattr(covariance, method, None))
            for method in ("solve", "log_determinant", "diagonal", "subset")
        ):
            covariance = DenseCovariance(covariance)  # type: ignore[arg-type]
        if int(getattr(covariance, "size", 0)) < 1:
            raise RobertValidationError("covariance operator size must be positive")
        _require_positive_effective_rank(covariance)
        if self.offset_parameter is not None and not str(self.offset_parameter).strip():
            raise RobertValidationError("offset_parameter must not be empty")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "name", str(self.name).strip())
        if self.offset_parameter is not None:
            object.__setattr__(
                self,
                "offset_parameter",
                str(self.offset_parameter).strip(),
            )
        object.__setattr__(self, "coordinate_rtol", rtol)
        object.__setattr__(self, "coordinate_atol", atol)
        object.__setattr__(self, "invalid_model_loglike", invalid)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    def _validated_inputs(
        self,
        prediction: Spectrum | Any,
        observation: Observation,
        parameters: Mapping[str, float] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Validate prediction/data compatibility and apply the mask."""

        spectrum = _prediction_spectrum(prediction)
        if spectrum.values.shape != observation.flux.shape:
            raise RobertValidationError("prediction and observation shapes must match")
        if spectrum.spectral_grid.unit != observation.wavelength_unit:
            raise RobertValidationError("prediction and observation wavelength units must match")
        if not np.allclose(
            spectrum.spectral_grid.values,
            observation.wavelength,
            rtol=self.coordinate_rtol,
            atol=self.coordinate_atol,
        ):
            raise RobertValidationError(
                "prediction must be evaluated on the observation wavelength grid"
            )
        if spectrum.unit != observation.flux_unit:
            raise RobertValidationError("prediction and observation units must match")
        if spectrum.observable != observation.observable:
            raise RobertValidationError("prediction and observation observables must match")
        if self.covariance.size != observation.n_points:
            raise RobertValidationError(
                "covariance size must match the unmasked observation size"
            )
        model = np.array(spectrum.values, dtype=float, copy=True)
        parameter_values = {} if parameters is None else parameters
        if self.offset_parameter is not None and self.offset_parameter in parameter_values:
            offset = float(parameter_values[self.offset_parameter])
            if not np.isfinite(offset):
                raise RobertValidationError("offset parameter must be finite")
            model += offset
        data = np.array(observation.flux, dtype=float, copy=True)
        valid = (
            np.ones(observation.n_points, dtype=bool)
            if observation.mask is None
            else np.array(observation.mask, dtype=bool, copy=True)
        )
        valid = _readonly_mask(valid, "likelihood mask", observation.n_points)
        model = np.array(model[valid], dtype=float, copy=True)
        data = np.array(data[valid], dtype=float, copy=True)
        if not np.all(np.isfinite(model)):
            model = np.asarray(model, dtype=float)
        for array in (model, data):
            array.setflags(write=False)
        return model, data, valid

    def effective_covariance(
        self,
        observation: Observation,
    ) -> CovarianceOperator:
        """Return the covariance after applying the observation mask."""

        if self.covariance.size != observation.n_points:
            raise RobertValidationError(
                "covariance size must match the unmasked observation size"
            )
        if observation.mask is None:
            covariance = self.covariance
        else:
            valid = _readonly_mask(observation.mask, "likelihood mask", observation.n_points)
            covariance = self.covariance.subset(valid)
        _require_positive_effective_rank(covariance)
        return covariance

    def effective_residual_rank(self, observation: Observation) -> int:
        """Return the supported residual dimension after the observation mask."""

        return _covariance_effective_rank(self.effective_covariance(observation))

    def chi_square(
        self,
        prediction: Spectrum | Any,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        """Return the exact covariance-weighted residual quadratic.

        ``inf`` is returned for a non-finite model or for a residual outside
        the support of a positive-semidefinite covariance.  The latter is a
        zero-probability residual under the corresponding degenerate Gaussian
        and is converted to ``invalid_model_loglike`` by :meth:`loglike`.
        """

        model, data, _ = self._validated_inputs(prediction, observation, parameters)
        if not np.all(np.isfinite(model)):
            return float("inf")
        covariance = self.effective_covariance(observation)
        residual = data - model
        if not _covariance_supports(covariance, residual):
            return float("inf")
        solved = covariance.solve(residual)
        if not np.all(np.isfinite(solved)):
            return float("inf")
        quadratic = float(residual @ solved)
        if not np.isfinite(quadratic):
            return float("inf")
        # A valid PSD solve is non-negative.  Clamp only round-off below zero;
        # a materially negative result signals an invalid custom operator.
        tolerance = 1.0e-12 * max(1.0, abs(quadratic))
        if quadratic < -tolerance:
            raise RobertValidationError(
                "covariance solve produced a negative chi-square"
            )
        return float(max(0.0, quadratic))

    def loglike(
        self,
        prediction: Spectrum | Any,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        """Return the correlated Gaussian log likelihood."""

        chi_square = self.chi_square(prediction, observation, parameters)
        if not np.isfinite(chi_square):
            return float(self.invalid_model_loglike)
        covariance = self.effective_covariance(observation)
        value = -0.5 * chi_square
        if self.include_normalization:
            rank = _covariance_effective_rank(covariance)
            value -= 0.5 * (
                _covariance_pseudo_log_determinant(covariance)
                + rank * np.log(2.0 * np.pi)
            )
        return float(value)

    def effective_inputs(
        self,
        prediction: Spectrum | Any,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return masked model, data, and diagonal standard deviations."""

        model, data, _ = self._validated_inputs(prediction, observation, parameters)
        covariance = self.effective_covariance(observation)
        uncertainty = np.sqrt(np.asarray(covariance.diagonal(), dtype=float))
        uncertainty.setflags(write=False)
        return model, data, uncertainty

    def pointwise_loglike(
        self,
        prediction: Spectrum | Any,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> NDArray[np.float64]:
        """Return one aggregate diagnostic term for correlated residuals.

        Correlated terms do not have a unique independent per-pixel split.  A
        single term preserves the exact scalar likelihood and avoids implying
        false pixel independence.
        """

        value = self.loglike(prediction, observation, parameters)
        output = np.asarray([value], dtype=float)
        output.setflags(write=False)
        return output


CorrelatedNoiseLikelihood = CorrelatedGaussianLikelihood
FullCovarianceLikelihood = CorrelatedGaussianLikelihood
DenseCovarianceOperator = DenseCovariance
PSDCovariance = PositiveSemidefiniteCovariance
LowRankCovariance = PositiveSemidefiniteCovariance
PositiveSemidefiniteCovarianceOperator = PositiveSemidefiniteCovariance


__all__ = [
    "CorrelatedGaussianLikelihood",
    "CorrelatedNoiseLikelihood",
    "CovarianceOperator",
    "DenseCovariance",
    "DenseCovarianceOperator",
    "FullCovarianceLikelihood",
    "LowRankCovariance",
    "PSDCovariance",
    "PositiveSemidefiniteCovariance",
    "PositiveSemidefiniteCovarianceOperator",
]
