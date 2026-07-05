"""Reference two-stream scattering closures for thermal emission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError


@dataclass(frozen=True)
class TwoStreamScatteringDiagnostics:
    """Inputs and output of the reference two-stream effective-opacity closure."""

    total_extinction_tau: ArrayLike
    scattering_tau: ArrayLike
    transport_scattering_tau: ArrayLike
    effective_tau: ArrayLike
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = _readonly_like(self.total_extinction_tau, "total_extinction_tau")
        scattering = _readonly_like(self.scattering_tau, "scattering_tau", total.shape)
        transport = _readonly_like(
            self.transport_scattering_tau,
            "transport_scattering_tau",
            total.shape,
        )
        effective = _readonly_like(self.effective_tau, "effective_tau", total.shape)
        if np.any(scattering - total > 1.0e-12):
            raise RobertValidationError("scattering_tau cannot exceed total_extinction_tau")
        if np.any(effective < total):
            raise RobertValidationError("effective_tau must be at least total_extinction_tau")

        object.__setattr__(self, "total_extinction_tau", total)
        object.__setattr__(self, "scattering_tau", scattering)
        object.__setattr__(self, "transport_scattering_tau", transport)
        object.__setattr__(self, "effective_tau", effective)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def absorption_tau(self) -> NDArray[np.float64]:
        """Absorbing part of the layer optical depth."""

        tau = np.maximum(self.total_extinction_tau - self.scattering_tau, 0.0)
        tau.setflags(write=False)
        return tau

    @property
    def single_scattering_albedo(self) -> NDArray[np.float64]:
        """Layer single-scattering albedo implied by the input optical depths."""

        omega = np.divide(
            self.scattering_tau,
            self.total_extinction_tau,
            out=np.zeros_like(self.scattering_tau),
            where=self.total_extinction_tau > 0.0,
        )
        omega.setflags(write=False)
        return omega


def two_stream_effective_optical_depth(
    total_extinction_tau: ArrayLike,
    scattering_tau: ArrayLike,
    transport_scattering_tau: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Return a conservative two-stream effective optical depth.

    This is ROBERT's first multiple-scattering reference closure. It preserves
    the extinction optical depth in the no-scattering and pure-scattering limits
    and adds a diffusion-style path-length boost where absorption and transport
    scattering coexist. It is intentionally simple and separately named so a
    fuller scattering solver can replace it without changing cloud inputs.
    """

    diagnostics = two_stream_scattering_diagnostics(
        total_extinction_tau,
        scattering_tau,
        transport_scattering_tau,
    )
    return diagnostics.effective_tau


def two_stream_scattering_diagnostics(
    total_extinction_tau: ArrayLike,
    scattering_tau: ArrayLike,
    transport_scattering_tau: ArrayLike | None = None,
) -> TwoStreamScatteringDiagnostics:
    """Return diagnostics for the reference two-stream closure."""

    total = _readonly_like(total_extinction_tau, "total_extinction_tau")
    scattering = _readonly_like(scattering_tau, "scattering_tau", total.shape)
    if transport_scattering_tau is None:
        transport = scattering
    else:
        transport = _readonly_like(transport_scattering_tau, "transport_scattering_tau", total.shape)
    if np.any(scattering - total > 1.0e-12):
        raise RobertValidationError("scattering_tau cannot exceed total_extinction_tau")
    absorption = np.maximum(total - scattering, 0.0)
    boost = np.sqrt(np.maximum(3.0 * absorption * transport, 0.0))
    effective = total + boost
    effective.setflags(write=False)
    return TwoStreamScatteringDiagnostics(
        total_extinction_tau=total,
        scattering_tau=scattering,
        transport_scattering_tau=transport,
        effective_tau=effective,
        metadata={
            "closure": "two_stream_effective_extinction_reference",
            "boost": "sqrt(3*tau_abs*tau_transport_scat)",
        },
    )


def _readonly_like(
    values: ArrayLike,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if shape is not None and array.shape != shape:
        raise RobertValidationError(f"{name} has incorrect shape")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    if np.any(array < 0.0):
        raise RobertValidationError(f"{name} must be non-negative")
    array.setflags(write=False)
    return array
