"""Disc-integration geometry helpers for radiative transfer."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping


@dataclass(frozen=True)
class DiscPoint:
    """One projected-disc quadrature point for an emission calculation."""

    emission_mu: float
    weight: float
    emission_angle_deg: float | None = None
    stellar_mu: float | None = None
    stellar_zenith_deg: float | None = None
    stellar_azimuth_deg: float | None = None
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    projected_radius: float | None = None
    projected_azimuth_deg: float | None = None

    def __post_init__(self) -> None:
        emission_mu = _finite_float(self.emission_mu, "emission_mu")
        if emission_mu <= 0.0 or emission_mu > 1.0:
            raise RobertValidationError("emission_mu must be in the interval (0, 1]")
        weight = _finite_float(self.weight, "weight")
        if weight < 0.0:
            raise RobertValidationError("disc geometry weights must be non-negative")

        emission_angle_deg = self.emission_angle_deg
        if emission_angle_deg is None:
            emission_angle_deg = float(np.degrees(np.arccos(emission_mu)))
        else:
            emission_angle_deg = _finite_float(emission_angle_deg, "emission_angle_deg")
            if emission_angle_deg < 0.0 or emission_angle_deg >= 90.0:
                raise RobertValidationError("emission_angle_deg must be in the interval [0, 90)")

        stellar_zenith_deg = _optional_finite(self.stellar_zenith_deg, "stellar_zenith_deg")
        stellar_mu = _optional_finite(self.stellar_mu, "stellar_mu")
        if stellar_mu is None and stellar_zenith_deg is not None:
            stellar_mu = float(np.cos(np.radians(stellar_zenith_deg)))
        if stellar_mu is not None and (stellar_mu < -1.0 or stellar_mu > 1.0):
            raise RobertValidationError("stellar_mu must be in the interval [-1, 1]")

        latitude_deg = _optional_finite(self.latitude_deg, "latitude_deg")
        longitude_deg = _optional_finite(self.longitude_deg, "longitude_deg")
        stellar_azimuth_deg = _optional_finite(self.stellar_azimuth_deg, "stellar_azimuth_deg")
        projected_radius = _optional_finite(self.projected_radius, "projected_radius")
        if projected_radius is not None and (projected_radius < 0.0 or projected_radius > 1.0):
            raise RobertValidationError("projected_radius must be in the interval [0, 1]")
        projected_azimuth_deg = _optional_finite(self.projected_azimuth_deg, "projected_azimuth_deg")

        object.__setattr__(self, "emission_mu", emission_mu)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "emission_angle_deg", emission_angle_deg)
        object.__setattr__(self, "stellar_mu", stellar_mu)
        object.__setattr__(self, "stellar_zenith_deg", stellar_zenith_deg)
        object.__setattr__(self, "stellar_azimuth_deg", stellar_azimuth_deg)
        object.__setattr__(self, "latitude_deg", latitude_deg)
        object.__setattr__(self, "longitude_deg", longitude_deg)
        object.__setattr__(self, "projected_radius", projected_radius)
        object.__setattr__(self, "projected_azimuth_deg", projected_azimuth_deg)


@dataclass(frozen=True)
class DiscGeometry:
    """Quadrature geometry for point spectra and disc-averaged emission."""

    points: Sequence[DiscPoint]
    name: str = "custom_disc"
    quadrature: str = "custom"
    phase_angle_deg: float | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if len(points) == 0:
            raise RobertValidationError("disc geometry must contain at least one point")
        if not str(self.name).strip():
            raise RobertValidationError("disc geometry name must not be empty")
        if not str(self.quadrature).strip():
            raise RobertValidationError("disc geometry quadrature must not be empty")
        phase_angle_deg = _optional_finite(self.phase_angle_deg, "phase_angle_deg")

        total_weight = float(sum(point.weight for point in points))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise RobertValidationError("disc geometry weights must have a finite positive sum")
        normalized_points = tuple(
            replace(point, weight=float(point.weight / total_weight)) for point in points
        )

        object.__setattr__(self, "points", normalized_points)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "quadrature", str(self.quadrature))
        object.__setattr__(self, "phase_angle_deg", phase_angle_deg)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def n_points(self) -> int:
        """Number of point spectra represented by this geometry."""

        return len(self.points)

    @property
    def emission_angle_cosines(self) -> NDArray[np.float64]:
        """Emission zenith cosines for all disc points."""

        return _readonly([point.emission_mu for point in self.points])

    @property
    def emission_angle_weights(self) -> NDArray[np.float64]:
        """Normalized disc-integration weights for all disc points."""

        return _readonly([point.weight for point in self.points])

    @property
    def emission_angles_deg(self) -> NDArray[np.float64]:
        """Emission zenith angles in degrees."""

        return _readonly([point.emission_angle_deg for point in self.points])

    @property
    def stellar_mu(self) -> NDArray[np.float64]:
        """Cosines of stellar zenith angles, or NaN when not defined."""

        return _readonly_optional([point.stellar_mu for point in self.points])

    @property
    def stellar_zenith_deg(self) -> NDArray[np.float64]:
        """Stellar zenith angles in degrees, or NaN when not defined."""

        return _readonly_optional([point.stellar_zenith_deg for point in self.points])

    @property
    def stellar_azimuth_deg(self) -> NDArray[np.float64]:
        """Stellar azimuth angles in degrees, or NaN when not defined."""

        return _readonly_optional([point.stellar_azimuth_deg for point in self.points])

    @property
    def latitudes_deg(self) -> NDArray[np.float64]:
        """Planetocentric latitudes in degrees, or NaN when not defined."""

        return _readonly_optional([point.latitude_deg for point in self.points])

    @property
    def longitudes_deg(self) -> NDArray[np.float64]:
        """Planetocentric longitudes in degrees, or NaN when not defined."""

        return _readonly_optional([point.longitude_deg for point in self.points])

    @property
    def projected_radius(self) -> NDArray[np.float64]:
        """Projected disc radii, or NaN when not defined."""

        return _readonly_optional([point.projected_radius for point in self.points])

    @property
    def projected_azimuth_deg(self) -> NDArray[np.float64]:
        """Projected disc azimuths in degrees, or NaN when not defined."""

        return _readonly_optional([point.projected_azimuth_deg for point in self.points])


def normal_emission_geometry() -> DiscGeometry:
    """Return a single normal-emission point with unit weight."""

    return DiscGeometry(
        points=(DiscPoint(emission_mu=1.0, weight=1.0, projected_radius=0.0),),
        name="normal_emission",
        quadrature="single_point",
    )


def geometry_from_emission_angles(
    emission_angle_cosines: ArrayLike,
    emission_angle_weights: ArrayLike,
    *,
    name: str = "emission_angle_quadrature",
    quadrature: str = "custom_mu",
    metadata: Mapping[str, str] | None = None,
) -> DiscGeometry:
    """Build a geometry from emission-angle cosines and integration weights."""

    mu = _readonly_1d(emission_angle_cosines, "emission_angle_cosines")
    weights = _readonly_1d(emission_angle_weights, "emission_angle_weights")
    if mu.shape != weights.shape:
        raise RobertValidationError("emission angle cosines and weights must have the same shape")
    points = tuple(
        DiscPoint(
            emission_mu=float(mu_value),
            weight=float(weight),
            projected_radius=float(np.sqrt(max(0.0, 1.0 - mu_value**2))),
        )
        for mu_value, weight in zip(mu, weights)
    )
    return DiscGeometry(
        points=points,
        name=name,
        quadrature=quadrature,
        metadata={} if metadata is None else metadata,
    )


def gauss_legendre_disk_geometry(n_mu: int = 4) -> DiscGeometry:
    """Return a uniform thermal-disc geometry using Gauss-Legendre mu quadrature."""

    n = _positive_int(n_mu, "n_mu")
    nodes, weights = np.polynomial.legendre.leggauss(n)
    mu = 0.5 * (nodes + 1.0)
    integral_weights = 0.5 * weights
    disk_weights = 2.0 * mu * integral_weights
    return geometry_from_emission_angles(
        mu,
        disk_weights,
        name="uniform_thermal_disc",
        quadrature="gauss_legendre_mu",
        metadata={"weight_integral": "2 * integral_0^1 I(mu) * mu dmu"},
    )


def lobatto_phase_geometry(
    phase_angle_deg: float,
    n_mu: int = 4,
) -> DiscGeometry:
    """Return a projected-disc phase quadrature using Lobatto mu rings.

    The orbital phase convention is 0 degrees at primary transit and 180
    degrees at secondary eclipse. The returned points carry stellar zenith and
    azimuth metadata for scattering source-function calculations.
    """

    n = _positive_int(n_mu, "n_mu")
    if n < 2 or n > 5:
        raise RobertValidationError("Lobatto phase geometry supports 2 <= n_mu <= 5")
    phase = _finite_float(phase_angle_deg, "phase_angle_deg") % 360.0
    mu, wtmu = _lobatto_mu_weights(n)
    dtr = np.pi / 180.0
    del_r = 1.0 / n
    z_term = np.linspace(-1.0, 1.0, 201)
    if 0.0 <= phase <= 180.0:
        theta_term = 2.0 * np.pi - np.arccos(z_term)
    else:
        theta_term = np.arccos(z_term)
    x_term = np.sin(theta_term) * np.around(np.cos(phase * dtr), 14)
    r_term = np.sqrt(x_term**2 + z_term**2)
    r_min = float(np.min(r_term))

    points: list[DiscPoint] = []
    for mu_value, mu_weight in zip(mu, wtmu):
        r_quad = float(np.sqrt(max(0.0, 1.0 - mu_value**2)))
        half_circumference = float(np.pi * r_quad)
        if r_quad > r_min:
            keep = np.where(r_term <= r_quad)[0]
            intersections = np.array([keep[0], keep[-1]])
            x_intersect = x_term[intersections]
            z_intersect = z_term[intersections]
            if z_intersect[1] > 0.0:
                alpha_intersect = _azimuth_angle(x_intersect[1], z_intersect[1]) / dtr
            else:
                alpha_intersect = _azimuth_angle(x_intersect[0], z_intersect[0]) / dtr
            nalpha1 = int(0.5 + half_circumference * (alpha_intersect / 180.0) / del_r)
            nalpha2 = int(0.5 + half_circumference * ((180.0 - alpha_intersect) / 180.0) / del_r)
            nalpha1 = max(nalpha1, 2)
            nalpha2 = max(nalpha2, 2)
            alpha1 = alpha_intersect / (nalpha1 - 1) * np.arange(nalpha1)
            alpha2 = alpha_intersect + (180.0 - alpha_intersect) / (nalpha2 - 1) * np.arange(nalpha2)
            alpha_samples = np.concatenate((alpha1, alpha2[1:]))
        else:
            if half_circumference > 0.0:
                nalpha = int(0.5 + half_circumference / del_r)
                alpha_samples = 180.0 * np.arange(nalpha) / (nalpha - 1)
            else:
                alpha_samples = np.array([0.0], dtype=float)

        if alpha_samples.size == 1:
            alpha = float(alpha_samples[0])
            stellar_zenith, stellar_azimuth, latitude, longitude = _phase_point_angles(
                phase,
                r_quad,
                alpha,
            )
            if np.isclose(np.degrees(np.arccos(mu_value)), 0.0):
                stellar_azimuth = 180.0
            points.append(
                DiscPoint(
                    emission_mu=float(mu_value),
                    weight=float(2.0 * mu_value * mu_weight),
                    stellar_zenith_deg=stellar_zenith,
                    stellar_azimuth_deg=stellar_azimuth,
                    latitude_deg=latitude,
                    longitude_deg=longitude % 360.0,
                    projected_radius=r_quad,
                    projected_azimuth_deg=alpha,
                )
            )
            continue

        for index, alpha in enumerate(alpha_samples):
            if index == 0:
                trap_weight = (alpha_samples[index + 1] - alpha_samples[index]) / 2.0
            elif index == alpha_samples.size - 1:
                trap_weight = (alpha_samples[index] - alpha_samples[index - 1]) / 2.0
            else:
                trap_weight = (alpha_samples[index + 1] - alpha_samples[index - 1]) / 2.0
            azimuth_weight = trap_weight / 180.0
            stellar_zenith, stellar_azimuth, latitude, longitude = _phase_point_angles(
                phase,
                r_quad,
                float(alpha),
            )
            points.append(
                DiscPoint(
                    emission_mu=float(mu_value),
                    weight=float(2.0 * mu_value * mu_weight * azimuth_weight),
                    stellar_zenith_deg=stellar_zenith,
                    stellar_azimuth_deg=stellar_azimuth,
                    latitude_deg=latitude,
                    longitude_deg=longitude % 360.0,
                    projected_radius=r_quad,
                    projected_azimuth_deg=float(alpha),
                )
            )

    return DiscGeometry(
        points=tuple(points),
        name="lobatto_phase_disc",
        quadrature="lobatto_projected_disc",
        phase_angle_deg=phase,
        metadata={
            "phase_convention": "0 deg primary transit, 180 deg secondary eclipse",
            "disc_model": "projected_disc_lobatto_mu_trapezoid_azimuth",
        },
    )


def _phase_point_angles(
    phase_deg: float,
    rho: float,
    alpha_deg: float,
) -> tuple[float, float, float, float]:
    dtr = np.pi / 180.0
    theta_star = np.pi / 2.0
    phi_star_deg = 90.0 + phase_deg
    v_star = np.array(
        [
            np.sin(theta_star) * np.cos(phi_star_deg * dtr),
            np.sin(theta_star) * np.sin(phi_star_deg * dtr),
            np.cos(theta_star),
        ],
        dtype=float,
    )

    theta_point = np.arccos(np.clip(rho * np.sin(alpha_deg * dtr), -1.0, 1.0))
    if not np.isclose(np.sin(theta_point), 0.0):
        cos_phi = rho * np.cos(alpha_deg * dtr) / abs(np.sin(theta_point))
        phi_point = (-np.arccos(np.clip(cos_phi, -1.0, 1.0))) % (2.0 * np.pi)
    else:
        phi_point = 0.0
    v_point = np.array(
        [
            np.sin(theta_point) * np.cos(phi_point),
            np.sin(theta_point) * np.sin(phi_point),
            np.cos(theta_point),
        ],
        dtype=float,
    )

    stellar_zenith = float(np.degrees(np.arccos(np.clip(np.sum(v_star * v_point), -1.0, 1.0))))
    latitude = float(np.degrees(np.pi / 2.0 - theta_point))
    longitude = float((phi_point / dtr - (phi_star_deg + 180.0)) % 360.0)

    v_observer = np.array([0.0, -1.0, 0.0], dtype=float)
    v_star_1 = _rotate_z(v_star, -phi_point)
    v_observer_1 = _rotate_z(v_observer, -phi_point)
    v_star_local = _rotate_y(v_star_1, -theta_point)
    v_observer_local = _rotate_y(v_observer_1, -theta_point)
    phi_star_local = _azimuth_angle(v_star_local[0], v_star_local[1])
    phi_observer_local = _azimuth_angle(v_observer_local[0], v_observer_local[1])
    azimuth = abs(phi_observer_local - phi_star_local)
    if azimuth > np.pi:
        azimuth = 2.0 * np.pi - azimuth
    stellar_azimuth = float(np.degrees(np.pi - azimuth))
    return stellar_zenith, stellar_azimuth, latitude, longitude


def _lobatto_mu_weights(n_mu: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = {
        2: (
            [0.447213595499958, 1.000000],
            [0.8333333333333333, 0.166666666666666666],
        ),
        3: (
            [0.28523151648064509, 0.7650553239294646, 1.0000],
            [0.5548583770354863, 0.3784749562978469, 0.06666666666666666],
        ),
        4: (
            [0.2092992179024788, 0.5917001814331423, 0.8717401485096066, 1.00000],
            [0.4124587946587038, 0.3411226924835043, 0.2107042271435060, 0.035714285714285],
        ),
        5: (
            [0.165278957666387, 0.477924949810444, 0.738773865105505, 0.919533908166459, 1.0],
            [0.327539761183898, 0.292042683679684, 0.224889342063117, 0.133305990851069, 0.0222222222222222],
        ),
    }
    mu, weights = values[n_mu]
    return np.asarray(mu, dtype=float), np.asarray(weights, dtype=float)


def _azimuth_angle(x: float, y: float) -> float:
    return float(np.mod(np.arctan2(y, x), 2.0 * np.pi))


def _rotate_y(vector: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    return np.array(
        [
            np.cos(angle) * vector[0] + np.sin(angle) * vector[2],
            vector[1],
            -np.sin(angle) * vector[0] + np.cos(angle) * vector[2],
        ],
        dtype=float,
    )


def _rotate_z(vector: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    return np.array(
        [
            np.cos(angle) * vector[0] - np.sin(angle) * vector[1],
            np.sin(angle) * vector[0] + np.cos(angle) * vector[1],
            vector[2],
        ],
        dtype=float,
    )


def _positive_int(value: int, name: str) -> int:
    number = int(value)
    if number < 1:
        raise RobertValidationError(f"{name} must be at least one")
    return number


def _finite_float(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise RobertValidationError(f"{name} must be finite")
    return number


def _optional_finite(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, name)


def _readonly(values: Sequence[float | None]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    array.setflags(write=False)
    return array


def _readonly_optional(values: Sequence[float | None]) -> NDArray[np.float64]:
    array = np.asarray([np.nan if value is None else value for value in values], dtype=float)
    array.setflags(write=False)
    return array


def _readonly_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise RobertValidationError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise RobertValidationError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array
