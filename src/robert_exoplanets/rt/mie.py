"""Composition-agnostic spherical-particle cloud optics.

The public boundary in this module is the complex refractive index
``m = n + i k`` with ``k >= 0``.  Material identity is deliberately kept in
metadata: radiative transfer only receives the optical properties implied by
``n``, ``k``, particle size, and particle density.
"""

from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertDataError, RobertValidationError, SpectralGrid
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.opacity import spectral_grid_values_in_unit

from .clouds import CloudOpticalProperties
from .optical_depth import GasOpticalDepth


@dataclass(frozen=True)
class RefractiveIndexSpectrum:
    """Real and imaginary refractive index tabulated against wavelength.

    Positive imaginary index follows the passive-medium convention used by
    the Mie solver. Interpolation is linear in log wavelength and in ``n`` and
    ``log(k)`` where both neighbouring ``k`` values are positive. Exact zeros
    remain supported for non-absorbing reference cases.
    """

    wavelength_micron: ArrayLike
    real_index: ArrayLike
    imaginary_index: ArrayLike
    name: str = "refractive index"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wavelength = _readonly_1d(self.wavelength_micron, "wavelength_micron")
        real = _readonly_1d(self.real_index, "real_index")
        imaginary = _readonly_1d(self.imaginary_index, "imaginary_index")
        if wavelength.size < 1 or real.shape != wavelength.shape or imaginary.shape != wavelength.shape:
            raise RobertValidationError("refractive-index arrays must have the same non-empty shape")
        if np.any(wavelength <= 0.0) or np.any(np.diff(wavelength) <= 0.0):
            raise RobertValidationError("refractive-index wavelengths must be positive and increasing")
        if np.any(real <= 0.0):
            raise RobertValidationError("real refractive index must be positive")
        if np.any(imaginary < 0.0):
            raise RobertValidationError("imaginary refractive index must be non-negative")
        if not str(self.name).strip():
            raise RobertValidationError("refractive-index name must not be empty")
        object.__setattr__(self, "wavelength_micron", wavelength)
        object.__setattr__(self, "real_index", real)
        object.__setattr__(self, "imaginary_index", imaginary)
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    def evaluate(
        self,
        wavelength_micron: ArrayLike,
        *,
        extrapolation: str = "raise",
    ) -> NDArray[np.complex128]:
        """Interpolate the complex index onto positive target wavelengths."""

        target = np.asarray(wavelength_micron, dtype=float)
        if not np.all(np.isfinite(target)) or np.any(target <= 0.0):
            raise RobertValidationError("target refractive-index wavelengths must be finite and positive")
        mode = str(extrapolation).strip().lower()
        if mode not in {"raise", "clip"}:
            raise RobertValidationError("refractive-index extrapolation must be 'raise' or 'clip'")
        if mode == "raise" and (
            np.any(target < self.wavelength_micron[0]) or np.any(target > self.wavelength_micron[-1])
        ):
            raise RobertValidationError("target wavelength lies outside refractive-index coverage")
        clipped = np.clip(target, self.wavelength_micron[0], self.wavelength_micron[-1])
        log_source = np.log(self.wavelength_micron)
        log_target = np.log(clipped)
        real = np.interp(log_target, log_source, self.real_index)
        if np.all(self.imaginary_index > 0.0):
            imaginary = np.exp(np.interp(log_target, log_source, np.log(self.imaginary_index)))
        else:
            imaginary = np.interp(log_target, log_source, self.imaginary_index)
        result = np.asarray(real + 1j * imaginary, dtype=np.complex128)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class MieParticleOptics:
    """Size-distribution-averaged optical properties per condensate mass."""

    wavelength_micron: ArrayLike
    mass_extinction_m2_kg: ArrayLike
    mass_scattering_m2_kg: ArrayLike
    single_scattering_albedo: ArrayLike
    asymmetry_factor: ArrayLike
    phase_function_moments: ArrayLike
    effective_radius_micron: float
    geometric_stddev: float
    particle_density_kg_m3: float
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wavelength = _readonly_1d(self.wavelength_micron, "wavelength_micron")
        extinction = _readonly_1d(self.mass_extinction_m2_kg, "mass_extinction_m2_kg")
        scattering = _readonly_1d(self.mass_scattering_m2_kg, "mass_scattering_m2_kg")
        albedo = _readonly_1d(self.single_scattering_albedo, "single_scattering_albedo")
        asymmetry = _readonly_1d(self.asymmetry_factor, "asymmetry_factor")
        if any(array.shape != wavelength.shape for array in (extinction, scattering, albedo, asymmetry)):
            raise RobertValidationError("Mie optical-property arrays must match wavelength")
        if np.any(extinction < 0.0) or np.any(scattering < 0.0) or np.any(scattering > extinction * (1.0 + 1e-10)):
            raise RobertValidationError("Mie extinction and scattering coefficients are inconsistent")
        if np.any(albedo < 0.0) or np.any(albedo > 1.0):
            raise RobertValidationError("Mie single-scattering albedo must lie in [0, 1]")
        if np.any(asymmetry < -1.0) or np.any(asymmetry > 1.0):
            raise RobertValidationError("Mie asymmetry factor must lie in [-1, 1]")
        moments = np.array(self.phase_function_moments, dtype=float, copy=True)
        if moments.shape != (5, wavelength.size) or not np.all(np.isfinite(moments)):
            raise RobertValidationError(
                "Mie phase_function_moments must have shape (5, wavelength)"
            )
        limits = (2.0 * np.arange(5) + 1.0)[:, None]
        if np.any(np.abs(moments) > limits * (1.0 + 1.0e-10)):
            raise RobertValidationError("Mie phase-function moments exceed physical bounds")
        if not np.allclose(moments[0], 1.0, rtol=0.0, atol=1.0e-12):
            raise RobertValidationError("Mie zeroth phase-function moment must equal one")
        if not np.allclose(moments[1] / 3.0, asymmetry, rtol=2.0e-8, atol=2.0e-10):
            raise RobertValidationError(
                "Mie first phase-function moment must match asymmetry factor"
            )
        radius = _positive_scalar(self.effective_radius_micron, "effective_radius_micron")
        width = float(self.geometric_stddev)
        density = _positive_scalar(self.particle_density_kg_m3, "particle_density_kg_m3")
        if not np.isfinite(width) or width < 1.0:
            raise RobertValidationError("geometric_stddev must be finite and at least one")
        object.__setattr__(self, "wavelength_micron", wavelength)
        object.__setattr__(self, "mass_extinction_m2_kg", extinction)
        object.__setattr__(self, "mass_scattering_m2_kg", scattering)
        object.__setattr__(self, "single_scattering_albedo", albedo)
        object.__setattr__(self, "asymmetry_factor", asymmetry)
        moments.setflags(write=False)
        object.__setattr__(self, "phase_function_moments", moments)
        object.__setattr__(self, "effective_radius_micron", radius)
        object.__setattr__(self, "geometric_stddev", width)
        object.__setattr__(self, "particle_density_kg_m3", density)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class OpticalConstantsCatalog:
    """A named directory of optical-constant tables.

    Tables are selected by filename stem, keeping the material catalogue
    independent of the cloud physics. The initial ROBERT catalogue uses the
    Exo Skryer five-line/header convention, but callers may also point this
    object at compatible collections assembled elsewhere.
    """

    root: str | Path
    source_format: str = "exo_skryer"

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        if not root.is_dir():
            raise RobertDataError(f"optical-constants catalogue does not exist: {root}")
        source_format = str(self.source_format).strip().lower()
        if source_format not in {"exo_skryer", "headerless"}:
            raise RobertValidationError("catalogue source_format must be 'exo_skryer' or 'headerless'")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "source_format", source_format)

    @property
    def materials(self) -> tuple[str, ...]:
        """Sorted material names available for selection."""

        return tuple(
            sorted(
                path.stem
                for path in Path(self.root).glob("*.txt")
                if not path.name.upper().startswith(("LICENSE", "README", "NOTICE"))
                and path.stem != "White"
            )
        )

    def path_for(self, material: str) -> Path:
        """Resolve one exact, case-sensitive material name safely."""

        name = str(material).strip()
        if not name or Path(name).name != name:
            raise RobertValidationError("optical-constant material must be a filename stem")
        path = Path(self.root) / f"{name}.txt"
        if not path.is_file():
            raise RobertDataError(
                f"optical-constant material is not in catalogue: {name}; "
                f"available: {', '.join(self.materials)}"
            )
        return path

    def load(self, material: str) -> RefractiveIndexSpectrum:
        """Load a selected material with source checksum provenance."""

        path = self.path_for(material)
        if self.source_format == "exo_skryer":
            return load_exo_skryer_refractive_index(path, name=material)
        return load_refractive_index_table(path, name=material)


def load_refractive_index_table(
    path: str | Path,
    *,
    wavelength_unit: str = "micron",
    delimiter: str | None = None,
    skiprows: int = 0,
    columns: tuple[int, int, int] = (0, 1, 2),
    name: str | None = None,
) -> RefractiveIndexSpectrum:
    """Read a headerless wavelength, ``n``, ``k`` text table.

    ``wavelength_unit`` may be micron, nm, m, or cm^-1. Extra columns are
    allowed and selected with ``columns``.
    """

    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise RobertDataError(f"refractive-index file does not exist: {file_path}")
    try:
        values = np.loadtxt(file_path, delimiter=delimiter, skiprows=int(skiprows), ndmin=2)
    except (OSError, ValueError) as exc:
        raise RobertDataError(f"could not read refractive-index table: {file_path}") from exc
    if len(columns) != 3 or min(columns) < 0 or max(columns) >= values.shape[1]:
        raise RobertValidationError("refractive-index columns must select wavelength, n, and k")
    wavelength = _wavelength_to_micron(values[:, columns[0]], wavelength_unit)
    order = np.argsort(wavelength)
    return RefractiveIndexSpectrum(
        wavelength_micron=wavelength[order],
        real_index=values[order, columns[1]],
        imaginary_index=values[order, columns[2]],
        name=name or file_path.stem,
        metadata={"source_path": str(file_path), "source_format": "text_n_k", "wavelength_unit": wavelength_unit},
    )


def load_refractive_index_csv(
    path: str | Path,
    *,
    wavelength_unit: str = "micron",
    name: str | None = None,
) -> RefractiveIndexSpectrum:
    """Read a headed CSV using common wavelength, ``n``, and ``k`` aliases."""

    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise RobertDataError(f"refractive-index file does not exist: {file_path}")
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RobertDataError("refractive-index CSV is missing a header")
        wavelength_key = _column(reader.fieldnames, ("wavelength_micron", "wavelength_um", "wavelength_nm", "wavelength", "lambda"), "wavelength")
        real_key = _column(reader.fieldnames, ("n", "real_index", "real", "refractive_index_real"), "real refractive index")
        imaginary_key = _column(reader.fieldnames, ("k", "imaginary_index", "imaginary", "refractive_index_imaginary"), "imaginary refractive index")
        rows = tuple(reader)
    if not rows:
        raise RobertDataError("refractive-index CSV is empty")
    inferred_unit = "nm" if wavelength_key.lower().endswith("_nm") else wavelength_unit
    wavelength = _wavelength_to_micron([float(row[wavelength_key]) for row in rows], inferred_unit)
    order = np.argsort(wavelength)
    return RefractiveIndexSpectrum(
        wavelength_micron=wavelength[order],
        real_index=np.asarray([float(row[real_key]) for row in rows])[order],
        imaginary_index=np.asarray([float(row[imaginary_key]) for row in rows])[order],
        name=name or file_path.stem,
        metadata={"source_path": str(file_path), "source_format": "csv_n_k", "wavelength_unit": inferred_unit},
    )


def load_exo_skryer_refractive_index(
    path: str | Path,
    *,
    name: str | None = None,
) -> RefractiveIndexSpectrum:
    """Read an Exo Skryer/LX-MIE ``nk_data`` table.

    The first row records the expected row count and a conducting-material
    flag. Remaining comment rows preserve the original laboratory citation;
    numeric rows contain wavelength in micron followed by ``n`` and ``k``.
    """

    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise RobertDataError(f"refractive-index file does not exist: {file_path}")
    try:
        first_line = file_path.read_text(encoding="utf-8").splitlines()[0]
        header = first_line.split("#", 1)[0].split()
        expected_rows = int(header[0])
        conducting = header[1].strip().strip(".").lower() == "true" if len(header) > 1 else False
        values = np.loadtxt(file_path, comments="#", skiprows=1, ndmin=2)
    except (IndexError, OSError, ValueError) as exc:
        raise RobertDataError(f"could not read Exo Skryer refractive-index table: {file_path}") from exc
    if values.shape[1] < 3:
        raise RobertDataError(f"Exo Skryer refractive-index table requires wavelength, n, k: {file_path}")
    wavelength = np.asarray(values[:, 0], dtype=float)
    order = np.argsort(wavelength)
    wavelength, real, imaginary, duplicates = _merge_duplicate_wavelengths(
        wavelength[order], values[order, 1], values[order, 2]
    )
    return RefractiveIndexSpectrum(
        wavelength_micron=wavelength,
        real_index=real,
        imaginary_index=imaginary,
        name=name or file_path.stem,
        metadata={
            "source_path": str(file_path),
            "source_format": "exo_skryer_nk_data",
            "source_sha256": _sha256(file_path),
            "conducting": str(conducting).lower(),
            "declared_rows": str(expected_rows),
            "parsed_rows": str(values.shape[0]),
            "row_count_matches_header": str(values.shape[0] == expected_rows).lower(),
            "duplicate_wavelength_rows_averaged": str(duplicates),
        },
    )


def mie_efficiencies(
    size_parameter: float,
    refractive_index: complex,
) -> tuple[float, float, float]:
    """Return spherical-particle ``(Qext, Qsca, g)`` for one size parameter."""

    q_ext, q_sca, g, _, _ = _mie_solution(size_parameter, refractive_index)
    return q_ext, q_sca, g


def mie_phase_function_moments(
    size_parameter: float,
    refractive_index: complex,
    *,
    order: int = 5,
) -> NDArray[np.float64]:
    """Return scalar Mie phase-function Legendre coefficients.

    The convention is ``P(mu)=sum_l chi_l P_l(mu)`` with ``chi_0=1`` and
    ``chi_1=3g``. Five terms are used by ROBERT so P3/SH4 retains degrees
    zero through three and degree four supplies the delta-M forward fraction.
    """

    if order < 1:
        raise RobertValidationError("Mie phase-function moment order must be positive")
    _, q_sca, _, a, b = _mie_solution(size_parameter, refractive_index)
    if q_sca <= 0.0:
        moments = np.zeros(order)
        moments[0] = 1.0
        return moments
    if a is None or b is None:
        moments = np.zeros(order)
        moments[0] = 1.0
        if order > 2:
            moments[2] = 0.5
        return moments
    return _phase_moments_from_coefficients(a, b, order=order)


def _mie_solution(
    size_parameter: float,
    refractive_index: complex,
) -> tuple[
    float,
    float,
    float,
    NDArray[np.complex128] | None,
    NDArray[np.complex128] | None,
]:
    x = _positive_scalar(size_parameter, "size_parameter")
    m = complex(refractive_index)
    if not np.isfinite(m.real) or not np.isfinite(m.imag) or m.real <= 0.0 or m.imag < 0.0:
        raise RobertValidationError("complex refractive index requires finite n > 0 and k >= 0")
    if x < 1.0e-3:
        polarizability = (m * m - 1.0) / (m * m + 2.0)
        q_sca = float((8.0 / 3.0) * x**4 * abs(polarizability) ** 2)
        q_abs = float(max(0.0, 4.0 * x * polarizability.imag))
        return q_abs + q_sca, q_sca, 0.0, None, None

    n_stop = max(1, int(np.floor(x + 4.05 * x ** (1.0 / 3.0) + 2.0)))
    z = m * x
    n_down = max(n_stop + 15, int(abs(z)) + 15)
    derivative = np.zeros(n_down + 1, dtype=np.complex128)
    for n in range(n_down, 0, -1):
        nx = n / z
        derivative[n - 1] = nx - 1.0 / (derivative[n] + nx)

    psi_nm1 = np.sin(x)
    psi_n = psi_nm1 / x - np.cos(x)
    chi_nm1 = np.cos(x)
    chi_n = chi_nm1 / x + np.sin(x)
    xi_nm1 = psi_nm1 - 1j * chi_nm1
    xi_n = psi_n - 1j * chi_n
    a = np.empty(n_stop, dtype=np.complex128)
    b = np.empty(n_stop, dtype=np.complex128)
    for index, n in enumerate(range(1, n_stop + 1)):
        d = derivative[n]
        a_factor = d / m + n / x
        b_factor = m * d + n / x
        a[index] = (a_factor * psi_n - psi_nm1) / (a_factor * xi_n - xi_nm1)
        b[index] = (b_factor * psi_n - psi_nm1) / (b_factor * xi_n - xi_nm1)
        psi_np1 = (2.0 * n + 1.0) * psi_n / x - psi_nm1
        chi_np1 = (2.0 * n + 1.0) * chi_n / x - chi_nm1
        psi_nm1, psi_n = psi_n, psi_np1
        chi_nm1, chi_n = chi_n, chi_np1
        xi_nm1, xi_n = xi_n, psi_n - 1j * chi_n

    orders = np.arange(1, n_stop + 1, dtype=float)
    weights = 2.0 * orders + 1.0
    q_ext = float(2.0 * np.sum(weights * np.real(a + b)) / x**2)
    q_sca = float(2.0 * np.sum(weights * (np.abs(a) ** 2 + np.abs(b) ** 2)) / x**2)
    q_ext = max(q_ext, 0.0)
    q_sca = min(max(q_sca, 0.0), q_ext * (1.0 + 1.0e-10))
    if q_sca <= 0.0:
        return q_ext, q_sca, 0.0, a, b
    adjacent = 0.0
    if n_stop > 1:
        n = orders[:-1]
        adjacent = float(np.sum(n * (n + 2.0) / (n + 1.0) * np.real(a[:-1] * np.conj(a[1:]) + b[:-1] * np.conj(b[1:]))))
    cross = float(np.sum(weights / (orders * (orders + 1.0)) * np.real(a * np.conj(b))))
    g = float(np.clip(4.0 * (adjacent + cross) / (x**2 * q_sca), -1.0, 1.0))
    return q_ext, q_sca, g, a, b


def _phase_moments_from_coefficients(
    a: NDArray[np.complex128],
    b: NDArray[np.complex128],
    *,
    order: int,
) -> NDArray[np.float64]:
    # The amplitude functions are polynomials through degree n_stop. A
    # Gauss-Legendre rule with n_stop + 2 points therefore integrates their
    # squared intensity and the low-order moments without angular gridding
    # error, up to floating-point recurrence error.
    n_angle = max(16, a.size + order + 1)
    mu, weights = _legendre_quadrature(n_angle)
    s1 = np.zeros(n_angle, dtype=np.complex128)
    s2 = np.zeros(n_angle, dtype=np.complex128)
    pi_nm1 = np.zeros(n_angle)
    pi_n = np.ones(n_angle)
    for index, n in enumerate(range(1, a.size + 1)):
        tau_n = n * mu * pi_n - (n + 1.0) * pi_nm1
        factor = (2.0 * n + 1.0) / (n * (n + 1.0))
        s1 += factor * (a[index] * pi_n + b[index] * tau_n)
        s2 += factor * (a[index] * tau_n + b[index] * pi_n)
        pi_np1 = (
            (2.0 * n + 1.0) * mu * pi_n / n
            - (n + 1.0) * pi_nm1 / n
        )
        pi_nm1, pi_n = pi_n, pi_np1
    intensity = 0.5 * (np.abs(s1) ** 2 + np.abs(s2) ** 2)
    normalization = float(np.sum(weights * intensity))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise RobertValidationError("Mie phase-function normalization is not positive")
    basis = np.polynomial.legendre.legvander(mu, order - 1).T
    moments = (2.0 * np.arange(order) + 1.0) * np.sum(
        basis * (weights * intensity)[None, :], axis=1
    ) / normalization
    moments[0] = 1.0
    return np.asarray(moments, dtype=float)


@lru_cache(maxsize=256)
def _legendre_quadrature(order: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mu, weights = np.polynomial.legendre.leggauss(order)
    mu.setflags(write=False)
    weights.setflags(write=False)
    return mu, weights


def lognormal_mie_optics(
    refractive_index: RefractiveIndexSpectrum,
    spectral_grid: SpectralGrid,
    *,
    effective_radius_micron: float,
    geometric_stddev: float,
    particle_density_kg_m3: float,
    quadrature_points: int = 24,
    extrapolation: str = "raise",
) -> MieParticleOptics:
    """Average Mie cross sections over a lognormal number distribution."""

    radius = _positive_scalar(effective_radius_micron, "effective_radius_micron")
    width = float(geometric_stddev)
    density = _positive_scalar(particle_density_kg_m3, "particle_density_kg_m3")
    points = int(quadrature_points)
    if not np.isfinite(width) or width < 1.0:
        raise RobertValidationError("geometric_stddev must be finite and at least one")
    if points < 1:
        raise RobertValidationError("quadrature_points must be positive")
    wavelength = spectral_grid_values_in_unit(spectral_grid, "micron")
    indices = refractive_index.evaluate(wavelength, extrapolation=extrapolation)
    if width == 1.0:
        radii_micron = np.array([radius])
        number_weights = np.array([1.0])
    else:
        nodes, weights = _legendre_quadrature(points)
        normal_coordinate = 6.0 * nodes
        number_weights = (
            6.0
            * weights
            * np.exp(-0.5 * normal_coordinate**2)
            / np.sqrt(2.0 * np.pi)
        )
        number_weights = number_weights / np.sum(number_weights)
        log_width = np.log(width)
        geometric_mean_radius = radius * np.exp(-2.5 * log_width**2)
        radii_micron = geometric_mean_radius * np.exp(
            log_width * normal_coordinate
        )
    radii_m = radii_micron * 1.0e-6
    particle_mass = (4.0 / 3.0) * np.pi * density * radii_m**3
    mean_mass = float(np.sum(number_weights * particle_mass))
    extinction = np.zeros(wavelength.size)
    scattering = np.zeros(wavelength.size)
    scattering_g = np.zeros(wavelength.size)
    scattering_moments = np.zeros((5, wavelength.size))
    for wave_index, (wave, index) in enumerate(zip(wavelength, indices, strict=True)):
        for particle_radius_micron, weight, particle_radius_m in zip(radii_micron, number_weights, radii_m, strict=True):
            x = 2.0 * np.pi * particle_radius_micron / wave
            q_ext, q_sca, g, a, b = _mie_solution(x, complex(index))
            if a is None or b is None:
                moments = np.array([1.0, 0.0, 0.5, 0.0, 0.0])
            else:
                moments = _phase_moments_from_coefficients(a, b, order=5)
            area = np.pi * particle_radius_m**2
            extinction[wave_index] += weight * area * q_ext
            scattering[wave_index] += weight * area * q_sca
            scattering_g[wave_index] += weight * area * q_sca * g
            scattering_moments[:, wave_index] += weight * area * q_sca * moments
    mass_extinction = extinction / mean_mass
    mass_scattering = scattering / mean_mass
    albedo = np.divide(mass_scattering, mass_extinction, out=np.zeros_like(mass_extinction), where=mass_extinction > 0.0)
    asymmetry = np.divide(scattering_g, scattering, out=np.zeros_like(scattering_g), where=scattering > 0.0)
    phase_moments = np.divide(
        scattering_moments,
        scattering[None, :],
        out=np.zeros_like(scattering_moments),
        where=scattering[None, :] > 0.0,
    )
    phase_moments[0, scattering <= 0.0] = 1.0
    # Use the quadrature-derived first moment as the single source of truth so
    # the SH4 moments and reported asymmetry are internally identical.
    asymmetry = phase_moments[1] / 3.0
    return MieParticleOptics(
        wavelength_micron=wavelength,
        mass_extinction_m2_kg=mass_extinction,
        mass_scattering_m2_kg=mass_scattering,
        single_scattering_albedo=np.clip(albedo, 0.0, 1.0),
        asymmetry_factor=np.clip(asymmetry, -1.0, 1.0),
        phase_function_moments=phase_moments,
        effective_radius_micron=radius,
        geometric_stddev=width,
        particle_density_kg_m3=density,
        metadata={
            "optical_model": "homogeneous_sphere_mie",
            "size_distribution": "lognormal_number",
            "radius_definition": "effective_radius=<r3>/<r2>",
            "refractive_index_name": refractive_index.name,
            "quadrature_points": str(points),
        },
    )


def mie_cloud_from_mass_fraction(
    gas_optical_depth: GasOpticalDepth,
    particle_optics: MieParticleOptics,
    *,
    condensate_mass_fraction: ArrayLike | float,
    name: str = "refractive-index Mie cloud",
) -> CloudOpticalProperties:
    """Convert condensate mass fraction into hydrostatic layer cloud optics."""

    if particle_optics.wavelength_micron.shape != (gas_optical_depth.spectral_grid.size,):
        raise RobertValidationError("particle optics do not match the gas spectral grid")
    expected_wavelength = spectral_grid_values_in_unit(gas_optical_depth.spectral_grid, "micron")
    if not np.allclose(particle_optics.wavelength_micron, expected_wavelength, rtol=1.0e-10, atol=0.0):
        raise RobertValidationError("particle-optics wavelengths do not match the gas spectral grid")
    mass_fraction = np.asarray(condensate_mass_fraction, dtype=float)
    if mass_fraction.ndim == 0:
        mass_fraction = np.full(gas_optical_depth.pressure_grid.n_layers, float(mass_fraction))
    if mass_fraction.shape != (gas_optical_depth.pressure_grid.n_layers,):
        raise RobertValidationError("condensate_mass_fraction must be scalar or one value per layer")
    if not np.all(np.isfinite(mass_fraction)) or np.any(mass_fraction < 0.0) or np.any(mass_fraction > 1.0):
        raise RobertValidationError("condensate_mass_fraction must lie in [0, 1]")
    bulk_mass_column = gas_optical_depth.layer_pressure_thickness_pa / gas_optical_depth.gravity_m_s2
    tau = bulk_mass_column[:, None] * mass_fraction[:, None] * particle_optics.mass_extinction_m2_kg[None, :]
    return CloudOpticalProperties(
        name=name,
        extinction_tau=tau,
        spectral_grid=gas_optical_depth.spectral_grid,
        pressure_grid=gas_optical_depth.pressure_grid,
        single_scattering_albedo=np.repeat(
            particle_optics.single_scattering_albedo[None, :],
            gas_optical_depth.pressure_grid.n_layers,
            axis=0,
        ),
        asymmetry_factor=np.repeat(
            particle_optics.asymmetry_factor[None, :],
            gas_optical_depth.pressure_grid.n_layers,
            axis=0,
        ),
        phase_function_moments=particle_optics.phase_function_moments,
        metadata={"vertical_model": "condensate_mass_fraction_per_bulk_atmosphere_mass", "hydrostatic_conversion": "tau=kappa_ext*q_cond*delta_pressure/gravity", **dict(particle_optics.metadata)},
    )


def refractive_index_from_parameters(
    wavelength_micron: ArrayLike,
    parameters: Mapping[str, float],
    *,
    real_parameter_names: Sequence[str],
    log10_imaginary_parameter_names: Sequence[str],
    name: str = "retrieved refractive index",
) -> RefractiveIndexSpectrum:
    """Build nodal ``n(lambda)`` and ``k(lambda)`` from retrieval parameters."""

    real_names = tuple(str(item) for item in real_parameter_names)
    imaginary_names = tuple(str(item) for item in log10_imaginary_parameter_names)
    wavelength = np.asarray(wavelength_micron, dtype=float)
    if len(real_names) != wavelength.size or len(imaginary_names) != wavelength.size:
        raise RobertValidationError("refractive-index parameter names must match wavelength nodes")
    missing = tuple(item for item in (*real_names, *imaginary_names) if item not in parameters)
    if missing:
        raise RobertValidationError("missing refractive-index parameters: " + ", ".join(missing))
    real = np.asarray([float(parameters[item]) for item in real_names])
    log_imaginary = np.asarray([float(parameters[item]) for item in imaginary_names])
    if not np.all(np.isfinite(log_imaginary)):
        raise RobertValidationError("log10 imaginary refractive indices must be finite")
    imaginary = np.power(10.0, log_imaginary)
    return RefractiveIndexSpectrum(wavelength, real, imaginary, name=name, metadata={"parameterization": "nodal_n_log10_k"})


def _readonly_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must be a finite one-dimensional array")
    array.setflags(write=False)
    return array


def _positive_scalar(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise RobertValidationError(f"{name} must be finite and positive")
    return number


def _wavelength_to_micron(values: ArrayLike, unit: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    normalized = str(unit).strip().lower().replace("μ", "u").replace("µ", "u")
    if normalized in {"micron", "microns", "um"}:
        result = array
    elif normalized == "nm":
        result = array * 1.0e-3
    elif normalized in {"m", "meter", "metre"}:
        result = array * 1.0e6
    elif normalized in {"cm^-1", "cm-1", "wavenumber"}:
        if np.any(array <= 0.0):
            raise RobertValidationError("wavenumber values must be positive")
        result = 1.0e4 / array
    else:
        raise RobertValidationError(f"unsupported refractive-index wavelength unit: {unit}")
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise RobertValidationError("refractive-index wavelengths must be finite and positive")
    return np.asarray(result, dtype=float)


def _column(fieldnames: Sequence[str], aliases: Sequence[str], label: str) -> str:
    lookup = {str(item).strip().lower(): str(item) for item in fieldnames}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    raise RobertDataError(f"refractive-index CSV is missing {label} column")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _merge_duplicate_wavelengths(
    wavelength: NDArray[np.float64],
    real: NDArray[np.float64],
    imaginary: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    unique, inverse, counts = np.unique(wavelength, return_inverse=True, return_counts=True)
    if unique.size == wavelength.size:
        return wavelength, real, imaginary, 0
    merged_real = np.zeros(unique.size)
    merged_imaginary = np.zeros(unique.size)
    np.add.at(merged_real, inverse, real)
    np.add.at(merged_imaginary, inverse, imaginary)
    merged_real /= counts
    merged_imaginary /= counts
    return unique, merged_real, merged_imaginary, int(wavelength.size - unique.size)


__all__ = [
    "MieParticleOptics",
    "OpticalConstantsCatalog",
    "RefractiveIndexSpectrum",
    "load_refractive_index_csv",
    "load_exo_skryer_refractive_index",
    "load_refractive_index_table",
    "lognormal_mie_optics",
    "mie_cloud_from_mass_fraction",
    "mie_efficiencies",
    "mie_phase_function_moments",
    "refractive_index_from_parameters",
]
