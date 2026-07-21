"""Prepared stellar surface spectra.

PHOENIX spectra are loaded through STScI's maintained ``stsynphot`` package.
The package expects ``PYSYN_CDBS`` to point at a Synphot reference-data root
containing ``grid/phoenix/catalog.fits``. File access and spectral preparation
happen when a forward model is constructed, never inside likelihood calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Protocol, runtime_checkable
import warnings

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.bodies import Star
from robert_exoplanets.core import RobertConfigError, RobertValidationError, SpectralGrid, Spectrum

PLANCK_CONSTANT_J_S = 6.62607015e-34
LIGHT_SPEED_M_S = 299792458.0
BOLTZMANN_CONSTANT_J_K = 1.380649e-23
STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8
STELLAR_RADIANCE_UNIT = "W m^-3 sr^-1"


@runtime_checkable
class StellarSpectrumModel(Protocol):
    """Protocol for models prepared on a forward model's spectral grid."""

    @property
    def name(self) -> str:
        """Stable model identifier."""

    def prepare(self, star: Star, spectral_grid: SpectralGrid) -> Spectrum:
        """Return stellar surface radiance on ``spectral_grid``."""


@dataclass(frozen=True)
class BlackbodyStellarSpectrumModel:
    """Controlled Planck approximation, not a stellar-validation standard."""

    name: str = "blackbody"

    def prepare(self, star: Star, spectral_grid: SpectralGrid) -> Spectrum:
        temperature = _required_star_parameter(
            star.effective_temperature_k,
            "effective_temperature_k",
            model=self.name,
        )
        wavelength_micron = _wavelength_micron(spectral_grid)
        values = _planck_radiance_wavelength(wavelength_micron, temperature)
        return Spectrum(
            spectral_grid=_output_grid(spectral_grid, wavelength_micron),
            values=values,
            unit=STELLAR_RADIANCE_UNIT,
            observable="stellar_spectral_radiance",
            metadata={
                "stellar_model": self.name,
                "stellar_model_scope": "controlled_approximation_not_validation_standard",
                "effective_temperature_k": f"{temperature:.17g}",
                "surface_flux_convention": "pi_times_radiance",
            },
        )


@dataclass(frozen=True)
class PhoenixStellarSpectrumModel:
    """STScI PHOENIX atmosphere spectrum interpolated in stellar parameters.

    The STScI atlas is interpolated in effective temperature, metallicity, and
    log10 surface gravity. Its FLAM surface flux is optionally normalized so
    its wavelength integral equals ``sigma * T_eff**4``, then divided by pi to
    produce the disk-averaged surface radiance used for eclipse-depth ratios.
    """

    bolometric_normalization: bool = True
    name: str = "phoenix"

    def prepare(self, star: Star, spectral_grid: SpectralGrid) -> Spectrum:
        temperature = _required_star_parameter(
            star.effective_temperature_k,
            "effective_temperature_k",
            model=self.name,
        )
        metallicity = _required_star_parameter(
            star.metallicity_dex,
            "metallicity_dex",
            model=self.name,
        )
        log_g = _required_star_parameter(
            star.log_g_cgs,
            "log_g_cgs",
            model=self.name,
        )
        source = _load_phoenix_source_spectrum(
            temperature,
            metallicity,
            log_g,
            os.environ.get("PYSYN_CDBS"),
        )

        try:
            from astropy import units as u
            from synphot import units as synphot_units
        except ImportError as exc:  # pragma: no cover - guarded by loader too
            raise RobertConfigError(
                "PHOENIX stellar spectra require the synphot and stsynphot packages"
            ) from exc

        waveset = source.waveset
        if waveset is None:
            raise RobertValidationError("PHOENIX spectrum has no native wavelength grid")
        native_wavelength_angstrom = np.asarray(
            waveset.to_value(u.AA), dtype=float
        )
        native_flux_flam = np.asarray(
            source(waveset, flux_unit=synphot_units.FLAM).value,
            dtype=float,
        )
        if (
            native_wavelength_angstrom.ndim != 1
            or native_flux_flam.shape != native_wavelength_angstrom.shape
            or not np.all(np.isfinite(native_wavelength_angstrom))
            or not np.all(np.isfinite(native_flux_flam))
            or np.any(native_flux_flam < 0.0)
        ):
            raise RobertValidationError("PHOENIX source spectrum contains invalid data")

        # FLAM is erg s^-1 cm^-2 Angstrom^-1. Integrating over Angstrom and
        # multiplying by 1e-3 converts the bolometric surface flux to W m^-2.
        raw_bolometric_flux = float(
            _trapezoid(native_flux_flam, native_wavelength_angstrom) * 1.0e-3
        )
        if not np.isfinite(raw_bolometric_flux) or raw_bolometric_flux <= 0.0:
            raise RobertValidationError(
                "PHOENIX spectrum has non-positive integrated surface flux"
            )
        expected_bolometric_flux = STEFAN_BOLTZMANN_W_M2_K4 * temperature**4
        normalization = (
            expected_bolometric_flux / raw_bolometric_flux
            if self.bolometric_normalization
            else 1.0
        )

        wavelength_micron = _wavelength_micron(spectral_grid)
        target_angstrom = wavelength_micron * 1.0e4
        edge_angstrom = (
            None
            if spectral_grid.bin_edges is None
            else _spectral_values_to_micron(
                np.asarray(spectral_grid.bin_edges, dtype=float), spectral_grid.unit
            )
            * 1.0e4
        )
        rebinned_flam = _resample_surface_flux(
            source,
            native_wavelength_angstrom,
            target_angstrom,
            edge_angstrom,
            flux_unit=synphot_units.FLAM,
            wavelength_unit=u.AA,
        )
        # 1 FLAM = 1e7 W m^-3. PHOENIX tabulates hemispherically integrated
        # surface flux, so F_lambda / pi is the radiance matching ROBERT's
        # disk-integrated planetary radiance convention.
        radiance = rebinned_flam * normalization * 1.0e7 / np.pi
        if not np.all(np.isfinite(radiance)) or np.any(radiance <= 0.0):
            raise RobertValidationError(
                "PHOENIX interpolation produced non-positive stellar radiance"
            )

        return Spectrum(
            spectral_grid=_output_grid(spectral_grid, wavelength_micron),
            values=radiance,
            unit=STELLAR_RADIANCE_UNIT,
            observable="stellar_spectral_radiance",
            metadata={
                "stellar_model": self.name,
                "stellar_library": "STScI PHOENIX (Allard et al. 2009)",
                "stellar_loader": "stsynphot.grid_to_spec",
                "effective_temperature_k": f"{temperature:.17g}",
                "metallicity_dex": f"{metallicity:.17g}",
                "log_g_cgs": f"{log_g:.17g}",
                "bolometric_normalization": str(self.bolometric_normalization).lower(),
                "raw_bolometric_surface_flux_w_m2": f"{raw_bolometric_flux:.17g}",
                "expected_bolometric_surface_flux_w_m2": (
                    f"{expected_bolometric_flux:.17g}"
                ),
                "bolometric_normalization_factor": f"{normalization:.17g}",
                "surface_flux_convention": "phoenix_f_lambda_divided_by_pi",
                "spectral_sampling": (
                    "flux_conserving_bin_average"
                    if edge_angstrom is not None
                    else "point_evaluation"
                ),
                "pysyn_cdbs": os.environ.get("PYSYN_CDBS", ""),
            },
        )


def prepare_stellar_spectrum(
    star: Star,
    spectral_grid: SpectralGrid,
    *,
    model: str = "phoenix",
) -> Spectrum:
    """Prepare the selected stellar model on a forward model grid.

    Parameters
    ----------
    star
        Stellar parameters used to select or construct the spectrum.
    spectral_grid
        Target wavelength or wavenumber grid. Bin edges trigger
        flux-conserving bin averages for PHOENIX spectra.
    model
        ``"phoenix"`` (default) or ``"blackbody"``.
    """

    normalized = str(model).strip().lower()
    if normalized == "phoenix":
        provider: StellarSpectrumModel = PhoenixStellarSpectrumModel()
    elif normalized == "blackbody":
        provider = BlackbodyStellarSpectrumModel()
    else:
        raise RobertConfigError(
            "stellar spectrum model must be 'phoenix' or 'blackbody'"
        )
    return provider.prepare(star, spectral_grid)


@lru_cache(maxsize=16)
def _load_phoenix_source_spectrum(
    effective_temperature_k: float,
    metallicity_dex: float,
    log_g_cgs: float,
    reference_data_root: str | None = None,
):
    root = reference_data_root or os.environ.get("PYSYN_CDBS")
    if not root:
        raise RobertConfigError(
            "PHOENIX stellar spectra require PYSYN_CDBS to point at an STScI "
            "Synphot reference-data root containing grid/phoenix"
        )
    catalog = Path(root).expanduser() / "grid" / "phoenix" / "catalog.fits"
    if not catalog.is_file():
        raise RobertConfigError(
            f"PHOENIX catalog was not found at {catalog}; PYSYN_CDBS must point "
            "at the directory above grid/"
        )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Failed to load Vega spectrum.*"
            )
            import stsynphot
    except ImportError as exc:
        raise RobertConfigError(
            "PHOENIX stellar spectra require the synphot and stsynphot packages"
        ) from exc
    try:
        return stsynphot.grid_to_spec(
            "phoenix",
            effective_temperature_k,
            metallicity_dex,
            log_g_cgs,
        )
    except Exception as exc:
        raise RobertValidationError(
            "unable to interpolate the STScI PHOENIX grid at "
            f"T_eff={effective_temperature_k:g} K, [M/H]={metallicity_dex:g}, "
            f"log(g)={log_g_cgs:g}: {exc}"
        ) from exc


def _resample_surface_flux(
    source,
    native_wavelength_angstrom: NDArray[np.float64],
    target_wavelength_angstrom: NDArray[np.float64],
    bin_edges_angstrom: NDArray[np.float64] | None,
    *,
    flux_unit,
    wavelength_unit,
) -> NDArray[np.float64]:
    coverage_min = float(np.min(native_wavelength_angstrom))
    coverage_max = float(np.max(native_wavelength_angstrom))
    requested = (
        target_wavelength_angstrom
        if bin_edges_angstrom is None
        else bin_edges_angstrom
    )
    requested_min = float(np.min(requested))
    requested_max = float(np.max(requested))
    if requested_min < coverage_min or requested_max > coverage_max:
        raise RobertValidationError(
            "stellar spectral grid lies outside PHOENIX coverage: requested "
            f"{requested_min / 1e4:.8g}-{requested_max / 1e4:.8g} micron, "
            f"available {coverage_min / 1e4:.8g}-{coverage_max / 1e4:.8g} micron"
        )

    if bin_edges_angstrom is None:
        return np.asarray(
            source(
                target_wavelength_angstrom * wavelength_unit,
                flux_unit=flux_unit,
            ).value,
            dtype=float,
        )

    values = np.empty(target_wavelength_angstrom.size, dtype=float)
    for index, (left, right) in enumerate(
        zip(bin_edges_angstrom[:-1], bin_edges_angstrom[1:], strict=True)
    ):
        lower = min(float(left), float(right))
        upper = max(float(left), float(right))
        inside = native_wavelength_angstrom[
            (native_wavelength_angstrom > lower)
            & (native_wavelength_angstrom < upper)
        ]
        sample_wavelength = np.concatenate(([lower], inside, [upper]))
        sample_flux = np.asarray(
            source(sample_wavelength * wavelength_unit, flux_unit=flux_unit).value,
            dtype=float,
        )
        values[index] = _trapezoid(sample_flux, sample_wavelength) / (
            upper - lower
        )
    return values


def _trapezoid(
    values: NDArray[np.float64], coordinates: NDArray[np.float64]
) -> np.float64:
    """Integrate with NumPy 1.x and 2.x without changing the package minimum."""

    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, coordinates)
    return np.trapz(values, coordinates)  # type: ignore[attr-defined]


def _required_star_parameter(
    value: float | None,
    name: str,
    *,
    model: str,
) -> float:
    if value is None:
        raise RobertValidationError(f"{model} stellar spectrum requires star.{name}")
    result = float(value)
    if not np.isfinite(result):
        raise RobertValidationError(f"star.{name} must be finite")
    return result


def _wavelength_micron(spectral_grid: SpectralGrid) -> NDArray[np.float64]:
    return _spectral_values_to_micron(
        np.asarray(spectral_grid.values, dtype=float), spectral_grid.unit
    )


def _spectral_values_to_micron(
    values: NDArray[np.float64], unit: str
) -> NDArray[np.float64]:
    normalized = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    if normalized in {"micron", "microns", "um"}:
        result = values
    elif normalized in {"nm", "nanometer", "nanometers"}:
        result = values * 1.0e-3
    elif normalized in {"angstrom", "angstroms", "aa"}:
        result = values * 1.0e-4
    elif normalized in {"m", "meter", "meters"}:
        result = values * 1.0e6
    elif normalized in {"cm^-1", "cm-1", "1/cm"}:
        result = 1.0e4 / values
    else:
        raise RobertValidationError(
            f"unsupported stellar spectral-grid unit {unit!r}"
        )
    result = np.asarray(result, dtype=float)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise RobertValidationError("stellar wavelength grid must be finite and positive")
    return result


def _output_grid(
    source: SpectralGrid,
    wavelength_micron: NDArray[np.float64],
) -> SpectralGrid:
    bin_edges = (
        None
        if source.bin_edges is None
        else _spectral_values_to_micron(
            np.asarray(source.bin_edges, dtype=float), source.unit
        )
    )
    return SpectralGrid(
        values=wavelength_micron,
        bin_edges=bin_edges,
        unit="micron",
        name=source.name,
        role="stellar_model",
        metadata={"source_spectral_unit": source.unit},
    )


def _planck_radiance_wavelength(
    wavelength_micron: NDArray[np.float64], temperature_k: float
) -> NDArray[np.float64]:
    wavelength_m = np.asarray(wavelength_micron, dtype=float) * 1.0e-6
    exponent = (
        PLANCK_CONSTANT_J_S
        * LIGHT_SPEED_M_S
        / (wavelength_m * BOLTZMANN_CONSTANT_J_K * temperature_k)
    )
    denominator = np.expm1(np.minimum(exponent, 700.0))
    return (
        2.0
        * PLANCK_CONSTANT_J_S
        * LIGHT_SPEED_M_S**2
        / (wavelength_m**5 * denominator)
    )
