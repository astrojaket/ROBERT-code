"""Optional adapter from Virga eddy-sed clouds to ROBERT cloud optics.

This adapter targets the documented ``virga-exo`` 2.x interface. It runs
Virga's Ackerman--Marley eddy-sed calculation from ROBERT's pressure-edge
temperature profile and returns Virga's layer optical depth, single-scattering
albedo, and asymmetry factor. It does not estimate a condensate profile when
Virga is unavailable.

Virga requires precomputed ``<condensible>.mieff`` files. Prepare these before
retrieval. ``gas_mass_mixing_ratios`` gives the bottom gas mass mixing ratio
(g/g) for each condensible; the adapter does not infer these values from
ROBERT's volume-mixing-ratio chemistry or use Virga's built-in abundance
defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module, metadata as importlib_metadata
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.core import RobertConfigError, RobertDataError, RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.opacity import pressure_values_in_unit, spectral_grid_values_in_unit
from robert_exoplanets.rt import CloudOpticalProperties, GasOpticalDepth


_SUPPORTED_MULTIPLE_SCATTERING_BACKENDS = {
    "two_stream",
    "toon_hemispheric_mean",
    "sh4",
    "p3",
}


@dataclass(frozen=True)
class VirgaEddySedCloudModel:
    """Use Virga to calculate one or more spherical eddy-sed cloud species.

    Parameters use Virga's conventions:

    - ``gas_mass_mixing_ratios`` are bottom gas mass mixing ratios in g/g.
    - Set exactly one of ``fsed`` or ``fsed_parameter``. A retrieved fsed value
      is the positive, dimensionless value used by Virga.
    - Set exactly one of ``kz_cm2_s`` or ``log10_kz_cm2_s_parameter``. The
      retrieved value is log10(Kzz / (cm^2/s)).
    - ``metallicity`` is a linear factor relative to solar, not log metallicity.
    - ``geometric_stddev`` is the width of Virga's lognormal size distribution.
    - ``mie_directory`` contains precomputed ``<condensible>.mieff`` tables.

    A retrieved log10 Kzz value must lie in [-300, 300]. This bound prevents
    floating-point underflow or overflow during conversion to Kzz. A retrieved
    fsed value must be finite and positive. The adapter supports constant
    controls, a lognormal particle distribution, homogeneous spherical
    particles, and constant atmospheric mean molecular weight and gravity. It
    passes ROBERT pressure edges and temperature edges to Virga, so each Virga
    layer maps to one ROBERT pressure layer. It fails on non-matching grids
    instead of silently rebinning pressure layers.
    """

    condensibles: tuple[str, ...]
    gas_mass_mixing_ratios: Mapping[str, float]
    mie_directory: Path | str
    fsed: float | None = None
    kz_cm2_s: float | None = None
    metallicity: float = 1.0
    geometric_stddev: float = 2.0
    supersaturation: float = 0.0
    multiple_scattering_backend: str = "sh4"
    metadata: Mapping[str, str] = field(default_factory=dict)
    fsed_parameter: str | None = None
    log10_kz_cm2_s_parameter: str | None = None

    def __post_init__(self) -> None:
        condensibles = tuple(str(item).strip() for item in self.condensibles)
        if not condensibles or any(not item for item in condensibles):
            raise RobertValidationError("Virga condensibles must be non-empty names")
        if len(set(condensibles)) != len(condensibles):
            raise RobertValidationError("Virga condensibles must be unique")

        raw_gas_mmr = dict(self.gas_mass_mixing_ratios)
        if set(raw_gas_mmr) != set(condensibles):
            raise RobertValidationError(
                "gas_mass_mixing_ratios must provide exactly one value for each "
                "Virga condensible"
            )
        gas_mmr = {
            name: _finite_positive(raw_gas_mmr[name], f"gas mass mixing ratio for {name}")
            for name in condensibles
        }
        if sum(gas_mmr.values()) > 1.0:
            raise RobertValidationError(
                "Virga condensible gas mass mixing ratios must sum to no more than one"
            )

        fsed = None if self.fsed is None else _finite_positive(self.fsed, "fsed")
        fsed_parameter = _optional_parameter_name(self.fsed_parameter, "fsed_parameter")
        _validate_control_source(fsed, fsed_parameter, "fsed")

        kz = (
            None
            if self.kz_cm2_s is None
            else _finite_positive(self.kz_cm2_s, "kz_cm2_s")
        )
        log10_kz_parameter = _optional_parameter_name(
            self.log10_kz_cm2_s_parameter, "log10_kz_cm2_s_parameter"
        )
        _validate_control_source(kz, log10_kz_parameter, "Kzz")
        if fsed_parameter is not None and fsed_parameter == log10_kz_parameter:
            raise RobertValidationError(
                "Virga fsed and log10 Kzz parameter names must be different"
            )
        metallicity = _finite_positive(self.metallicity, "metallicity")
        width = float(self.geometric_stddev)
        if not np.isfinite(width) or width <= 1.0:
            raise RobertValidationError(
                "geometric_stddev must be finite and greater than one for Virga's "
                "lognormal distribution"
            )
        supersaturation = float(self.supersaturation)
        if not np.isfinite(supersaturation) or supersaturation < 0.0:
            raise RobertValidationError(
                "supersaturation must be finite and non-negative"
            )

        backend = str(self.multiple_scattering_backend).strip().lower()
        if backend not in _SUPPORTED_MULTIPLE_SCATTERING_BACKENDS:
            raise RobertValidationError(
                "multiple_scattering_backend must be one of "
                + ", ".join(sorted(_SUPPORTED_MULTIPLE_SCATTERING_BACKENDS))
            )

        mie_directory = Path(self.mie_directory).expanduser().resolve()
        if not mie_directory.is_dir():
            raise RobertConfigError(
                f"Virga Mie directory does not exist or is not a directory: "
                f"{mie_directory}"
            )
        for condensible in condensibles:
            mie_path = mie_directory / f"{condensible}.mieff"
            if not mie_path.is_file() or mie_path.stat().st_size == 0:
                raise RobertConfigError(
                    f"Virga requires a non-empty precomputed Mie table: {mie_path}"
                )

        object.__setattr__(self, "condensibles", condensibles)
        object.__setattr__(self, "gas_mass_mixing_ratios", immutable_mapping(gas_mmr))
        object.__setattr__(self, "mie_directory", mie_directory)
        object.__setattr__(self, "fsed", fsed)
        object.__setattr__(self, "kz_cm2_s", kz)
        object.__setattr__(self, "fsed_parameter", fsed_parameter)
        object.__setattr__(self, "log10_kz_cm2_s_parameter", log10_kz_parameter)
        object.__setattr__(self, "metallicity", metallicity)
        object.__setattr__(self, "geometric_stddev", width)
        object.__setattr__(self, "supersaturation", supersaturation)
        object.__setattr__(self, "multiple_scattering_backend", backend)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def required_parameters(self) -> tuple[str, ...]:
        """Return the retrieved Virga controls used by this provider."""

        return tuple(
            name
            for name in (self.fsed_parameter, self.log10_kz_cm2_s_parameter)
            if name is not None
        )

    @property
    def manifest_metadata(self) -> Mapping[str, str]:
        """Return the Virga assumptions and control sources for the manifest."""

        return immutable_mapping(
            {
                **dict(self.metadata),
                "cloud_model": "virga_ackerman_marley_eddy_sed",
                "cloud_geometry_independent": "true",
                "cloud_provider": "virga-exo",
                "cloud_microphysics_reference": "Ackerman and Marley (2001)",
                "cloud_condensibles": ",".join(self.condensibles),
                "cloud_gas_mass_mixing_ratios_g_g": ",".join(
                    f"{name}:{self.gas_mass_mixing_ratios[name]:.17g}"
                    for name in self.condensibles
                ),
                "cloud_fsed": _control_source_metadata(self.fsed, self.fsed_parameter),
                "cloud_kz_cm2_s": _control_source_metadata(
                    self.kz_cm2_s, self.log10_kz_cm2_s_parameter
                ),
                "cloud_fsed_parameter": self.fsed_parameter or "fixed",
                "cloud_fsed_units": "dimensionless",
                "cloud_log10_kz_cm2_s_parameter": (
                    self.log10_kz_cm2_s_parameter or "fixed"
                ),
                "cloud_kz_units": "cm^2/s",
                "cloud_metallicity_linear_solar": f"{self.metallicity:.17g}",
                "cloud_geometric_stddev": f"{self.geometric_stddev:.17g}",
                "cloud_supersaturation": f"{self.supersaturation:.17g}",
                "cloud_particle_distribution": "lognormal_homogeneous_spheres",
                "cloud_phase_function_closure": (
                    "henyey_greenstein_from_virga_asymmetry"
                ),
                "cloud_virga_grid_mapping": (
                    "input_pressure_edges_and_temperature_edges; "
                    "one_virga_layer_per_robert_layer"
                ),
                "cloud_virga_do_virtual": "false",
                "cloud_multiple_scattering_backend": self.multiple_scattering_backend,
                "cloud_mie_directory": str(self.mie_directory),
            }
        )

    def evaluate(
        self,
        gas_optical_depth: GasOpticalDepth,
        parameters: Mapping[str, float],
    ) -> tuple[CloudOpticalProperties, ...]:
        """Run Virga and return its layer optical properties on ROBERT's grid."""

        fsed, kz_cm2_s = self._resolved_controls(parameters)
        atmosphere = gas_optical_depth.atmosphere
        pressure_grid = atmosphere.pressure_grid
        if atmosphere.temperature_edges is None:
            raise RobertValidationError(
                "Virga requires atmosphere.temperature_edges on the pressure-edge grid"
            )
        if atmosphere.mean_molecular_weight_unit.strip().lower() not in {
            "amu",
            "u",
            "da",
            "dalton",
        }:
            raise RobertValidationError(
                "Virga requires mean_molecular_weight in amu; received "
                f"{atmosphere.mean_molecular_weight_unit!r}"
            )

        gravity_m_s2 = _uniform_profile(
            gas_optical_depth.gravity_m_s2,
            "Virga requires constant gravity across the atmosphere",
        )
        mean_molecular_weight = _uniform_profile(
            atmosphere.mean_molecular_weight,
            "Virga requires constant mean molecular weight across the atmosphere",
        )

        pressure_bar = pressure_values_in_unit(
            pressure_grid.edges,
            pressure_grid.unit,
            "bar",
        )
        temperature_edges = np.asarray(atmosphere.temperature_edges, dtype=float)
        if temperature_edges.shape != (pressure_grid.n_layers + 1,):
            raise RobertValidationError(
                "atmosphere.temperature_edges must match pressure-grid edges"
            )
        if not np.all(np.isfinite(temperature_edges)) or np.any(temperature_edges <= 0.0):
            raise RobertValidationError(
                "atmosphere.temperature_edges must be finite and positive"
            )

        # Virga sorts ptk rows into increasing pressure. Supply edge-level state
        # in that order, then map returned rows back to ROBERT's edge order.
        if pressure_grid.orientation == "decreasing":
            pressure_bar = pressure_bar[::-1]
            temperature_edges = temperature_edges[::-1]
        ptk = {
            "pressure": np.asarray(pressure_bar, dtype=float),
            "temperature": temperature_edges,
            "kz": np.full(pressure_bar.size, kz_cm2_s, dtype=float),
        }

        virga, virga_version = _load_virga()
        try:
            import astropy.units as units

            virga_atmosphere = virga.Atmosphere(
                list(self.condensibles),
                fsed=fsed,
                mh=self.metallicity,
                mmw=mean_molecular_weight,
                sig=self.geometric_stddev,
                param="const",
                verbose=False,
                supsat=self.supersaturation,
                gas_mmr=dict(self.gas_mass_mixing_ratios),
            )
            virga_atmosphere.gravity(
                gravity=gravity_m_s2 * 100.0,
                gravity_unit=units.cm / units.s**2,
            )
            # Set kz_min to the configured value. This prevents Virga's default
            # floor from silently replacing the user's constant profile.
            virga_atmosphere.ptk(
                df=ptk,
                kz_min=kz_cm2_s,
            )
            result = virga.compute(
                virga_atmosphere,
                directory=str(self.mie_directory),
                as_dict=True,
                do_virtual=False,
            )
        except Exception as exc:
            raise RobertDataError(
                "Virga cloud evaluation failed for condensibles "
                f"{', '.join(self.condensibles)}: {exc}"
            ) from exc

        cloud = _cloud_optics_from_virga(
            result,
            gas_optical_depth,
            condensibles=self.condensibles,
            virga_version=virga_version,
            model_metadata={
                **self.manifest_metadata,
                "cloud_fsed_used": f"{fsed:.17g}",
                "cloud_kz_cm2_s_used": f"{kz_cm2_s:.17g}",
            },
        )
        return (cloud,)

    def _resolved_controls(
        self,
        parameters: Mapping[str, float],
    ) -> tuple[float, float]:
        if self.fsed_parameter is None:
            assert self.fsed is not None
            fsed = self.fsed
        else:
            if self.fsed_parameter not in parameters:
                raise RobertValidationError(
                    "Virga fsed parameter is missing: " + self.fsed_parameter
                )
            fsed = _finite_positive(
                parameters[self.fsed_parameter],
                f"Virga fsed parameter {self.fsed_parameter}",
            )

        if self.log10_kz_cm2_s_parameter is None:
            assert self.kz_cm2_s is not None
            kz_cm2_s = self.kz_cm2_s
        else:
            name = self.log10_kz_cm2_s_parameter
            if name not in parameters:
                raise RobertValidationError(
                    "Virga log10 Kzz parameter is missing: " + name
                )
            try:
                log10_kz = float(parameters[name])
            except (TypeError, ValueError, OverflowError) as exc:
                raise RobertValidationError(
                    f"Virga log10 Kzz parameter {name} must be numeric"
                ) from exc
            if not np.isfinite(log10_kz) or not -300.0 <= log10_kz <= 300.0:
                raise RobertValidationError(
                    f"Virga log10 Kzz parameter {name} must be finite and lie in [-300, 300]"
                )
            kz_cm2_s = float(10.0**log10_kz)
            if not np.isfinite(kz_cm2_s) or kz_cm2_s <= 0.0:
                raise RobertValidationError(
                    f"Virga log10 Kzz parameter {name} does not produce a finite positive Kzz"
                )
        return fsed, kz_cm2_s


def _cloud_optics_from_virga(
    result: object,
    gas_optical_depth: GasOpticalDepth,
    *,
    condensibles: tuple[str, ...],
    virga_version: str,
    model_metadata: Mapping[str, str],
) -> CloudOpticalProperties:
    """Validate and map a documented Virga ``as_dict`` result."""

    if not isinstance(result, Mapping):
        raise RobertDataError("Virga compute(as_dict=True) must return a mapping")
    required = (
        "pressure",
        "pressure_unit",
        "wave",
        "wave_unit",
        "opd_per_layer",
        "single_scattering",
        "asymmetry",
        "condensate_mmr",
        "condensibles",
    )
    missing = tuple(key for key in required if key not in result)
    if missing:
        raise RobertDataError(
            "Virga result is missing required documented fields: "
            + ", ".join(missing)
        )

    returned_condensibles = tuple(str(item) for item in result["condensibles"])
    if returned_condensibles != condensibles:
        raise RobertDataError(
            "Virga result condensible order does not match the requested species"
        )

    pressure_unit = str(result["pressure_unit"])
    virga_pressure = pressure_values_in_unit(
        result["pressure"], pressure_unit, "bar"
    )
    pressure_grid = gas_optical_depth.pressure_grid
    edges_bar = pressure_values_in_unit(
        pressure_grid.edges,
        pressure_grid.unit,
        "bar",
    )
    ascending_edges = np.sort(edges_bar)
    expected_pressure = 0.5 * (ascending_edges[:-1] + ascending_edges[1:])
    if virga_pressure.shape != expected_pressure.shape or not np.allclose(
        virga_pressure,
        expected_pressure,
        rtol=1.0e-10,
        atol=1.0e-14,
    ):
        raise RobertDataError(
            "Virga pressure layers do not match ROBERT's pressure-edge intervals; "
            "the adapter requires one Virga layer per ROBERT layer"
        )

    n_layers = pressure_grid.n_layers
    wave_micron = _virga_wavelength_micron(result["wave"], result["wave_unit"])
    tau = _virga_layer_spectrum(
        result["opd_per_layer"], n_layers, wave_micron.size, "opd_per_layer"
    )
    albedo = _virga_layer_spectrum(
        result["single_scattering"], n_layers, wave_micron.size, "single_scattering"
    )
    asymmetry = _virga_layer_spectrum(
        result["asymmetry"], n_layers, wave_micron.size, "asymmetry"
    )
    condensate_mmr = np.asarray(result["condensate_mmr"], dtype=float)
    if condensate_mmr.shape != (n_layers, len(condensibles)):
        raise RobertDataError(
            "Virga condensate_mmr must have shape "
            f"({n_layers}, {len(condensibles)}); received {condensate_mmr.shape}"
        )
    if not np.all(np.isfinite(condensate_mmr)) or np.any(condensate_mmr < 0.0):
        raise RobertDataError(
            "Virga condensate_mmr must contain finite non-negative values"
        )
    if np.any(tau < 0.0):
        raise RobertDataError("Virga opd_per_layer must be non-negative")
    if np.any((albedo < 0.0) | (albedo > 1.0)):
        raise RobertDataError("Virga single_scattering must lie in [0, 1]")
    if np.any((asymmetry < -1.0) | (asymmetry > 1.0)):
        raise RobertDataError("Virga asymmetry must lie in [-1, 1]")

    spectral_grid = gas_optical_depth.spectral_grid
    target_wave_micron = spectral_grid_values_in_unit(spectral_grid, "micron")
    tau = _interpolate_spectrum(tau, wave_micron, target_wave_micron, "opd_per_layer")
    albedo = _interpolate_spectrum(
        albedo, wave_micron, target_wave_micron, "single_scattering"
    )
    asymmetry = _interpolate_spectrum(
        asymmetry, wave_micron, target_wave_micron, "asymmetry"
    )

    if pressure_grid.orientation == "decreasing":
        tau = tau[::-1]
        albedo = albedo[::-1]
        asymmetry = asymmetry[::-1]
        condensate_mmr = condensate_mmr[::-1]

    metadata = {
        **dict(model_metadata),
        "virga_version": virga_version,
        "virga_output_mapping": "opd_per_layer,single_scattering,asymmetry",
        "virga_spectral_interpolation": "linear_in_log_wavelength_no_extrapolation",
        "virga_pressure_layer_mapping": "matched_pressure_edges",
        "virga_condensate_mmr_min": f"{float(np.min(condensate_mmr)):.17g}",
        "virga_condensate_mmr_max": f"{float(np.max(condensate_mmr)):.17g}",
    }
    return CloudOpticalProperties(
        name="Virga " + "+".join(condensibles),
        extinction_tau=tau,
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        single_scattering_albedo=albedo,
        asymmetry_factor=asymmetry,
        metadata=metadata,
    )


def _load_virga():
    """Load the optional documented Virga module and its package version."""

    try:
        virga = import_module("virga.justdoit")
    except ModuleNotFoundError as exc:
        raise RobertConfigError(
            "Virga cloud model requires the optional 'virga-exo' package and its "
            "dependencies. Install Virga in the active Python environment; the "
            "current Virga 2.x release requires Python 3.11 or newer."
        ) from exc
    if not all(hasattr(virga, name) for name in ("Atmosphere", "compute")):
        raise RobertConfigError(
            "Virga cloud model requires the documented virga-exo 2.x API: "
            "virga.justdoit.Atmosphere and virga.justdoit.compute"
        )
    try:
        version = importlib_metadata.version("virga-exo")
    except importlib_metadata.PackageNotFoundError:
        try:
            version = importlib_metadata.version("virga")
        except importlib_metadata.PackageNotFoundError:
            version = "unknown"
    major = version.lstrip("v").split(".", maxsplit=1)[0]
    if major.isdigit() and int(major) < 2:
        raise RobertConfigError(
            f"Virga version {version} is not supported; install virga-exo 2.x or newer"
        )
    return virga, version


def _virga_wavelength_micron(values: object, unit: object) -> NDArray[np.float64]:
    wave = np.asarray(values, dtype=float)
    if wave.ndim != 1 or wave.size < 2 or not np.all(np.isfinite(wave)):
        raise RobertDataError("Virga wave must be a finite one-dimensional array")
    if np.any(wave <= 0.0):
        raise RobertDataError("Virga wave values must be positive")
    normalized_unit = str(unit).strip().lower().replace("μ", "u")
    if normalized_unit in {"micron", "microns", "um", "wavelength_micron"}:
        wave_micron = wave
    elif normalized_unit in {"cm^-1", "cm-1", "1/cm", "wavenumber"}:
        wave_micron = 10000.0 / wave
    else:
        raise RobertDataError(f"unsupported Virga wave unit: {unit!r}")
    if np.unique(wave_micron).size != wave_micron.size:
        raise RobertDataError("Virga wave grid must not contain duplicate values")
    return np.asarray(wave_micron, dtype=float)


def _virga_layer_spectrum(
    values: object,
    n_layers: int,
    n_wavelengths: int,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.shape != (n_layers, n_wavelengths):
        raise RobertDataError(
            f"Virga {name} must have shape ({n_layers}, {n_wavelengths}); "
            f"received {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise RobertDataError(f"Virga {name} must contain only finite values")
    return array


def _interpolate_spectrum(
    values: NDArray[np.float64],
    source_wave_micron: NDArray[np.float64],
    target_wave_micron: NDArray[np.float64],
    name: str,
) -> NDArray[np.float64]:
    source_order = np.argsort(source_wave_micron)
    source_wave = source_wave_micron[source_order]
    if target_wave_micron.size == 0:
        raise RobertValidationError("ROBERT spectral grid must not be empty")
    lower = float(np.min(target_wave_micron))
    upper = float(np.max(target_wave_micron))
    tolerance = 1.0e-12
    if lower < source_wave[0] * (1.0 - tolerance) or upper > source_wave[-1] * (
        1.0 + tolerance
    ):
        raise RobertDataError(
            f"ROBERT wavelength coverage [{lower:g}, {upper:g}] micron is outside "
            f"Virga {name} coverage [{source_wave[0]:g}, {source_wave[-1]:g}] micron"
        )
    source_log_wave = np.log(source_wave)
    target_log_wave = np.log(target_wave_micron)
    sorted_values = values[:, source_order]
    result = np.vstack(
        [np.interp(target_log_wave, source_log_wave, row) for row in sorted_values]
    )
    if not np.all(np.isfinite(result)):
        raise RobertDataError(f"interpolated Virga {name} contains non-finite values")
    return result


def _uniform_profile(values: object, message: str) -> float:
    profile = np.asarray(values, dtype=float)
    if profile.ndim != 1 or profile.size == 0 or not np.all(np.isfinite(profile)):
        raise RobertValidationError("Virga state profiles must be finite one-dimensional arrays")
    if not np.allclose(profile, profile[0], rtol=1.0e-10, atol=0.0):
        raise RobertValidationError(message)
    if profile[0] <= 0.0:
        raise RobertValidationError("Virga gravity and mean molecular weight must be positive")
    return float(profile[0])


def _finite_positive(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RobertValidationError(f"{name} must be finite and positive") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise RobertValidationError(f"{name} must be finite and positive")
    return number


def _optional_parameter_name(value: object | None, label: str) -> str | None:
    if value is None:
        return None
    name = str(value).strip()
    if not name:
        raise RobertValidationError(f"{label} must be a non-empty name")
    return name


def _validate_control_source(
    fixed_value: float | None,
    parameter_name: str | None,
    label: str,
) -> None:
    if (fixed_value is None) == (parameter_name is None):
        raise RobertValidationError(
            f"provide exactly one of fixed {label} or its retrieval parameter"
        )


def _control_source_metadata(
    fixed_value: float | None,
    parameter_name: str | None,
) -> str:
    if parameter_name is not None:
        return f"retrieved_parameter:{parameter_name}"
    assert fixed_value is not None
    return f"{fixed_value:.17g}"


__all__ = ["VirgaEddySedCloudModel"]
