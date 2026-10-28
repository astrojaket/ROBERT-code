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
defaults. Virga's ``gas_mmr`` argument is passed through its gas-property
functions, which convert a mole fraction to mass mixing ratio. This adapter
therefore converts each requested mass fraction to a mole fraction and checks
that Virga's gas-property function converts it back to the requested value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module, metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.core import (
    PressureGrid,
    RobertConfigError,
    RobertDataError,
    RobertValidationError,
)
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.opacity import (
    pressure_values_in_unit,
    spectral_grid_values_in_unit,
)
from robert_exoplanets.rt import CloudOpticalProperties, GasOpticalDepth

if TYPE_CHECKING:
    from robert_exoplanets.atmosphere import AtmosphereState


_SUPPORTED_MULTIPLE_SCATTERING_BACKENDS = {
    "two_stream",
    "toon_hemispheric_mean",
    "sh4",
    "p3",
}


@dataclass(frozen=True)
class VirgaCloudProfileResult:
    """Validated Virga condensate and radius profiles on one pressure grid.

    ``condensate_mass_mixing_ratio_kg_kg`` has shape ``(n_layer, n_species)``.
    It contains Virga's ``condensate_mmr`` values, documented in g/g; mass
    mixing ratios have the same numeric value in kg/kg. The radius array has
    the same shape and contains Virga's ``droplet_eff_r`` values in microns.
    Both arrays follow the orientation and layer order in ``pressure_grid``.
    """

    condensibles: tuple[str, ...]
    pressure_grid: PressureGrid
    condensate_mass_mixing_ratio_kg_kg: NDArray[np.float64]
    effective_particle_radius_micron: NDArray[np.float64]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        condensibles = tuple(str(item).strip() for item in self.condensibles)
        if not condensibles or any(not item for item in condensibles):
            raise RobertValidationError(
                "Virga profile condensibles must be non-empty names"
            )
        if len(set(condensibles)) != len(condensibles):
            raise RobertValidationError("Virga profile condensibles must be unique")
        if not isinstance(self.pressure_grid, PressureGrid):
            raise RobertValidationError(
                "Virga profile pressure_grid must be a PressureGrid"
            )
        expected_shape = (self.pressure_grid.n_layers, len(condensibles))
        condensate = np.array(
            self.condensate_mass_mixing_ratio_kg_kg,
            dtype=np.float64,
            copy=True,
        )
        radius = np.array(
            self.effective_particle_radius_micron,
            dtype=np.float64,
            copy=True,
        )
        if condensate.shape != expected_shape or radius.shape != expected_shape:
            raise RobertValidationError(
                "Virga cloud profile arrays must have shape "
                f"{expected_shape}; received {condensate.shape} and {radius.shape}"
            )
        if not np.all(np.isfinite(condensate)) or np.any(condensate < 0.0):
            raise RobertValidationError(
                "Virga condensate mass mixing ratios must be finite and non-negative"
            )
        if not np.all(np.isfinite(radius)) or np.any(radius < 0.0):
            raise RobertValidationError(
                "Virga effective particle radii must be finite and non-negative"
            )
        if np.any((condensate > 0.0) & (radius <= 0.0)):
            raise RobertValidationError(
                "Virga effective particle radius must be positive where condensate exists"
            )
        condensate.setflags(write=False)
        radius.setflags(write=False)
        object.__setattr__(self, "condensibles", condensibles)
        object.__setattr__(self, "condensate_mass_mixing_ratio_kg_kg", condensate)
        object.__setattr__(self, "effective_particle_radius_micron", radius)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def condensate_mass_mixing_ratio_unit(self) -> str:
        """Mass-ratio unit of ``condensate_mass_mixing_ratio_kg_kg``."""

        return "kg/kg"

    @property
    def effective_particle_radius_unit(self) -> str:
        """Radius unit of ``effective_particle_radius_micron``."""

        return "micron"

    @property
    def pressure_orientation(self) -> str:
        """Layer ordering used by both profiles."""

        return self.pressure_grid.orientation


@dataclass(frozen=True)
class VirgaEddySedCloudModel:
    """Use Virga to calculate one or more spherical eddy-sed cloud species.

    The user-facing condensable abundances are physical gas mass mixing ratios.
    The adapter converts these to the mole fractions required by Virga's
    ``gas_mmr`` input and validates the inverse conversion with Virga's own
    species gas-property functions.

    Parameters:

    - ``gas_mass_mixing_ratios`` are bottom gas mass mixing ratios in g/g.
    - Set exactly one of ``fsed`` or ``fsed_parameter``. A retrieved fsed value
      is the positive, dimensionless value used by Virga.
    - Set exactly one of ``kz_cm2_s`` or ``log10_kz_cm2_s_parameter``. The
      retrieved value is log10(Kzz / (cm^2/s)).
    - ``metallicity`` is a linear factor relative to solar, not log metallicity.
    - ``geometric_stddev`` is the width of Virga's lognormal size distribution.
    - ``mie_directory`` contains precomputed ``<condensible>.mieff`` tables.
    - ``mean_molecular_weight_amu`` is an optional cloud-only scalar override.
      Without it, the atmosphere MMW profile must be uniform. With it, Virga
      uses the configured scalar for its cloud calculation while ROBERT keeps
      the complete profile for the gas and radiative-transfer calculations.
      If the override differs from the atmosphere profile, this is an explicit
      approximation, not a variable-MMW Virga calculation.

    A retrieved log10 Kzz value must lie in [-300, 300]. This bound prevents
    floating-point underflow or overflow during conversion to Kzz. A retrieved
    fsed value must be finite and positive. The adapter supports constant
    controls, a lognormal particle distribution, homogeneous spherical
    particles, constant gravity, and constant atmospheric mean molecular
    weight by default. It
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
    mean_molecular_weight_amu: float | None = None

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
            name: _finite_positive(
                raw_gas_mmr[name], f"gas mass mixing ratio for {name}"
            )
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
        mean_molecular_weight_amu = (
            None
            if self.mean_molecular_weight_amu is None
            else _finite_positive(
                self.mean_molecular_weight_amu,
                "mean_molecular_weight_amu",
            )
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
        object.__setattr__(
            self,
            "mean_molecular_weight_amu",
            mean_molecular_weight_amu,
        )

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
                "cloud_virga_gas_mmr_input_semantics": (
                    "mole_fraction_number_of_condensible_gas_molecules_per_total_gas_molecule"
                ),
                "cloud_virga_gas_mass_fraction_conversion": (
                    "x_gas=q_gas*mean_molecular_weight/gas_molecular_weight"
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
                "cloud_mean_molecular_weight_input_units": "amu",
                "cloud_mean_molecular_weight_policy": (
                    "uniform_atmosphere_profile_required"
                    if self.mean_molecular_weight_amu is None
                    else "cloud_only_scalar_override"
                ),
                "cloud_mean_molecular_weight_override_amu": (
                    "not_set"
                    if self.mean_molecular_weight_amu is None
                    else f"{self.mean_molecular_weight_amu:.17g}"
                ),
                "cloud_mean_molecular_weight_scope": (
                    "virga_cloud_microphysics_only; "
                    "ROBERT_gas_and_rt_use_original_atmosphere_profile"
                ),
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
        atmosphere = gas_optical_depth.atmosphere
        result, virga_version, model_metadata = self._run_virga(
            atmosphere,
            gas_optical_depth.gravity_m_s2,
            parameters,
        )
        cloud = _cloud_optics_from_virga(
            result,
            gas_optical_depth,
            condensibles=self.condensibles,
            virga_version=virga_version,
            model_metadata=model_metadata,
        )
        return (cloud,)

    def compute_cloud_profiles(
        self,
        atmosphere: AtmosphereState,
        gravity_m_s2: object,
        parameters: Mapping[str, float],
    ) -> "VirgaCloudProfileResult":
        """Run Virga and return its physical layer profiles on ROBERT's grid.

        This runs a fresh Virga calculation for each call. It uses the same
        atmosphere, gravity, retrieved controls, and fixed cloud settings as
        :meth:`evaluate`; it does not reuse state from another posterior draw.
        The returned condensate mass fraction uses the numeric g/g values from
        Virga, which are also the mass-ratio values in kg/kg. Radius values
        come from Virga's ``droplet_eff_r`` field in microns.
        """

        result, version, model_metadata = self._run_virga(
            atmosphere, gravity_m_s2, parameters
        )
        return _cloud_profiles_from_virga(
            result,
            pressure_grid=atmosphere.pressure_grid,
            condensibles=self.condensibles,
            virga_version=version,
            model_metadata=model_metadata,
        )

    def _run_virga(
        self,
        atmosphere: AtmosphereState,
        gravity_profile: object,
        parameters: Mapping[str, float],
    ) -> tuple[object, str, Mapping[str, str]]:
        """Run Virga once after validating the shared physical state."""

        fsed, kz_cm2_s = self._resolved_controls(parameters)
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

        gravity_values = np.asarray(gravity_profile, dtype=float)
        if gravity_values.ndim == 0:
            gravity_values = np.full(pressure_grid.n_layers, float(gravity_values))
        gravity_m_s2 = _uniform_profile(
            gravity_values,
            "Virga requires constant gravity across the atmosphere",
        )
        mean_molecular_weight_profile = np.asarray(
            atmosphere.mean_molecular_weight,
            dtype=float,
        )
        if (
            mean_molecular_weight_profile.shape != (pressure_grid.n_layers,)
            or not np.all(np.isfinite(mean_molecular_weight_profile))
            or np.any(mean_molecular_weight_profile <= 0.0)
        ):
            raise RobertValidationError(
                "Virga requires a finite positive mean molecular weight profile "
                "with one value per atmosphere layer"
            )
        mean_molecular_weight_profile_min = float(np.min(mean_molecular_weight_profile))
        mean_molecular_weight_profile_max = float(np.max(mean_molecular_weight_profile))
        if self.mean_molecular_weight_amu is None:
            mean_molecular_weight = _uniform_profile(
                mean_molecular_weight_profile,
                "Virga requires constant mean molecular weight across the atmosphere; "
                "set mean_molecular_weight_amu to use a cloud-only scalar override",
            )
            mean_molecular_weight_policy = "uniform_atmosphere_profile"
            mean_molecular_weight_is_approximation = False
        else:
            mean_molecular_weight = self.mean_molecular_weight_amu
            mean_molecular_weight_policy = "cloud_only_scalar_override"
            mean_molecular_weight_is_approximation = not np.allclose(
                mean_molecular_weight_profile,
                mean_molecular_weight,
                rtol=1.0e-10,
                atol=0.0,
            )

        pressure_bar = pressure_values_in_unit(
            pressure_grid.edges, pressure_grid.unit, "bar"
        )
        temperature_edges = np.asarray(atmosphere.temperature_edges, dtype=float)
        if temperature_edges.shape != (pressure_grid.n_layers + 1,):
            raise RobertValidationError(
                "atmosphere.temperature_edges must match pressure-grid edges"
            )
        if not np.all(np.isfinite(temperature_edges)) or np.any(
            temperature_edges <= 0.0
        ):
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
        virga_gas_mole_fractions, virga_gas_molecular_weights = (
            _virga_gas_mole_fractions_from_mass_fractions(
                virga.gas_properties,
                condensibles=self.condensibles,
                gas_mass_mixing_ratios=self.gas_mass_mixing_ratios,
                mean_molecular_weight=mean_molecular_weight,
                metallicity=self.metallicity,
            )
        )
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
                gas_mmr=virga_gas_mole_fractions,
            )
            virga_atmosphere.gravity(
                gravity=gravity_m_s2 * 100.0,
                gravity_unit=units.cm / units.s**2,
            )
            # Set kz_min to the configured value. This prevents Virga's default
            # floor from silently replacing the user's constant profile.
            virga_atmosphere.ptk(df=ptk, kz_min=kz_cm2_s)
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

        metadata = {
            **self.manifest_metadata,
            "cloud_virga_gas_mole_fractions": ",".join(
                f"{name}:{virga_gas_mole_fractions[name]:.17g}"
                for name in self.condensibles
            ),
            "cloud_virga_gas_molecular_weights_amu": ",".join(
                f"{name}:{virga_gas_molecular_weights[name]:.17g}"
                for name in self.condensibles
            ),
            "cloud_virga_gas_mean_molecular_weight_amu_used": (
                f"{mean_molecular_weight:.17g}"
            ),
            "cloud_virga_gas_mass_fraction_roundtrip_validated": "true",
            "cloud_fsed_used": f"{fsed:.17g}",
            "cloud_kz_cm2_s_used": f"{kz_cm2_s:.17g}",
            "cloud_mean_molecular_weight_policy_used": mean_molecular_weight_policy,
            "cloud_mean_molecular_weight_profile_min_amu": (
                f"{mean_molecular_weight_profile_min:.17g}"
            ),
            "cloud_mean_molecular_weight_profile_max_amu": (
                f"{mean_molecular_weight_profile_max:.17g}"
            ),
            "cloud_mean_molecular_weight_amu_used": (f"{mean_molecular_weight:.17g}"),
            "cloud_mean_molecular_weight_override_amu": (
                "not_set"
                if self.mean_molecular_weight_amu is None
                else f"{self.mean_molecular_weight_amu:.17g}"
            ),
            "cloud_mean_molecular_weight_is_approximation": (
                str(mean_molecular_weight_is_approximation).lower()
            ),
            "cloud_mean_molecular_weight_approximation_scope": (
                "virga_cloud_microphysics_only; "
                "ROBERT_gas_and_rt_use_original_atmosphere_profile"
            ),
        }
        return result, virga_version, metadata

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


def _cloud_profiles_from_virga(
    result: object,
    *,
    pressure_grid: PressureGrid,
    condensibles: tuple[str, ...],
    virga_version: str,
    model_metadata: Mapping[str, str],
) -> VirgaCloudProfileResult:
    """Validate Virga's physical profile fields and align their layer order."""

    if not isinstance(result, Mapping):
        raise RobertDataError("Virga compute(as_dict=True) must return a mapping")
    required = (
        "pressure",
        "pressure_unit",
        "condensate_mmr",
        "droplet_eff_r",
        "r_units",
        "condensibles",
    )
    missing = tuple(key for key in required if key not in result)
    if missing:
        raise RobertDataError(
            "Virga result is missing required cloud profile fields: "
            + ", ".join(missing)
        )

    returned_condensibles = tuple(str(item) for item in result["condensibles"])
    if returned_condensibles != condensibles:
        raise RobertDataError(
            "Virga result condensible order does not match the requested species"
        )
    _validate_virga_pressure_layers(result, pressure_grid)

    expected_shape = (pressure_grid.n_layers, len(condensibles))
    condensate_mmr = np.asarray(result["condensate_mmr"], dtype=np.float64)
    if condensate_mmr.shape != expected_shape:
        raise RobertDataError(
            "Virga condensate_mmr must have shape "
            f"{expected_shape}; received {condensate_mmr.shape}"
        )
    if not np.all(np.isfinite(condensate_mmr)) or np.any(condensate_mmr < 0.0):
        raise RobertDataError(
            "Virga condensate_mmr must contain finite non-negative values"
        )

    radius_unit = str(result["r_units"]).strip().lower().replace("μ", "u")
    if radius_unit not in {"micron", "microns", "um"}:
        raise RobertDataError(
            f"Virga droplet_eff_r must use microns; received {result['r_units']!r}"
        )
    radius = np.asarray(result["droplet_eff_r"], dtype=np.float64)
    if radius.shape != expected_shape:
        raise RobertDataError(
            "Virga droplet_eff_r must have shape "
            f"{expected_shape}; received {radius.shape}"
        )
    if not np.all(np.isfinite(radius)) or np.any(radius < 0.0):
        raise RobertDataError(
            "Virga droplet_eff_r must contain finite non-negative values"
        )
    if np.any((condensate_mmr > 0.0) & (radius <= 0.0)):
        raise RobertDataError(
            "Virga droplet_eff_r must be positive where condensate_mmr is positive"
        )

    if pressure_grid.orientation == "decreasing":
        condensate_mmr = condensate_mmr[::-1]
        radius = radius[::-1]

    metadata = {
        **dict(model_metadata),
        "virga_version": virga_version,
        "virga_profile_condensate_field": "condensate_mmr",
        "virga_profile_condensate_source_unit": "g/g",
        "virga_profile_condensate_unit": "kg/kg",
        "virga_profile_condensate_unit_conversion": (
            "mass_ratio_numeric_value_is_unchanged_from_g/g_to_kg/kg"
        ),
        "virga_profile_radius_field": "droplet_eff_r",
        "virga_profile_radius_unit": "micron",
        "virga_profile_species_order": ",".join(condensibles),
        "virga_profile_pressure_layer_mapping": "matched_pressure_edges",
        "virga_profile_pressure_orientation": pressure_grid.orientation,
        "virga_profile_source_layer_orientation": "increasing_pressure",
        "virga_profile_layer_reversal_applied": str(
            pressure_grid.orientation == "decreasing"
        ).lower(),
    }
    return VirgaCloudProfileResult(
        condensibles=condensibles,
        pressure_grid=pressure_grid,
        condensate_mass_mixing_ratio_kg_kg=condensate_mmr,
        effective_particle_radius_micron=radius,
        metadata=metadata,
    )


def _validate_virga_pressure_layers(
    result: Mapping[str, object], pressure_grid: PressureGrid
) -> None:
    """Require Virga layer centers to match ROBERT's pressure-edge intervals."""

    pressure_unit = str(result["pressure_unit"])
    virga_pressure = pressure_values_in_unit(result["pressure"], pressure_unit, "bar")
    if virga_pressure.ndim != 1 or not np.all(np.isfinite(virga_pressure)):
        raise RobertDataError(
            "Virga pressure layers must be a finite one-dimensional array"
        )
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
            "Virga result is missing required documented fields: " + ", ".join(missing)
        )

    returned_condensibles = tuple(str(item) for item in result["condensibles"])
    if returned_condensibles != condensibles:
        raise RobertDataError(
            "Virga result condensible order does not match the requested species"
        )

    pressure_grid = gas_optical_depth.pressure_grid
    _validate_virga_pressure_layers(result, pressure_grid)

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
    if not all(
        hasattr(virga, name) for name in ("Atmosphere", "compute", "gas_properties")
    ):
        raise RobertConfigError(
            "Virga cloud model requires the documented virga-exo 2.x API: "
            "virga.justdoit.Atmosphere, virga.justdoit.compute, and "
            "virga.justdoit.gas_properties"
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


def _virga_gas_mole_fractions_from_mass_fractions(
    gas_properties: object,
    *,
    condensibles: tuple[str, ...],
    gas_mass_mixing_ratios: Mapping[str, float],
    mean_molecular_weight: float,
    metallicity: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Convert physical gas mass fractions to Virga's mole-fraction inputs.

    Virga's gas-property functions return ``q = x * mu_condensible / mu_atmos``.
    We use the same function to get each species molecular weight, then check
    that the converted input recovers the user's physical mass fraction.
    """

    mole_fractions: dict[str, float] = {}
    molecular_weights: dict[str, float] = {}
    for species in condensibles:
        gas_property = getattr(gas_properties, species, None)
        if not callable(gas_property):
            raise RobertConfigError(
                f"Virga does not provide gas properties for condensible {species!r}"
            )
        try:
            gas_molecular_weight, _, _ = gas_property(
                mean_molecular_weight,
                mh=metallicity,
                gas_mmr=0.0,
            )
            gas_molecular_weight = float(gas_molecular_weight)
            if not np.isfinite(gas_molecular_weight) or gas_molecular_weight <= 0.0:
                raise ValueError("gas molecular weight must be finite and positive")

            mass_fraction = float(gas_mass_mixing_ratios[species])
            mole_fraction = mass_fraction * mean_molecular_weight / gas_molecular_weight
            if not np.isfinite(mole_fraction) or mole_fraction <= 0.0:
                raise ValueError(
                    "converted gas mole fraction must be finite and positive"
                )

            _, recovered_mass_fraction, _ = gas_property(
                mean_molecular_weight,
                mh=metallicity,
                gas_mmr=mole_fraction,
            )
            recovered_mass_fraction = float(recovered_mass_fraction)
        except Exception as exc:
            raise RobertDataError(
                f"Virga gas-property conversion failed for {species}: {exc}"
            ) from exc

        if not np.isfinite(recovered_mass_fraction) or not np.isclose(
            recovered_mass_fraction,
            mass_fraction,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise RobertDataError(
                "Virga gas-property conversion did not recover the configured "
                f"mass mixing ratio for {species}: requested {mass_fraction:.17g} g/g, "
                f"recovered {recovered_mass_fraction:.17g} g/g"
            )
        mole_fractions[species] = float(mole_fraction)
        molecular_weights[species] = gas_molecular_weight

    if sum(mole_fractions.values()) > 1.0 + 1.0e-12:
        raise RobertValidationError(
            "Converted Virga condensable gas mole fractions must sum to no more than one"
        )
    return mole_fractions, molecular_weights


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
        raise RobertValidationError(
            "Virga state profiles must be finite one-dimensional arrays"
        )
    if not np.allclose(profile, profile[0], rtol=1.0e-10, atol=0.0):
        raise RobertValidationError(message)
    if profile[0] <= 0.0:
        raise RobertValidationError(
            "Virga gravity and mean molecular weight must be positive"
        )
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


__all__ = ["VirgaCloudProfileResult", "VirgaEddySedCloudModel"]
