"""Transit light source effect transforms for transmission spectra.

The spectrum-based core is independent of any stellar atmosphere grid.  Grid
file access belongs in :func:`prepare_stellar_contamination_model`, which is
called while a forward model is constructed rather than in a likelihood call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from robert_exoplanets.bodies import Star
from robert_exoplanets.core import RobertValidationError, SpectralGrid, Spectrum
from robert_exoplanets.core._immutability import immutable_mapping

from .spectra import STELLAR_RADIANCE_UNIT, prepare_stellar_spectrum

RACKHAM_2018_DOI = "10.3847/1538-4357/aaa08c"
POSEIDON_STELLAR_CONTAMINATION_VERSION = "1.4.0"


@dataclass(frozen=True)
class StellarHeterogeneity:
    """One unocculted stellar surface component on the visible disk.

    Exactly one of ``covering_fraction`` and ``covering_fraction_parameter``
    is required. Fractions are projected visible-disk area fractions and are
    validated again at evaluation time so retrieved mixtures cannot exceed
    unity.
    """

    name: str
    spectrum: Spectrum
    kind: str = "heterogeneity"
    covering_fraction: float | None = None
    covering_fraction_parameter: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        kind = str(self.kind).strip().lower()
        if not name:
            raise RobertValidationError("stellar heterogeneity name must not be empty")
        if kind not in {"spot", "facula", "heterogeneity"}:
            raise RobertValidationError(
                "stellar heterogeneity kind must be 'spot', 'facula', or 'heterogeneity'"
            )
        fixed = self.covering_fraction
        parameter = self.covering_fraction_parameter
        if (fixed is None) == (parameter is None):
            raise RobertValidationError(
                f"stellar heterogeneity {name!r} requires exactly one fixed or parameterized covering fraction"
            )
        if fixed is not None:
            fixed = _fraction(fixed, f"{name} covering_fraction")
        if parameter is not None:
            parameter = str(parameter).strip()
            if not parameter:
                raise RobertValidationError(
                    "stellar covering-fraction parameter must not be empty"
                )
        _validate_stellar_spectrum(self.spectrum, f"{name} spectrum")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "covering_fraction", fixed)
        object.__setattr__(self, "covering_fraction_parameter", parameter)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    def fraction(self, parameters: Mapping[str, float]) -> float:
        """Return this component's validated projected-disk fraction."""

        if self.covering_fraction is not None:
            return self.covering_fraction
        assert self.covering_fraction_parameter is not None
        if self.covering_fraction_parameter not in parameters:
            raise RobertValidationError(
                "stellar contamination parameter is missing: "
                + self.covering_fraction_parameter
            )
        return _fraction(
            parameters[self.covering_fraction_parameter],
            self.covering_fraction_parameter,
        )


@dataclass(frozen=True)
class StellarContaminationResult:
    """Immutable spectra defining one evaluated TSLE transform."""

    photosphere_spectrum: Spectrum
    disk_integrated_spectrum: Spectrum
    transit_chord_spectrum: Spectrum
    contamination_factor: Spectrum
    covering_fractions: Mapping[str, float]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "covering_fractions",
            immutable_mapping(
                {str(name): float(value) for name, value in self.covering_fractions.items()}
            ),
        )
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class StellarContaminationModel:
    """Prepared TSLE model with an explicit disk and transit-chord spectrum.

    The out-of-transit disk radiance is

    ``I_disk = (1 - sum(f_i)) I_phot + sum(f_i I_i)``.

    The multiplicative factor is ``epsilon = I_chord / I_disk`` and the
    observed depth is ``D_obs = epsilon D_planet``.  Omitting
    ``transit_chord_spectrum`` selects ``I_chord = I_phot``, exactly matching
    POSEIDON 1.4's Rackham one- and two-heterogeneity models.
    """

    photosphere_spectrum: Spectrum
    heterogeneities: tuple[StellarHeterogeneity, ...] = ()
    transit_chord_spectrum: Spectrum | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_stellar_spectrum(self.photosphere_spectrum, "photosphere spectrum")
        heterogeneities = tuple(self.heterogeneities)
        if len({item.name for item in heterogeneities}) != len(heterogeneities):
            raise RobertValidationError("stellar heterogeneity names must be unique")
        parameter_names = tuple(
            item.covering_fraction_parameter
            for item in heterogeneities
            if item.covering_fraction_parameter is not None
        )
        if len(set(parameter_names)) != len(parameter_names):
            raise RobertValidationError(
                "stellar covering-fraction parameter names must be unique"
            )
        for item in heterogeneities:
            _validate_same_stellar_grid(
                self.photosphere_spectrum,
                item.spectrum,
                f"heterogeneity {item.name!r}",
            )
        fixed_total = sum(
            item.covering_fraction or 0.0 for item in heterogeneities
        )
        if fixed_total > 1.0:
            raise RobertValidationError(
                "fixed stellar heterogeneity fractions must sum to at most one"
            )
        chord = self.transit_chord_spectrum
        if chord is not None:
            _validate_stellar_spectrum(chord, "transit chord spectrum")
            _validate_same_stellar_grid(
                self.photosphere_spectrum,
                chord,
                "transit chord spectrum",
            )
        object.__setattr__(self, "heterogeneities", heterogeneities)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def required_parameters(self) -> tuple[str, ...]:
        """Covering-fraction parameters required at evaluation time."""

        return tuple(
            item.covering_fraction_parameter
            for item in self.heterogeneities
            if item.covering_fraction_parameter is not None
        )

    @property
    def manifest_metadata(self) -> Mapping[str, str]:
        """Compact science/provenance settings for run manifests."""

        records: dict[str, str] = {
            "stellar_contamination": "enabled",
            "stellar_contamination_formalism": "rackham_disk_mixture",
            "stellar_contamination_equation": "observed_depth=planet_depth*chord_radiance/disk_radiance",
            "stellar_contamination_poseidon_reference": POSEIDON_STELLAR_CONTAMINATION_VERSION,
            "stellar_contamination_rackham_doi": RACKHAM_2018_DOI,
            "stellar_contamination_region_count": str(len(self.heterogeneities)),
            "stellar_contamination_chord": (
                "photosphere"
                if self.transit_chord_spectrum is None
                else "explicit_spectrum"
            ),
            "stellar_contamination_required_parameters": ",".join(
                self.required_parameters
            ),
            "stellar_contamination_photosphere_model": str(
                self.photosphere_spectrum.metadata.get("stellar_model", "provided_spectrum")
            ),
        }
        records.update(
            _spectrum_provenance(
                self.photosphere_spectrum,
                "stellar_contamination_photosphere",
            )
        )
        for index, item in enumerate(self.heterogeneities):
            prefix = f"stellar_contamination_region_{index}"
            records.update(
                {
                    f"{prefix}_name": item.name,
                    f"{prefix}_kind": item.kind,
                    f"{prefix}_fraction": (
                        f"{item.covering_fraction:.17g}"
                        if item.covering_fraction is not None
                        else f"parameter:{item.covering_fraction_parameter}"
                    ),
                    f"{prefix}_model": str(
                        item.spectrum.metadata.get("stellar_model", "provided_spectrum")
                    ),
                    f"{prefix}_temperature_k": str(
                        item.spectrum.metadata.get("effective_temperature_k", "")
                    ),
                }
            )
            records.update(_spectrum_provenance(item.spectrum, prefix))
        if self.transit_chord_spectrum is not None:
            records.update(
                _spectrum_provenance(
                    self.transit_chord_spectrum,
                    "stellar_contamination_chord",
                )
            )
        records.update(self.metadata)
        return immutable_mapping(records)

    def evaluate(
        self,
        parameters: Mapping[str, float] | None = None,
    ) -> StellarContaminationResult:
        """Evaluate disk mixture, chord radiance, and multiplicative factor."""

        values = {} if parameters is None else parameters
        fractions = {
            item.name: item.fraction(values) for item in self.heterogeneities
        }
        total = float(sum(fractions.values()))
        if total > 1.0:
            raise RobertValidationError(
                "stellar heterogeneity fractions must sum to at most one"
            )
        photosphere = np.asarray(self.photosphere_spectrum.values, dtype=float)
        chord_spectrum = self.transit_chord_spectrum or self.photosphere_spectrum

        # Preserve exact identity for every physically homogeneous limit.
        homogeneous = (
            chord_spectrum is self.photosphere_spectrum
            and all(
                fraction == 0.0
                or np.array_equal(item.spectrum.values, photosphere)
                for item, fraction in zip(
                    self.heterogeneities, fractions.values(), strict=True
                )
            )
        )
        if homogeneous:
            disk_values = np.array(photosphere, copy=True)
            factor = np.ones_like(photosphere)
        else:
            disk_values = (1.0 - total) * photosphere
            for item in self.heterogeneities:
                disk_values = (
                    disk_values
                    + fractions[item.name] * np.asarray(item.spectrum.values, dtype=float)
                )
            if not np.all(np.isfinite(disk_values)) or np.any(disk_values <= 0.0):
                raise RobertValidationError(
                    "disk-integrated stellar radiance must be finite and positive"
                )
            if chord_spectrum is self.photosphere_spectrum:
                # Retain POSEIDON's operation ordering for bit-level parity of
                # the benchmarked immaculate-chord subset.
                contrast_sum = np.zeros_like(photosphere)
                for item in self.heterogeneities:
                    contrast_sum += fractions[item.name] * (
                        1.0 - np.asarray(item.spectrum.values, dtype=float) / photosphere
                    )
                factor = 1.0 / (1.0 - contrast_sum)
            else:
                factor = np.asarray(chord_spectrum.values, dtype=float) / disk_values
            if not np.all(np.isfinite(factor)) or np.any(factor <= 0.0):
                raise RobertValidationError(
                    "stellar contamination factor must be finite and positive"
                )

        result_metadata = {
            **dict(self.manifest_metadata),
            "stellar_photosphere_fraction": f"{1.0 - total:.17g}",
            **{
                f"stellar_fraction_{name}": f"{fraction:.17g}"
                for name, fraction in fractions.items()
            },
        }
        grid = self.photosphere_spectrum.spectral_grid
        disk = Spectrum(
            spectral_grid=grid,
            values=disk_values,
            unit=STELLAR_RADIANCE_UNIT,
            observable="disk_integrated_stellar_radiance",
            metadata=result_metadata,
        )
        chord = Spectrum(
            spectral_grid=grid,
            values=chord_spectrum.values,
            unit=STELLAR_RADIANCE_UNIT,
            observable="transit_chord_stellar_radiance",
            metadata=result_metadata,
        )
        epsilon = Spectrum(
            spectral_grid=grid,
            values=factor,
            unit="dimensionless",
            observable="stellar_contamination_factor",
            metadata=result_metadata,
        )
        return StellarContaminationResult(
            photosphere_spectrum=self.photosphere_spectrum,
            disk_integrated_spectrum=disk,
            transit_chord_spectrum=chord,
            contamination_factor=epsilon,
            covering_fractions=fractions,
            metadata=result_metadata,
        )

    def apply(
        self,
        uncontaminated_transit_depth: Spectrum,
        parameters: Mapping[str, float] | None = None,
    ) -> Spectrum:
        """Apply TSLE to a native planetary transit-depth spectrum."""

        if (
            uncontaminated_transit_depth.observable != "transit_depth"
            or uncontaminated_transit_depth.unit != "transit_depth"
        ):
            raise RobertValidationError(
                "stellar contamination requires a transit_depth spectrum"
            )
        _validate_equivalent_wavelength_grid(
            uncontaminated_transit_depth.spectral_grid,
            self.photosphere_spectrum.spectral_grid,
        )
        result = self.evaluate(parameters)
        metadata = {
            **dict(uncontaminated_transit_depth.metadata),
            **dict(result.metadata),
            "stellar_contamination_application_order": "native_transit_depth_before_instrument_response",
            "transmission_diagnostics_convention": "effective_radius_and_annulus_are_uncontaminated_planetary_quantities",
        }
        return Spectrum(
            spectral_grid=uncontaminated_transit_depth.spectral_grid,
            values=(
                uncontaminated_transit_depth.values
                * result.contamination_factor.values
            ),
            unit=uncontaminated_transit_depth.unit,
            observable=uncontaminated_transit_depth.observable,
            metadata=metadata,
        )


@dataclass(frozen=True)
class StellarHeterogeneityDefinition:
    """Setup-time stellar-grid coordinates for one active-region spectrum."""

    name: str
    kind: str
    temperature_k: float
    covering_fraction: float | None = None
    covering_fraction_parameter: str | None = None
    log_g_cgs: float | None = None
    metallicity_dex: float | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        kind = str(self.kind).strip().lower()
        if not name:
            raise RobertValidationError("stellar heterogeneity name must not be empty")
        if kind not in {"spot", "facula", "heterogeneity"}:
            raise RobertValidationError(
                "stellar heterogeneity kind must be 'spot', 'facula', or 'heterogeneity'"
            )
        temperature = float(self.temperature_k)
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise RobertValidationError(
                "stellar heterogeneity temperature_k must be finite and positive"
            )
        if (self.covering_fraction is None) == (
            self.covering_fraction_parameter is None
        ):
            raise RobertValidationError(
                f"stellar heterogeneity {name!r} requires exactly one fixed or parameterized covering fraction"
            )
        fixed = self.covering_fraction
        if fixed is not None:
            fixed = _fraction(fixed, f"{name} covering_fraction")
        parameter = self.covering_fraction_parameter
        if parameter is not None:
            parameter = str(parameter).strip()
            if not parameter:
                raise RobertValidationError(
                    "stellar covering-fraction parameter must not be empty"
                )
        for value, label in (
            (self.log_g_cgs, "log_g_cgs"),
            (self.metallicity_dex, "metallicity_dex"),
        ):
            if value is not None and not np.isfinite(float(value)):
                raise RobertValidationError(
                    f"stellar heterogeneity {label} must be finite when provided"
                )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "covering_fraction", fixed)
        object.__setattr__(self, "covering_fraction_parameter", parameter)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


def prepare_stellar_contamination_model(
    star: Star,
    spectral_grid: SpectralGrid,
    *,
    heterogeneities: Sequence[StellarHeterogeneityDefinition] = (),
    transit_chord_temperature_k: float | None = None,
    transit_chord_log_g_cgs: float | None = None,
    transit_chord_metallicity_dex: float | None = None,
    spectrum_model: str = "phoenix",
    metadata: Mapping[str, str] | None = None,
) -> StellarContaminationModel:
    """Prepare all TSLE spectra once on a transmission model's native grid.

    PHOENIX is the science path. ``blackbody`` is retained only as a controlled
    approximation for demonstrations and algebraic tests; it is not the
    stellar-atmosphere validation standard.
    """

    if star.effective_temperature_k is None:
        raise RobertValidationError(
            "stellar contamination requires star.effective_temperature_k"
        )
    definitions = tuple(heterogeneities)
    photosphere = prepare_stellar_spectrum(star, spectral_grid, model=spectrum_model)
    regions: list[StellarHeterogeneity] = []
    for definition in definitions:
        kind = str(definition.kind).strip().lower()
        if kind == "spot" and definition.temperature_k >= star.effective_temperature_k:
            raise RobertValidationError(
                "stellar spot temperature must be cooler than the photosphere"
            )
        if kind == "facula" and definition.temperature_k <= star.effective_temperature_k:
            raise RobertValidationError(
                "stellar facula temperature must be hotter than the photosphere"
            )
        region_star = Star(
            name=f"{star.name} {definition.name}",
            radius_m=star.radius_m,
            effective_temperature_k=definition.temperature_k,
            log_g_cgs=(
                star.log_g_cgs
                if definition.log_g_cgs is None
                else definition.log_g_cgs
            ),
            metallicity_dex=(
                star.metallicity_dex
                if definition.metallicity_dex is None
                else definition.metallicity_dex
            ),
            metadata={"stellar_region": definition.name},
        )
        regions.append(
            StellarHeterogeneity(
                name=definition.name,
                kind=kind,
                spectrum=prepare_stellar_spectrum(
                    region_star, spectral_grid, model=spectrum_model
                ),
                covering_fraction=definition.covering_fraction,
                covering_fraction_parameter=definition.covering_fraction_parameter,
                metadata=definition.metadata,
            )
        )

    chord = None
    if transit_chord_temperature_k is not None:
        temperature = float(transit_chord_temperature_k)
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise RobertValidationError(
                "transit chord temperature must be finite and positive"
            )
        chord_star = Star(
            name=f"{star.name} transit chord",
            radius_m=star.radius_m,
            effective_temperature_k=temperature,
            log_g_cgs=(
                star.log_g_cgs
                if transit_chord_log_g_cgs is None
                else transit_chord_log_g_cgs
            ),
            metallicity_dex=(
                star.metallicity_dex
                if transit_chord_metallicity_dex is None
                else transit_chord_metallicity_dex
            ),
            metadata={"stellar_region": "transit_chord"},
        )
        chord = prepare_stellar_spectrum(
            chord_star, spectral_grid, model=spectrum_model
        )

    model_metadata = {
        "stellar_spectrum_model": str(spectrum_model).lower(),
        "blackbody_scope": (
            "controlled_approximation_not_validation_standard"
            if str(spectrum_model).lower() == "blackbody"
            else "not_applicable"
        ),
        **({} if metadata is None else dict(metadata)),
    }
    return StellarContaminationModel(
        photosphere_spectrum=photosphere,
        heterogeneities=tuple(regions),
        transit_chord_spectrum=chord,
        metadata=model_metadata,
    )


def _validate_stellar_spectrum(spectrum: Spectrum, label: str) -> None:
    if spectrum.unit != STELLAR_RADIANCE_UNIT:
        raise RobertValidationError(
            f"{label} unit must be {STELLAR_RADIANCE_UNIT!r}"
        )
    if spectrum.observable != "stellar_spectral_radiance":
        raise RobertValidationError(
            f"{label} observable must be 'stellar_spectral_radiance'"
        )
    values = np.asarray(spectrum.values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise RobertValidationError(f"{label} must be finite and positive")


def _spectrum_provenance(spectrum: Spectrum, prefix: str) -> dict[str, str]:
    keys = (
        "stellar_model",
        "stellar_model_scope",
        "stellar_library",
        "stellar_loader",
        "effective_temperature_k",
        "metallicity_dex",
        "log_g_cgs",
        "bolometric_normalization",
        "bolometric_normalization_factor",
        "surface_flux_convention",
        "spectral_sampling",
        "pysyn_cdbs",
    )
    return {
        f"{prefix}_{key}": str(spectrum.metadata[key])
        for key in keys
        if key in spectrum.metadata
    }


def _validate_same_stellar_grid(
    reference: Spectrum, candidate: Spectrum, label: str
) -> None:
    if reference.spectral_grid.unit != candidate.spectral_grid.unit or not np.array_equal(
        reference.spectral_grid.values, candidate.spectral_grid.values
    ):
        raise RobertValidationError(
            f"{label} must share the photosphere spectral grid without interpolation"
        )


def _fraction(value: float, label: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise RobertValidationError(f"{label} must be finite and lie in [0, 1]")
    return result


def _validate_equivalent_wavelength_grid(
    transit_grid: SpectralGrid,
    stellar_grid: SpectralGrid,
) -> None:
    transit_wavelength = _wavelength_micron(transit_grid)
    stellar_wavelength = _wavelength_micron(stellar_grid)
    if transit_wavelength.shape != stellar_wavelength.shape or not np.allclose(
        transit_wavelength,
        stellar_wavelength,
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise RobertValidationError(
            "stellar contamination spectrum does not cover the transmission spectral grid"
        )


def _wavelength_micron(grid: SpectralGrid) -> np.ndarray:
    values = np.asarray(grid.values, dtype=float)
    unit = grid.unit.strip().lower().replace("μ", "u").replace("µ", "u")
    if unit in {"micron", "microns", "um"}:
        return values
    if unit in {"nm", "nanometer", "nanometers"}:
        return values * 1.0e-3
    if unit in {"angstrom", "angstroms", "aa"}:
        return values * 1.0e-4
    if unit in {"m", "meter", "meters"}:
        return values * 1.0e6
    if unit in {"cm^-1", "cm-1", "1/cm"}:
        return 1.0e4 / values
    raise RobertValidationError(
        f"unsupported stellar contamination spectral-grid unit {grid.unit!r}"
    )


__all__ = [
    "StellarContaminationModel",
    "StellarContaminationResult",
    "StellarHeterogeneity",
    "StellarHeterogeneityDefinition",
    "prepare_stellar_contamination_model",
]
