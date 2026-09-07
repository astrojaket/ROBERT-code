"""Prepared cloud-free emission simulation and likelihood on JAX Metal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from robert_exoplanets.core import PressureGrid, RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.retrieval.priors import RetrievalParameterSet
from robert_exoplanets.rt.geometry import gauss_legendre_disk_geometry

from .forward import (
    PreparedMetalCorrelatedK,
    assemble_gas_optical_depth_device,
    integrate_clear_thermal_device,
    pg14_temperature_device,
    planck_radiance_wavelength_device,
)
from .likelihood import MetalLinearGaussianLikelihood
from .likelihood import MetalLinearProjection
from .problem import MetalRetrievalProblem


@dataclass(frozen=True)
class PreparedMetalCloudFreeEmission:
    """Fixed-grid free-chemistry emission graph retained on one Metal device.

    Parameter order is the five PG14 parameters followed by one ``log10(VMR)``
    for each opacity species. The remaining gas is an explicit H2/He background;
    mean molecular weight is evaluated from the retrieved trace abundances.
    """

    runtime: Any
    opacity: PreparedMetalCorrelatedK
    species: tuple[str, ...]
    molecular_weights_amu: Any
    pressure_center_bar: Any
    pressure_edge_bar: Any
    wavelength_micron: Any
    gravity_m_s2: float
    planet_radius_m: float
    star_radius_m: float
    star_temperature_k: float
    emission_path_factors: Any
    emission_point_weights: Any
    likelihood: MetalLinearGaussianLikelihood
    opacity_identifiers: Mapping[str, str] = field(default_factory=dict)
    background_mean_molecular_weight_amu: float = 2.3
    internal_temperature_k: float = 100.0

    def __post_init__(self) -> None:
        if (
            not self.species
            or any(not name for name in self.species)
            or len(set(self.species)) != len(self.species)
        ):
            raise RobertValidationError(
                "cloud-free opacity species must be non-empty and unique"
            )
        if self.molecular_weights_amu.shape != (len(self.species),):
            raise RobertValidationError(
                "one molecular weight is required for each opacity species"
            )
        if self.pressure_center_bar.shape != (self.pressure_edge_bar.shape[0] - 1,):
            raise RobertValidationError("pressure centers and edges are inconsistent")
        if (
            len(self.emission_path_factors.shape) != 2
            or self.emission_path_factors.shape[1] != self.pressure_center_bar.shape[0]
            or self.emission_point_weights.shape
            != (self.emission_path_factors.shape[0],)
        ):
            raise RobertValidationError(
                "emission path factors and weights are inconsistent"
            )
        if self.likelihood.projection.native_size != self.wavelength_micron.shape[0]:
            raise RobertValidationError(
                "likelihood projection columns must match native wavelengths"
            )
        positive_scalars = {
            "gravity_m_s2": self.gravity_m_s2,
            "planet_radius_m": self.planet_radius_m,
            "star_radius_m": self.star_radius_m,
            "star_temperature_k": self.star_temperature_k,
            "background_mean_molecular_weight_amu": (
                self.background_mean_molecular_weight_amu
            ),
        }
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in positive_scalars.values()
        ):
            raise RobertValidationError(
                "gravity, radii, stellar temperature, and background MMW must be finite and positive"
            )
        if (
            not np.isfinite(self.internal_temperature_k)
            or self.internal_temperature_k < 0.0
        ):
            raise RobertValidationError(
                "internal_temperature_k must be finite and non-negative"
            )
        object.__setattr__(
            self, "opacity_identifiers", immutable_mapping(self.opacity_identifiers)
        )

    @property
    def n_parameters(self) -> int:
        return len(self.species) + 5

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "kappa_IR",
            "gamma1",
            "gamma2",
            "T_irr",
            "alpha",
            *(f"log_{name}" for name in self.species),
        )

    def native_eclipse_depth(self, parameter_vector: Any):
        """Return a native-grid eclipse-depth vector without host transfer."""

        jnp = self.runtime.jnp
        parameters = jnp.asarray(parameter_vector, dtype=jnp.float32)
        if parameters.shape != (self.n_parameters,):
            raise RobertValidationError(
                "cloud-free parameter vector must contain PG14 and species log VMRs"
            )
        kappa_ir, gamma1, gamma2, irradiation, alpha = parameters[:5]
        trace_vmr = jnp.power(jnp.float32(10.0), parameters[5:])
        trace_sum = jnp.sum(trace_vmr)
        background = 1.0 - trace_sum
        mean_molecular_weight = background * jnp.float32(
            self.background_mean_molecular_weight_amu
        ) + jnp.sum(trace_vmr * self.molecular_weights_amu)
        n_layers = self.pressure_center_bar.shape[0]
        temperature = pg14_temperature_device(
            self.pressure_center_bar,
            kappa_ir_m2_kg=kappa_ir,
            gamma1=gamma1,
            gamma2=gamma2,
            irradiation_temperature_k=irradiation,
            alpha=alpha,
            gravity_m_s2=jnp.float32(self.gravity_m_s2),
            internal_temperature_k=jnp.float32(self.internal_temperature_k),
        )
        level_temperature = pg14_temperature_device(
            self.pressure_edge_bar,
            kappa_ir_m2_kg=kappa_ir,
            gamma1=gamma1,
            gamma2=gamma2,
            irradiation_temperature_k=irradiation,
            alpha=alpha,
            gravity_m_s2=jnp.float32(self.gravity_m_s2),
            internal_temperature_k=jnp.float32(self.internal_temperature_k),
        )
        composition = jnp.broadcast_to(
            trace_vmr[:, None], (len(self.species), n_layers)
        )
        kcoeff = self.opacity.interpolate(self.pressure_center_bar, temperature)
        tau = assemble_gas_optical_depth_device(
            kcoeff,
            composition,
            self.pressure_edge_bar,
            jnp.full((n_layers,), mean_molecular_weight),
            jnp.float32(self.gravity_m_s2),
            self.opacity.g_weights,
            opacity_unit_scale_m2=self.opacity.opacity_unit_scale_m2,
            random_overlap=True,
        )
        level_source = planck_radiance_wavelength_device(
            self.wavelength_micron, level_temperature
        )
        planet_radiance = integrate_clear_thermal_device(
            tau,
            level_source,
            self.opacity.g_weights,
            self.emission_path_factors,
            self.emission_point_weights,
            level_source[-1],
            jnp.ones((self.emission_path_factors.shape[0],), dtype=bool),
        )
        stellar_radiance = planck_radiance_wavelength_device(
            self.wavelength_micron, jnp.float32(self.star_temperature_k)
        )
        radius_ratio_squared = jnp.float32(
            (self.planet_radius_m / self.star_radius_m) ** 2
        )
        eclipse_depth = radius_ratio_squared * planet_radiance / stellar_radiance
        valid = (
            jnp.all(jnp.isfinite(temperature))
            & jnp.all(temperature > 0.0)
            & jnp.all(jnp.isfinite(trace_vmr))
            & jnp.all(trace_vmr > 0.0)
            & (trace_sum < 1.0)
            & (mean_molecular_weight > 0.0)
            & jnp.all(jnp.isfinite(eclipse_depth))
        )
        return jnp.where(valid, eclipse_depth, jnp.nan)

    def loglike(self, parameter_vector: Any):
        """Return the projected cloud-free Gaussian likelihood on device."""

        return self.likelihood.loglike(self.native_eclipse_depth(parameter_vector))

    def retrieval_problem(
        self,
        *,
        name: str,
        parameters: RetrievalParameterSet,
    ) -> MetalRetrievalProblem:
        """Bind the compiled graph to ROBERT's sampler-facing contract."""

        if parameters.ndim != self.n_parameters:
            raise RobertValidationError(
                "retrieval parameters do not match cloud-free graph parameter order"
            )
        if parameters.names != self.parameter_names:
            raise RobertValidationError(
                "retrieval parameter order must be: " + ", ".join(self.parameter_names)
            )
        return MetalRetrievalProblem.from_device_graph(
            name=name,
            parameters=parameters,
            parameter_to_loglike=self.loglike,
            runtime=self.runtime,
            likelihood=self.likelihood,
            opacity_identifiers=self.opacity_identifiers,
            metadata={
                "model": "cloud-free-emission",
                "temperature": "parmentier-guillot-2014",
                "chemistry": "free-trace-plus-h2-he-background",
                "species": ",".join(self.species),
            },
        )


def molecular_weights_for_species(species: tuple[str, ...]) -> np.ndarray:
    """Return exact configured molecular masses for supported free chemistry."""

    known = {
        "H2O": 18.01528,
        "CO2": 44.0095,
        "CO": 28.0101,
        "CH4": 16.0425,
        "NH3": 17.0305,
        "HCN": 27.0253,
    }
    missing = tuple(name for name in species if name not in known)
    if missing:
        raise RobertValidationError(
            "missing molecular weights for Metal free chemistry: " + ", ".join(missing)
        )
    return np.asarray([known[name] for name in species], dtype=np.float32)


def prepare_cloud_free_emission_from_provider(
    *,
    provider: Any,
    species: tuple[str, ...],
    spectral_indices: Any,
    pressure_center_bar: Any,
    pressure_edge_bar: Any,
    projection_matrix: Any,
    observation: Any,
    uncertainty: Any,
    runtime: Any,
    gravity_m_s2: float,
    planet_radius_m: float,
    star_radius_m: float,
    star_temperature_k: float,
    include_likelihood_normalization: bool = False,
    disc_quadrature_points: int = 4,
    background_mean_molecular_weight_amu: float = 2.3,
    internal_temperature_k: float = 100.0,
) -> PreparedMetalCloudFreeEmission:
    """Prepare a user-facing cloud-free graph from loaded CPU table containers.

    Table loading and validation are one-time host setup.  The returned model's
    simulation and likelihood methods contain no CPU numerical callback.
    All species must share pressure, temperature, spectral, and g grids.
    """

    if (
        not species
        or any(not name for name in species)
        or len(set(species)) != len(species)
        or any(name not in provider.tables for name in species)
    ):
        raise RobertValidationError("cloud-free provider is missing requested species")
    interpolation = str(getattr(provider, "interpolation", ""))
    interpolation_modes = {
        "log_pressure_temperature_log_k": False,
        "log_pressure_temperature_log_k_clip": True,
    }
    if interpolation not in interpolation_modes:
        raise RobertValidationError(
            "Metal cloud-free correlated-k requires log-pressure/temperature/log-k interpolation"
        )
    if (
        not isinstance(disc_quadrature_points, int)
        or isinstance(disc_quadrature_points, bool)
        or disc_quadrature_points < 1
    ):
        raise RobertValidationError("disc_quadrature_points must be a positive integer")
    reference = provider.tables[species[0]]
    selected = np.asarray(spectral_indices, dtype=np.int32)
    if (
        selected.ndim != 1
        or selected.size < 1
        or np.any(selected < 0)
        or np.any(selected >= reference.wavenumber_cm_inverse.size)
        or np.unique(selected).size != selected.size
    ):
        raise RobertValidationError("spectral_indices must be a non-empty vector")
    tables = []
    opacity_identifiers: dict[str, str] = {}
    for name in species:
        table = provider.tables[name]
        if (
            not np.array_equal(table.pressure_bar, reference.pressure_bar)
            or not np.array_equal(table.temperature_K, reference.temperature_K)
            or not np.array_equal(
                table.wavenumber_cm_inverse, reference.wavenumber_cm_inverse
            )
            or not np.array_equal(table.wavelength_micron, reference.wavelength_micron)
            or not np.array_equal(table.g_samples, reference.g_samples)
            or not np.array_equal(table.g_weights, reference.g_weights)
            or table.unit != reference.unit
        ):
            raise RobertValidationError(
                "Metal cloud-free species must share all correlated-k grids"
            )
        tables.append(table.kcoeff)
        for key in ("source_path", "checksum_sha256", "doi", "line_list"):
            value = str(table.metadata.get(key, ""))
            if value:
                opacity_identifiers[f"{name}:{key}"] = value
    opacity = PreparedMetalCorrelatedK.from_arrays(
        np.stack(tables),
        reference.pressure_bar,
        reference.temperature_K,
        np.broadcast_to(selected, (len(species), selected.size)),
        reference.g_weights,
        runtime,
        opacity_unit=reference.unit,
        clip=interpolation_modes[interpolation],
    )
    projection = MetalLinearProjection.from_matrix(projection_matrix, runtime)
    likelihood = MetalLinearGaussianLikelihood.from_arrays(
        observation,
        uncertainty,
        projection,
        runtime,
        include_normalization=include_likelihood_normalization,
    )
    pressure_grid = PressureGrid(
        edges=np.asarray(pressure_edge_bar, dtype=float),
        centers=np.asarray(pressure_center_bar, dtype=float),
        unit="bar",
    )
    centers = np.asarray(pressure_grid.centers, dtype=np.float32)
    edges = np.asarray(pressure_grid.edges, dtype=np.float32)
    if not interpolation_modes[interpolation] and (
        np.min(centers) < reference.pressure_bar[0]
        or np.max(centers) > reference.pressure_bar[-1]
    ):
        raise RobertValidationError(
            "cloud-free pressure centers lie outside strict opacity coverage"
        )
    geometry = gauss_legendre_disk_geometry(disc_quadrature_points)
    path_factors = np.broadcast_to(
        1.0 / geometry.emission_angle_cosines[:, None],
        (geometry.n_points, centers.size),
    ).copy()
    return PreparedMetalCloudFreeEmission(
        runtime=runtime,
        opacity=opacity,
        species=species,
        molecular_weights_amu=runtime.put(molecular_weights_for_species(species)),
        pressure_center_bar=runtime.put(centers),
        pressure_edge_bar=runtime.put(edges),
        wavelength_micron=runtime.put(reference.wavelength_micron[selected]),
        gravity_m_s2=float(gravity_m_s2),
        planet_radius_m=float(planet_radius_m),
        star_radius_m=float(star_radius_m),
        star_temperature_k=float(star_temperature_k),
        emission_path_factors=runtime.put(path_factors),
        emission_point_weights=runtime.put(geometry.emission_angle_weights),
        likelihood=likelihood,
        opacity_identifiers=opacity_identifiers,
        background_mean_molecular_weight_amu=float(
            background_mean_molecular_weight_amu
        ),
        internal_temperature_k=float(internal_temperature_k),
    )


__all__ = [
    "PreparedMetalCloudFreeEmission",
    "molecular_weights_for_species",
    "prepare_cloud_free_emission_from_provider",
]
