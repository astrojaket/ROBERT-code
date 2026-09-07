"""JAX opacity-sampling clear-emission graph.

This is deliberately separate from both ROBERT's CPU opacity-sampling path and
the Metal correlated-k graph.  It preserves the sampled cross sections on
their shared native spectral points and combines species by direct optical-depth
addition; it never sorts opacities or invokes random overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.opacity.metadata import (
    pressure_values_in_unit,
    spectral_grid_values_in_unit,
)
from robert_exoplanets.opacity.opacity_sampling import _exact_spectral_indices
from robert_exoplanets.retrieval.priors import RetrievalParameterSet
from robert_exoplanets.rt.geometry import gauss_legendre_disk_geometry

from .cloud_free import molecular_weights_for_species
from .forward import (
    ATOMIC_MASS_KG,
    integrate_clear_thermal_device,
    pg14_temperature_device,
    planck_radiance_wavelength_device,
)
from .likelihood import (
    MetalBinnedProjection,
    MetalIdentityProjection,
    MetalLinearGaussianLikelihood,
    MetalLinearProjection,
)
from .problem import MetalRetrievalProblem
from .resources import MetalResourcePolicy


def _opacity_unit_scale_m2(unit: str) -> float:
    normalized = unit.strip().lower().replace(" ", "")
    if normalized in {
        "cm^2/molecule",
        "cm2/molecule",
        "cm^2molecule^-1",
        "cm2molecule-1",
    }:
        return 1.0e-4
    if normalized in {"m^2/molecule", "m2/molecule", "m^2molecule^-1", "m2molecule-1"}:
        return 1.0
    raise RobertValidationError(f"unsupported opacity unit: {unit}")


def interpolate_opacity_sampling_device(
    log_cross_sections: Any,
    log_pressure_grid_bar: Any,
    temperature_grid_k: Any,
    pressure_bar: Any,
    temperature_k: Any,
    *,
    clip: bool = False,
    return_log: bool = False,
):
    """Bilinearly interpolate native sampled cross sections in log space.

    The array has shape ``species, pressure, temperature, wavelength``.  This
    is the same interpolation space and direct spectral sampling as the CPU
    ``OpacitySamplingProvider`` implementation.
    """

    from .forward import _jax_modules

    jax, jnp = _jax_modules()
    table = jnp.asarray(log_cross_sections, dtype=jnp.float32)
    pgrid = jnp.asarray(log_pressure_grid_bar, dtype=jnp.float32)
    tgrid = jnp.asarray(temperature_grid_k, dtype=jnp.float32)
    pressure = jnp.log(jnp.asarray(pressure_bar, dtype=jnp.float32))
    temperature = jnp.asarray(temperature_k, dtype=jnp.float32)
    valid = (
        jnp.all(jnp.isfinite(pressure))
        & jnp.all(jnp.isfinite(temperature))
        & jnp.all(pressure >= pgrid[0])
        & jnp.all(pressure <= pgrid[-1])
        & jnp.all(temperature >= tgrid[0])
        & jnp.all(temperature <= tgrid[-1])
    )
    if clip:
        pressure = jnp.clip(pressure, pgrid[0], pgrid[-1])
        temperature = jnp.clip(temperature, tgrid[0], tgrid[-1])
    p1 = jnp.clip(jnp.searchsorted(pgrid, pressure, side="right"), 1, pgrid.size - 1)
    p0 = p1 - 1
    t1 = jnp.clip(jnp.searchsorted(tgrid, temperature, side="right"), 1, tgrid.size - 1)
    t0 = t1 - 1
    wp = (pressure - pgrid[p0]) / (pgrid[p1] - pgrid[p0])
    wt = (temperature - tgrid[t0]) / (tgrid[t1] - tgrid[t0])

    def one_species(values):
        v00 = values[p0, t0, :]
        v10 = values[p1, t0, :]
        v01 = values[p0, t1, :]
        v11 = values[p1, t1, :]
        value = (
            (1.0 - wp[:, None]) * (1.0 - wt[:, None]) * v00
            + wp[:, None] * (1.0 - wt[:, None]) * v10
            + (1.0 - wp[:, None]) * wt[:, None] * v01
            + wp[:, None] * wt[:, None] * v11
        )
        return value if return_log else jnp.exp(value)

    result = jax.vmap(one_species)(table)
    return result if clip else jnp.where(valid, result, jnp.nan)


def assemble_opacity_sampling_log_optical_depth_device(
    log_cross_sections: Any,
    volume_mixing_ratio: Any,
    pressure_edge_bar: Any,
    mean_molecular_weight_amu: Any,
    gravity_m_s2: Any,
    *,
    opacity_unit_scale_m2: float,
):
    """Assemble sampled tau in log space before the final exponentiation."""

    from .forward import _jax_modules

    _, jnp = _jax_modules()
    log_xsec = jnp.asarray(log_cross_sections, dtype=jnp.float32)
    vmr = jnp.asarray(volume_mixing_ratio, dtype=jnp.float32)
    edges = jnp.asarray(pressure_edge_bar, dtype=jnp.float32)
    mmw = jnp.asarray(mean_molecular_weight_amu, dtype=jnp.float32)
    gravity = jnp.asarray(gravity_m_s2, dtype=jnp.float32)
    delta_pressure = jnp.abs(jnp.diff(edges)) * jnp.float32(1.0e5)
    log_column = jnp.log(delta_pressure) - jnp.log(
        mmw * jnp.float32(ATOMIC_MASS_KG) * gravity
    )
    log_species_tau = (
        log_xsec
        + jnp.log(vmr[:, :, None])
        + log_column[None, :, None]
        + jnp.log(jnp.float32(opacity_unit_scale_m2))
    )
    return jnp.sum(jnp.exp(log_species_tau), axis=0)[..., None]


def assemble_opacity_sampling_optical_depth_device(
    cross_sections: Any,
    volume_mixing_ratio: Any,
    pressure_edge_bar: Any,
    mean_molecular_weight_amu: Any,
    gravity_m_s2: Any,
    *,
    opacity_unit_scale_m2: float,
):
    """Directly sum sampled species optical depths on the device."""

    from .forward import _jax_modules

    _, jnp = _jax_modules()
    xsec = jnp.asarray(cross_sections, dtype=jnp.float32)
    vmr = jnp.asarray(volume_mixing_ratio, dtype=jnp.float32)
    edges = jnp.asarray(pressure_edge_bar, dtype=jnp.float32)
    mmw = jnp.asarray(mean_molecular_weight_amu, dtype=jnp.float32)
    gravity = jnp.asarray(gravity_m_s2, dtype=jnp.float32)
    column = jnp.abs(jnp.diff(edges)) * jnp.float32(1.0e5)
    column = column / (mmw * jnp.float32(ATOMIC_MASS_KG) * gravity)
    mixture = jnp.sum(xsec * vmr[:, :, None], axis=0)
    # A singleton compatibility ordinate preserves the shared RT interface.
    return (mixture * column[:, None] * jnp.float32(opacity_unit_scale_m2))[..., None]


@dataclass(frozen=True)
class PreparedMetalOpacitySampling:
    """Validated, device-resident opacity-sampling table state."""

    log_cross_sections: Any
    log_pressure_grid_bar: Any
    temperature_grid_k: Any
    opacity_unit_scale_m2: float
    clip: bool = False

    @classmethod
    def from_arrays(
        cls,
        cross_sections: Any,
        pressure_grid_bar: Any,
        temperature_grid_k: Any,
        runtime: Any,
        *,
        opacity_unit: str = "cm^2/molecule",
        clip: bool = False,
        resource_policy: MetalResourcePolicy | None = None,
    ) -> "PreparedMetalOpacitySampling":
        values = np.asarray(cross_sections, dtype=np.float64)
        pressure = np.asarray(pressure_grid_bar, dtype=np.float64)
        temperature = np.asarray(temperature_grid_k, dtype=np.float64)
        if (
            values.ndim != 4
            or values.shape[1:3] != (pressure.size, temperature.size)
            or np.any(values < 0.0)
            or not np.all(np.isfinite(values))
            or pressure.size < 2
            or temperature.size < 2
            or np.any(pressure <= 0.0)
            or np.any(np.diff(pressure) <= 0.0)
            or np.any(np.diff(temperature) <= 0.0)
        ):
            raise RobertValidationError("invalid opacity-sampling preparation arrays")
        policy = (
            MetalResourcePolicy.laptop_safe()
            if resource_policy is None
            else resource_policy
        )
        policy.require_safe("opacity-sampling table transfer", values.size * 4 * 3)
        return cls(
            runtime.put(np.log(np.maximum(values, 1.0e-300))),
            runtime.put(np.log(pressure)),
            runtime.put(temperature),
            _opacity_unit_scale_m2(opacity_unit),
            bool(clip),
        )

    def interpolate(self, pressure_bar: Any, temperature_k: Any):
        return interpolate_opacity_sampling_device(
            self.log_cross_sections,
            self.log_pressure_grid_bar,
            self.temperature_grid_k,
            pressure_bar,
            temperature_k,
            clip=self.clip,
        )

    def interpolate_log(self, pressure_bar: Any, temperature_k: Any):
        return interpolate_opacity_sampling_device(
            self.log_cross_sections,
            self.log_pressure_grid_bar,
            self.temperature_grid_k,
            pressure_bar,
            temperature_k,
            clip=self.clip,
            return_log=True,
        )


@dataclass(frozen=True)
class PreparedMetalOpacitySamplingCloudFreeEmission:
    """Single-proposal PG14 free-chemistry emission graph using sampled opacity."""

    runtime: Any
    opacity: PreparedMetalOpacitySampling
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
    metadata: Mapping[str, str] = field(default_factory=dict)
    background_mean_molecular_weight_amu: float = 2.3
    internal_temperature_k: float = 100.0

    def __post_init__(self) -> None:
        if (
            not self.species
            or any(not name for name in self.species)
            or len(set(self.species)) != len(self.species)
        ):
            raise RobertValidationError(
                "opacity-sampling species must be unique and non-empty"
            )
        n_layers = self.pressure_center_bar.shape[0]
        if self.pressure_edge_bar.shape != (n_layers + 1,):
            raise RobertValidationError("pressure centers and edges are inconsistent")
        if self.molecular_weights_amu.shape != (len(self.species),):
            raise RobertValidationError(
                "one molecular weight is required for each opacity species"
            )
        if (
            len(self.emission_path_factors.shape) != 2
            or self.emission_path_factors.shape[1] != n_layers
            or self.emission_point_weights.shape
            != (self.emission_path_factors.shape[0],)
        ):
            raise RobertValidationError(
                "emission path factors and weights are inconsistent"
            )
        if self.likelihood.projection.native_size != self.wavelength_micron.shape[0]:
            raise RobertValidationError(
                "likelihood projection columns must match sampled wavelengths"
            )
        positive = (
            self.gravity_m_s2,
            self.planet_radius_m,
            self.star_radius_m,
            self.star_temperature_k,
            self.background_mean_molecular_weight_amu,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
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
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

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
        jnp = self.runtime.jnp
        parameters = jnp.asarray(parameter_vector, dtype=jnp.float32)
        if parameters.shape != (self.n_parameters,):
            raise RobertValidationError(
                "parameter vector must contain PG14 and species log VMRs"
            )
        kappa, gamma1, gamma2, irradiation, alpha = parameters[:5]
        trace_vmr = jnp.power(jnp.float32(10.0), parameters[5:])
        trace_sum = jnp.sum(trace_vmr)
        mmw = (1.0 - trace_sum) * jnp.float32(self.background_mean_molecular_weight_amu)
        mmw = mmw + jnp.sum(trace_vmr * self.molecular_weights_amu)
        temperature = pg14_temperature_device(
            self.pressure_center_bar,
            kappa_ir_m2_kg=kappa,
            gamma1=gamma1,
            gamma2=gamma2,
            irradiation_temperature_k=irradiation,
            alpha=alpha,
            gravity_m_s2=jnp.float32(self.gravity_m_s2),
            internal_temperature_k=jnp.float32(self.internal_temperature_k),
        )
        level_temperature = pg14_temperature_device(
            self.pressure_edge_bar,
            kappa_ir_m2_kg=kappa,
            gamma1=gamma1,
            gamma2=gamma2,
            irradiation_temperature_k=irradiation,
            alpha=alpha,
            gravity_m_s2=jnp.float32(self.gravity_m_s2),
            internal_temperature_k=jnp.float32(self.internal_temperature_k),
        )
        n_layers = self.pressure_center_bar.shape[0]
        vmr = jnp.broadcast_to(trace_vmr[:, None], (len(self.species), n_layers))
        tau = assemble_opacity_sampling_log_optical_depth_device(
            self.opacity.interpolate_log(self.pressure_center_bar, temperature),
            vmr,
            self.pressure_edge_bar,
            jnp.full((n_layers,), mmw),
            jnp.float32(self.gravity_m_s2),
            opacity_unit_scale_m2=self.opacity.opacity_unit_scale_m2,
        )
        source = planck_radiance_wavelength_device(
            self.wavelength_micron, level_temperature
        )
        planet = integrate_clear_thermal_device(
            tau,
            source,
            jnp.ones((1,), dtype=jnp.float32),
            self.emission_path_factors,
            self.emission_point_weights,
            source[-1],
            jnp.ones((self.emission_path_factors.shape[0],), dtype=bool),
        )
        star = planck_radiance_wavelength_device(
            self.wavelength_micron, jnp.float32(self.star_temperature_k)
        )
        eclipse = (
            jnp.float32((self.planet_radius_m / self.star_radius_m) ** 2)
            * planet
            / star
        )
        valid = (
            jnp.all(jnp.isfinite(temperature))
            & jnp.all(temperature > 0.0)
            & jnp.all(trace_vmr > 0.0)
            & (trace_sum < 1.0)
            & (mmw > 0.0)
            & jnp.all(jnp.isfinite(eclipse))
        )
        return jnp.where(valid, eclipse, jnp.nan)

    def loglike(self, parameter_vector: Any):
        return self.likelihood.loglike(self.native_eclipse_depth(parameter_vector))

    def retrieval_problem(
        self, *, name: str, parameters: RetrievalParameterSet
    ) -> MetalRetrievalProblem:
        """Bind the sampled graph to ROBERT's sampler-facing contract."""

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
            metadata={
                **dict(self.metadata),
                "model": "cloud-free-emission",
                "opacity_mode": "opacity-sampling",
                "temperature": "parmentier-guillot-2014",
                "chemistry": "free-trace-plus-h2-he-background",
                "species": ",".join(self.species),
            },
            opacity_identifiers=self.opacity_identifiers,
        )


def prepare_opacity_sampling_cloud_free_emission_from_prepared(
    *,
    provider: Any,
    prepared: Any,
    runtime: Any,
    projection_matrix: Any | None = None,
    projection: MetalLinearProjection
    | MetalIdentityProjection
    | MetalBinnedProjection
    | None = None,
    observation: Any,
    uncertainty: Any,
    gravity_m_s2: float,
    planet_radius_m: float,
    star_radius_m: float,
    star_temperature_k: float,
    include_likelihood_normalization: bool = False,
    disc_quadrature_points: int = 4,
    background_mean_molecular_weight_amu: float = 2.3,
    internal_temperature_k: float = 100.0,
    resource_policy: MetalResourcePolicy | None = None,
) -> PreparedMetalOpacitySamplingCloudFreeEmission:
    """Transfer an already loaded CPU prepared sampling table into JAX state."""

    species = tuple(prepared.species)
    if not species or len(set(species)) != len(species):
        raise RobertValidationError(
            "opacity-sampling species must be unique and non-empty"
        )
    if provider.interpolation not in {
        "log_pressure_temperature_log_xsec",
        "log_pressure_temperature_log_xsec_clip",
    }:
        raise RobertValidationError("unsupported opacity-sampling interpolation mode")
    if prepared.provider_name != provider.name:
        raise RobertValidationError(
            "prepared opacity-sampling provider identity does not match provider"
        )
    if str(prepared.metadata.get("interpolation", "")) != provider.interpolation:
        raise RobertValidationError(
            "prepared opacity-sampling interpolation does not match provider"
        )
    if (
        not isinstance(disc_quadrature_points, int)
        or isinstance(disc_quadrature_points, bool)
        or disc_quadrature_points < 1
    ):
        raise RobertValidationError("disc_quadrature_points must be a positive integer")
    reference = provider.tables[species[0]]
    opacity_identifiers: dict[str, str] = {}
    for name in species:
        if name not in provider.tables:
            raise RobertValidationError(
                f"opacity-sampling provider is missing species: {name}"
            )
        table = provider.tables[name]
        _exact_spectral_indices(prepared.spectral_grid, table)
        if (
            not np.array_equal(table.pressure_bar, reference.pressure_bar)
            or not np.array_equal(table.temperature_K, reference.temperature_K)
            or not np.array_equal(
                table.wavenumber_cm_inverse, reference.wavenumber_cm_inverse
            )
            or table.unit != reference.unit
        ):
            raise RobertValidationError(
                "Metal opacity-sampling species must share all grids and units"
            )
        for key in ("source_path", "checksum_sha256", "doi", "line_list"):
            value = str(table.metadata.get(key, ""))
            if value:
                opacity_identifiers[f"{name}:{key}"] = value
    # ``PreparedOpacitySampling`` intentionally stores logarithms, so retain
    # those values verbatim rather than taking a logarithm a second time.
    log_values = np.asarray(prepared.stacked_log_cross_sections, dtype=np.float64)
    if log_values.ndim != 4 or not np.all(np.isfinite(log_values)):
        raise RobertValidationError("prepared sampled log cross sections are invalid")
    if log_values.shape[0] != len(species):
        raise RobertValidationError(
            "prepared sampled tables must contain one array per species"
        )
    centers_bar = pressure_values_in_unit(
        prepared.pressure_grid.centers,
        prepared.pressure_grid.unit,
        "bar",
    )
    edges_bar = pressure_values_in_unit(
        prepared.pressure_grid.edges,
        prepared.pressure_grid.unit,
        "bar",
    )
    if prepared.pressure_grid.orientation != "increasing":
        raise RobertValidationError(
            "JAX opacity-sampling currently requires top-to-bottom increasing pressure"
        )
    wavelength_micron = spectral_grid_values_in_unit(prepared.spectral_grid, "micron")
    if (projection_matrix is None) == (projection is None):
        raise RobertValidationError(
            "provide exactly one of projection_matrix or projection"
        )
    projection_array = None
    projection_bytes = 0
    if projection is None:
        projection_array = np.asarray(projection_matrix, dtype=np.float32)
        if projection_array.ndim != 2:
            raise RobertValidationError("projection matrix must be two-dimensional")
        projection_bytes = projection_array.nbytes
    else:
        projection_bytes = projection.device_bytes
        if projection.native_size != log_values.shape[3]:
            raise RobertValidationError(
                "projection native size must match sampled wavelengths"
            )
    policy = (
        MetalResourcePolicy.laptop_safe()
        if resource_policy is None
        else resource_policy
    )
    policy.require_safe_batch("opacity-sampling likelihood", 1)
    estimated_peak_bytes = policy.estimate_opacity_sampling_bytes(
        species=len(species),
        pressure_grid=log_values.shape[1],
        temperature_grid=log_values.shape[2],
        layers=centers_bar.size,
        spectral=log_values.shape[3],
        disc_points=disc_quadrature_points,
    )
    estimated_peak_bytes += projection_bytes * 3
    policy.require_safe(
        "opacity-sampling graph",
        estimated_peak_bytes,
    )
    opacity = PreparedMetalOpacitySampling(
        runtime.put(log_values),
        runtime.put(np.log(reference.pressure_bar)),
        runtime.put(reference.temperature_K),
        _opacity_unit_scale_m2(reference.unit),
        provider.interpolation.endswith("_clip"),
    )
    if projection is None:
        projection = MetalLinearProjection.from_matrix(projection_array, runtime)
    likelihood = MetalLinearGaussianLikelihood.from_arrays(
        observation,
        uncertainty,
        projection,
        runtime,
        include_normalization=include_likelihood_normalization,
    )
    centers = np.asarray(centers_bar, dtype=np.float32)
    edges = np.asarray(edges_bar, dtype=np.float32)
    if centers.ndim != 1 or edges.shape != (centers.size + 1,):
        raise RobertValidationError("pressure centers/edges have incompatible shapes")
    geometry = gauss_legendre_disk_geometry(disc_quadrature_points)
    path_factors = np.broadcast_to(
        1.0 / geometry.emission_angle_cosines[:, None],
        (geometry.n_points, centers.size),
    ).copy()
    return PreparedMetalOpacitySamplingCloudFreeEmission(
        runtime=runtime,
        opacity=opacity,
        species=species,
        molecular_weights_amu=runtime.put(molecular_weights_for_species(species)),
        pressure_center_bar=runtime.put(centers),
        pressure_edge_bar=runtime.put(edges),
        wavelength_micron=runtime.put(wavelength_micron),
        gravity_m_s2=float(gravity_m_s2),
        planet_radius_m=float(planet_radius_m),
        star_radius_m=float(star_radius_m),
        star_temperature_k=float(star_temperature_k),
        emission_path_factors=runtime.put(path_factors),
        emission_point_weights=runtime.put(geometry.emission_angle_weights),
        likelihood=likelihood,
        opacity_identifiers=opacity_identifiers,
        metadata={
            "prepared_cache_key": str(prepared.cache_key),
            "opacity_provider": str(provider.name),
            "opacity_interpolation": str(provider.interpolation),
            "native_wavelength_count": str(wavelength_micron.size),
            "wavelength_min_micron": repr(float(np.min(wavelength_micron))),
            "wavelength_max_micron": repr(float(np.max(wavelength_micron))),
            "pressure_min_bar": repr(float(np.min(centers))),
            "pressure_max_bar": repr(float(np.max(centers))),
            "disc_quadrature": "gauss_legendre_mu",
            "disc_quadrature_points": str(disc_quadrature_points),
            "internal_temperature_k": repr(float(internal_temperature_k)),
            "gravity_m_s2": repr(float(gravity_m_s2)),
            "planet_radius_m": repr(float(planet_radius_m)),
            "star_radius_m": repr(float(star_radius_m)),
            "star_temperature_k": repr(float(star_temperature_k)),
            "likelihood_projection": type(projection).__name__,
            "likelihood_output_size": str(projection.output_size),
        },
        background_mean_molecular_weight_amu=float(
            background_mean_molecular_weight_amu
        ),
        internal_temperature_k=float(internal_temperature_k),
    )


__all__ = [
    "PreparedMetalOpacitySampling",
    "PreparedMetalOpacitySamplingCloudFreeEmission",
    "assemble_opacity_sampling_optical_depth_device",
    "assemble_opacity_sampling_log_optical_depth_device",
    "interpolate_opacity_sampling_device",
    "prepare_opacity_sampling_cloud_free_emission_from_prepared",
]
