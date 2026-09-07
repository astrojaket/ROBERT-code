"""Experimental Apple-Metal compatible numerical kernels.

The modules in this package deliberately do not replace the CPU reference
implementations. They use real-valued float32 JAX operations so the same
kernels can be compiled by JAX Metal, whose complex and float64 support is
currently incomplete.
"""

from .likelihood import (
    MetalBinnedProjection,
    MetalIdentityProjection,
    MetalLinearGaussianLikelihood,
    MetalLinearProjection,
    MetalMultiDatasetGaussianLikelihood,
)
from .forward import (
    PreparedMetalCorrelatedK,
    assemble_gas_optical_depth_device,
    compile_metal_batched_likelihood,
    compile_metal_likelihood,
    exponential_integral_e2_device,
    integrate_clear_thermal_device,
    interpolate_correlated_k_device,
    pg14_temperature_device,
    planck_radiance_wavelength_device,
)
from .cloud_free import (
    PreparedMetalCloudFreeEmission,
    molecular_weights_for_species,
    prepare_cloud_free_emission_from_provider,
)
from .opacity_sampling import (
    PreparedMetalOpacitySampling,
    PreparedMetalOpacitySamplingCloudFreeEmission,
    assemble_opacity_sampling_log_optical_depth_device,
    assemble_opacity_sampling_optical_depth_device,
    interpolate_opacity_sampling_device,
    prepare_opacity_sampling_cloud_free_emission_from_prepared,
)
from .mie import (
    cloud_extinction_tau,
    cloud_extinction_tau_device,
    lognormal_mie_optics,
    lognormal_mie_optics_device,
    mie_efficiencies_and_moments,
    mie_efficiencies_and_moments_device,
)
from .random_overlap import metal_random_overlap_species_tau
from .resources import MetalResourcePolicy, physical_memory_bytes
from .runtime import (
    JaxAcceleratorRuntime,
    MetalRuntime,
    clear_metal_caches,
    require_accelerator_runtime,
    require_metal_runtime,
)
from .problem import MetalRetrievalProblem
from .sh4 import metal_solve_thermal_sh4_spectrum

__all__ = [
    "MetalLinearGaussianLikelihood",
    "MetalIdentityProjection",
    "MetalBinnedProjection",
    "MetalLinearProjection",
    "MetalMultiDatasetGaussianLikelihood",
    "MetalRuntime",
    "JaxAcceleratorRuntime",
    "MetalResourcePolicy",
    "MetalRetrievalProblem",
    "PreparedMetalCorrelatedK",
    "PreparedMetalCloudFreeEmission",
    "PreparedMetalOpacitySampling",
    "PreparedMetalOpacitySamplingCloudFreeEmission",
    "assemble_gas_optical_depth_device",
    "assemble_opacity_sampling_optical_depth_device",
    "assemble_opacity_sampling_log_optical_depth_device",
    "cloud_extinction_tau",
    "cloud_extinction_tau_device",
    "clear_metal_caches",
    "compile_metal_likelihood",
    "compile_metal_batched_likelihood",
    "exponential_integral_e2_device",
    "integrate_clear_thermal_device",
    "interpolate_correlated_k_device",
    "interpolate_opacity_sampling_device",
    "lognormal_mie_optics",
    "lognormal_mie_optics_device",
    "metal_random_overlap_species_tau",
    "mie_efficiencies_and_moments",
    "mie_efficiencies_and_moments_device",
    "metal_solve_thermal_sh4_spectrum",
    "molecular_weights_for_species",
    "pg14_temperature_device",
    "physical_memory_bytes",
    "prepare_cloud_free_emission_from_provider",
    "prepare_opacity_sampling_cloud_free_emission_from_prepared",
    "planck_radiance_wavelength_device",
    "require_metal_runtime",
    "require_accelerator_runtime",
]
