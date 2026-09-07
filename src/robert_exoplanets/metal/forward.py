"""Device-native atmospheric, opacity, and clear-emission primitives.

These functions form a separate JAX graph.  They do not dispatch into the
NumPy/Numba reference implementation, and none of them transfers a result to
the host.  Static table preparation and scientific validation happen before
the graph is compiled.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np

from robert_exoplanets.core import RobertValidationError

from .random_overlap import metal_random_overlap_species_tau
from .resources import MetalResourcePolicy

ATOMIC_MASS_KG = 1.66053906660e-27
PLANCK_CONSTANT_J_S = 6.62607015e-34
LIGHT_SPEED_M_S = 299792458.0
BOLTZMANN_CONSTANT_J_K = 1.380649e-23
EULER_MASCHERONI = 0.5772156649015329


def _jax_modules():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RobertValidationError("Metal forward kernels require JAX") from exc
    return jax, jnp


def exponential_integral_e2_device(argument: Any):
    """Evaluate :math:`E_2(x)` for positive float32 device arrays.

    The implementation uses the convergent power series for ``E1`` at
    ``x <= 1`` and a modified-Lentz continued fraction above one.  Eighty
    fixed iterations make the routine traceable while converging beyond
    float32 precision over the PG14 parameter domain.
    """

    jax, jnp = _jax_modules()
    x = jnp.asarray(argument, dtype=jnp.float32)
    safe_x = jnp.maximum(x, jnp.finfo(jnp.float32).tiny)

    def series_body(index, carry):
        term, total = carry
        k = jnp.asarray(index, dtype=x.dtype)
        term = term * (-safe_x) / k
        total = total - term / k
        return term, total

    _, series_sum = jax.lax.fori_loop(
        1,
        81,
        series_body,
        (jnp.ones_like(safe_x), jnp.zeros_like(safe_x)),
    )
    e1_series = -jnp.asarray(EULER_MASCHERONI, x.dtype) - jnp.log(safe_x) + series_sum

    tiny = jnp.asarray(1.0e-30, x.dtype)
    b0 = safe_x + 1.0
    c0 = jnp.full_like(safe_x, 1.0) / tiny
    d0 = 1.0 / b0
    h0 = d0

    def fraction_body(index, carry):
        b, c, d, h = carry
        i = jnp.asarray(index, dtype=x.dtype)
        a = -(i * i)
        b = b + 2.0
        d = a * d + b
        d = jnp.where(jnp.abs(d) < tiny, tiny, d)
        c = b + a / c
        c = jnp.where(jnp.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        delta = c * d
        return b, c, d, h * delta

    _, _, _, fraction = jax.lax.fori_loop(1, 81, fraction_body, (b0, c0, d0, h0))
    e1_fraction = fraction * jnp.exp(-safe_x)
    e1 = jnp.where(safe_x <= 1.0, e1_series, e1_fraction)
    e2 = jnp.exp(-safe_x) - safe_x * e1
    return jnp.where(x == 0.0, jnp.ones_like(e2), jnp.maximum(e2, 0.0))


def pg14_temperature_device(
    pressure_bar: Any,
    *,
    kappa_ir_m2_kg: Any,
    gamma1: Any,
    gamma2: Any,
    irradiation_temperature_k: Any,
    alpha: Any,
    gravity_m_s2: Any,
    internal_temperature_k: Any,
):
    """Device-native Parmentier--Guillot 2014 temperature profile."""

    _, jnp = _jax_modules()
    pressure = jnp.asarray(pressure_bar, dtype=jnp.float32)
    kappa = jnp.asarray(kappa_ir_m2_kg, dtype=jnp.float32)
    first_gamma = jnp.asarray(gamma1, dtype=jnp.float32)
    second_gamma = jnp.asarray(gamma2, dtype=jnp.float32)
    irradiation = jnp.asarray(irradiation_temperature_k, dtype=jnp.float32)
    fraction = jnp.asarray(alpha, dtype=jnp.float32)
    gravity = jnp.asarray(gravity_m_s2, dtype=jnp.float32)
    internal = jnp.asarray(internal_temperature_k, dtype=jnp.float32)
    tau = kappa * pressure * jnp.float32(1.0e5) / gravity

    def eta(gamma):
        value = gamma * tau
        return (
            jnp.float32(2.0 / 3.0)
            + jnp.float32(2.0 / 3.0)
            / gamma
            * (1.0 + (value / 2.0 - 1.0) * jnp.exp(-value))
            + jnp.float32(2.0 / 3.0)
            * gamma
            * (1.0 - 0.5 * tau * tau)
            * exponential_integral_e2_device(value)
        )

    fourth = 0.75 * internal**4 * (jnp.float32(2.0 / 3.0) + tau)
    fourth = fourth + 0.75 * irradiation**4 * (
        (1.0 - fraction) * eta(first_gamma) + fraction * eta(second_gamma)
    )
    valid = (
        (kappa > 0.0)
        & (first_gamma > 0.0)
        & (second_gamma > 0.0)
        & (irradiation > 0.0)
        & (internal >= 0.0)
        & (gravity > 0.0)
        & (fraction >= 0.0)
        & (fraction <= 1.0)
        & jnp.all(jnp.isfinite(fourth))
        & jnp.all(fourth > 0.0)
    )
    return jnp.where(valid, fourth ** jnp.float32(0.25), jnp.nan)


def interpolate_correlated_k_device(
    log_kcoeff: Any,
    log10_pressure_grid_bar: Any,
    temperature_grid_k: Any,
    spectral_indices: Any,
    pressure_bar: Any,
    temperature_k: Any,
    *,
    clip: bool = False,
):
    """Bilinearly interpolate common-grid species tables in log(k).

    ``log_kcoeff`` has shape ``species, pressure, temperature, native, g`` and
    ``spectral_indices`` has shape ``species, model_spectral``.  Tables are
    prepared with a float32-representable floor; no physical coefficient is
    wavelength-interpolated.
    """

    jax, jnp = _jax_modules()
    table = jnp.asarray(log_kcoeff, dtype=jnp.float32)
    pressure_grid = jnp.asarray(log10_pressure_grid_bar, dtype=jnp.float32)
    temperature_grid = jnp.asarray(temperature_grid_k, dtype=jnp.float32)
    indices = jnp.asarray(spectral_indices, dtype=jnp.int32)
    pressure = jnp.log10(jnp.asarray(pressure_bar, dtype=jnp.float32))
    temperature = jnp.asarray(temperature_k, dtype=jnp.float32)
    in_bounds = (
        jnp.all(jnp.isfinite(pressure))
        & jnp.all(jnp.isfinite(temperature))
        & jnp.all(pressure >= pressure_grid[0])
        & jnp.all(pressure <= pressure_grid[-1])
        & jnp.all(temperature >= temperature_grid[0])
        & jnp.all(temperature <= temperature_grid[-1])
    )
    if clip:
        pressure = jnp.clip(pressure, pressure_grid[0], pressure_grid[-1])
        temperature = jnp.clip(temperature, temperature_grid[0], temperature_grid[-1])

    p1 = jnp.clip(
        jnp.searchsorted(pressure_grid, pressure, side="right"),
        1,
        pressure_grid.size - 1,
    )
    p0 = p1 - 1
    t1 = jnp.clip(
        jnp.searchsorted(temperature_grid, temperature, side="right"),
        1,
        temperature_grid.size - 1,
    )
    t0 = t1 - 1
    wp = (pressure - pressure_grid[p0]) / (pressure_grid[p1] - pressure_grid[p0])
    wt = (temperature - temperature_grid[t0]) / (
        temperature_grid[t1] - temperature_grid[t0]
    )

    def species_interpolation(species_table, species_indices):
        v00 = species_table[p0[:, None], t0[:, None], species_indices[None, :], :]
        v10 = species_table[p1[:, None], t0[:, None], species_indices[None, :], :]
        v01 = species_table[p0[:, None], t1[:, None], species_indices[None, :], :]
        v11 = species_table[p1[:, None], t1[:, None], species_indices[None, :], :]
        wp3 = wp[:, None, None]
        wt3 = wt[:, None, None]
        value = (
            (1.0 - wp3) * (1.0 - wt3) * v00
            + wp3 * (1.0 - wt3) * v10
            + (1.0 - wp3) * wt3 * v01
            + wp3 * wt3 * v11
        )
        return jnp.exp(value)

    result = jax.vmap(species_interpolation)(table, indices)
    return result if clip else jnp.where(in_bounds, result, jnp.nan)


def assemble_gas_optical_depth_device(
    kcoeff: Any,
    volume_mixing_ratio: Any,
    pressure_edge_bar: Any,
    mean_molecular_weight_amu: Any,
    gravity_m_s2: Any,
    g_weights: Any,
    *,
    opacity_unit_scale_m2: float,
    random_overlap: bool = True,
):
    """Assemble hydrostatic gas optical depth entirely on the device."""

    _, jnp = _jax_modules()
    coefficients = jnp.asarray(kcoeff, dtype=jnp.float32)
    vmr = jnp.asarray(volume_mixing_ratio, dtype=jnp.float32)
    edges = jnp.asarray(pressure_edge_bar, dtype=jnp.float32)
    mmw = jnp.asarray(mean_molecular_weight_amu, dtype=jnp.float32)
    gravity = jnp.asarray(gravity_m_s2, dtype=jnp.float32)
    delta_pressure_pa = jnp.abs(jnp.diff(edges)) * jnp.float32(1.0e5)
    column = delta_pressure_pa / (mmw * jnp.float32(ATOMIC_MASS_KG) * gravity)
    species_column = vmr * column[None, :]
    species_tau = (
        coefficients
        * species_column[:, :, None, None]
        * jnp.float32(opacity_unit_scale_m2)
    )
    if random_overlap:
        return metal_random_overlap_species_tau(species_tau, g_weights)
    return jnp.sum(species_tau, axis=0)


def planck_radiance_wavelength_device(wavelength_micron: Any, temperature_k: Any):
    """Spectral radiance per metre at wavelength, matching the CPU equation."""

    _, jnp = _jax_modules()
    wavelength_m = jnp.asarray(wavelength_micron, dtype=jnp.float32) * jnp.float32(
        1.0e-6
    )
    temperature = jnp.asarray(temperature_k, dtype=jnp.float32)
    exponent = jnp.float32(
        PLANCK_CONSTANT_J_S * LIGHT_SPEED_M_S / BOLTZMANN_CONSTANT_J_K
    ) / (temperature[..., None] * wavelength_m)
    numerator = (
        jnp.float32(2.0 * PLANCK_CONSTANT_J_S * LIGHT_SPEED_M_S**2) / wavelength_m**5
    )
    return numerator / jnp.expm1(exponent)


def integrate_clear_thermal_device(
    tau_top_to_bottom: Any,
    level_source_top_to_bottom: Any,
    g_weights: Any,
    emission_path_factors: Any,
    emission_point_weights: Any,
    bottom_source: Any,
    bottom_visible: Any,
):
    """Exact linear-source clear thermal integration matching the CPU kernel."""

    _, jnp = _jax_modules()
    tau = jnp.asarray(tau_top_to_bottom, dtype=jnp.float32)
    level_source = jnp.asarray(level_source_top_to_bottom, dtype=jnp.float32)
    weights = jnp.asarray(g_weights, dtype=jnp.float32)
    weights = weights / jnp.sum(weights)
    paths = jnp.asarray(emission_path_factors, dtype=jnp.float32)
    point_weights = jnp.asarray(emission_point_weights, dtype=jnp.float32)
    point_weights = point_weights / jnp.sum(point_weights)
    bottom = jnp.asarray(bottom_source, dtype=jnp.float32)
    visible = jnp.asarray(bottom_visible, dtype=bool)

    def one_point(path, is_visible):
        slant = tau * path[:, None, None]
        cumulative = jnp.concatenate(
            (jnp.zeros_like(slant[:1]), jnp.cumsum(slant[:-1], axis=0)), axis=0
        )
        escape = -jnp.expm1(-slant)
        polynomial = slant / 2.0 - slant**2 / 3.0 + slant**3 / 8.0 - slant**4 / 30.0
        general = (escape - slant * jnp.exp(-slant)) / jnp.where(
            slant != 0.0, slant, 1.0
        )
        linear_weight = jnp.where(jnp.abs(slant) < 1.0e-5, polynomial, general)
        emitted = (
            level_source[:-1, :, None] * escape
            + (level_source[1:, :, None] - level_source[:-1, :, None]) * linear_weight
        )
        radiance = jnp.sum(
            jnp.exp(-cumulative) * emitted * weights[None, None, :], axis=(0, 2)
        )
        transmitted_bottom = (
            jnp.sum(jnp.exp(-jnp.sum(slant, axis=0)) * weights[None, :], axis=-1)
            * bottom
        )
        return radiance + jnp.where(is_visible, transmitted_bottom, 0.0)

    points = _jax_modules()[0].vmap(one_point)(paths, visible)
    return jnp.sum(points * point_weights[:, None], axis=0)


@dataclass(frozen=True)
class PreparedMetalCorrelatedK:
    """One-time validated, device-resident common-grid k-table state."""

    log_kcoeff: Any
    log10_pressure_grid_bar: Any
    temperature_grid_k: Any
    spectral_indices: Any
    g_weights: Any
    opacity_unit_scale_m2: float
    clip: bool = False

    @classmethod
    def from_arrays(
        cls,
        kcoeff: Any,
        pressure_grid_bar: Any,
        temperature_grid_k: Any,
        spectral_indices: Any,
        g_weights: Any,
        runtime: Any,
        *,
        opacity_unit: str = "cm^2/molecule",
        clip: bool = False,
    ) -> "PreparedMetalCorrelatedK":
        values = np.asarray(kcoeff, dtype=np.float64)
        pressure = np.asarray(pressure_grid_bar, dtype=np.float64)
        temperature = np.asarray(temperature_grid_k, dtype=np.float64)
        indices = np.asarray(spectral_indices, dtype=np.int32)
        weights = np.asarray(g_weights, dtype=np.float64)
        if values.ndim != 5 or values.shape[1:3] != (pressure.size, temperature.size):
            raise RobertValidationError(
                "kcoeff must have species x pressure x temperature x spectral x g shape"
            )
        if indices.ndim != 2 or indices.shape[0] != values.shape[0]:
            raise RobertValidationError(
                "spectral_indices must have one row per species"
            )
        if weights.shape != (values.shape[-1],):
            raise RobertValidationError("g_weights must match the kcoeff g axis")
        if (
            np.any(values < 0.0)
            or not np.all(np.isfinite(values))
            or np.any(pressure <= 0.0)
            or np.any(np.diff(pressure) <= 0.0)
            or np.any(np.diff(temperature) <= 0.0)
            or np.any(indices < 0)
            or np.any(indices >= values.shape[3])
            or np.any(weights < 0.0)
            or not np.isfinite(weights).all()
            or weights.sum() <= 0.0
        ):
            raise RobertValidationError("invalid correlated-k preparation arrays")
        normalized_unit = opacity_unit.strip().lower().replace(" ", "")
        if normalized_unit in {
            "cm^2/molecule",
            "cm2/molecule",
            "cm^2molecule^-1",
            "cm2molecule-1",
        }:
            scale = 1.0e-4
        elif normalized_unit in {
            "m^2/molecule",
            "m2/molecule",
            "m^2molecule^-1",
            "m2molecule-1",
        }:
            scale = 1.0
        else:
            raise RobertValidationError(f"unsupported opacity unit: {opacity_unit}")
        floor = np.finfo(np.float32).tiny
        return cls(
            runtime.put(np.log(np.maximum(values, floor))),
            runtime.put(np.log10(pressure)),
            runtime.put(temperature),
            runtime.jax.device_put(indices, runtime.device),
            runtime.put(weights / weights.sum()),
            scale,
            bool(clip),
        )

    def interpolate(self, pressure_bar: Any, temperature_k: Any):
        return interpolate_correlated_k_device(
            self.log_kcoeff,
            self.log10_pressure_grid_bar,
            self.temperature_grid_k,
            self.spectral_indices,
            pressure_bar,
            temperature_k,
            clip=self.clip,
        )


@lru_cache(maxsize=32)
def _compile_outer(function: Callable[[Any], Any], device: Any):
    jax, _ = _jax_modules()
    # Device placement is inherited from the Metal-resident input and bound
    # constants. Passing ``device=`` is deprecated in current JAX.
    del device
    return jax.jit(function)


def compile_metal_likelihood(
    parameter_to_loglike: Callable[[Any], Any], runtime: Any
) -> Callable[[Any], Any]:
    """Compile one complete ``parameter vector -> scalar loglike`` graph."""

    return _compile_outer(parameter_to_loglike, runtime.device)


def compile_metal_batched_likelihood(
    parameter_to_loglike: Callable[[Any], Any],
    runtime: Any,
    *,
    batch_size: int,
    resource_policy: MetalResourcePolicy | None = None,
) -> Callable[[Any], Any]:
    """Compile a guarded proposal batch, refusing before ``jax.jit``.

    The laptop-safe policy currently permits only batch one.  This explicit
    entry point exists so callers cannot accidentally hide a batch dimension
    from primitive-level resource checks with a raw outer ``vmap``.
    """

    policy = (
        MetalResourcePolicy.laptop_safe()
        if resource_policy is None
        else resource_policy
    )
    policy.require_safe_batch("likelihood compilation", batch_size)
    return runtime.jax.jit(runtime.jax.vmap(parameter_to_loglike))


__all__ = [
    "PreparedMetalCorrelatedK",
    "assemble_gas_optical_depth_device",
    "compile_metal_likelihood",
    "compile_metal_batched_likelihood",
    "exponential_integral_e2_device",
    "integrate_clear_thermal_device",
    "interpolate_correlated_k_device",
    "pg14_temperature_device",
    "planck_radiance_wavelength_device",
]
