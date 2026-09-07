"""Scientific parity for the device-native atmosphere/opacity/RT graph."""

from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.special import expn

from robert_exoplanets.metal import (
    MetalLinearGaussianLikelihood,
    MetalLinearProjection,
    MetalRuntime,
    PreparedMetalCorrelatedK,
    assemble_gas_optical_depth_device,
    compile_metal_likelihood,
    exponential_integral_e2_device,
    integrate_clear_thermal_device,
    pg14_temperature_device,
    planck_radiance_wavelength_device,
)
from robert_exoplanets.rt.thermal_integration import (
    integrate_thermal_emission_spectrum,
)


def _cpu_runtime() -> MetalRuntime:
    if os.environ.get("ROBERT_RUN_JAX_CPU_TESTS") != "1":
        pytest.skip("set ROBERT_RUN_JAX_CPU_TESTS=1 for optional JAX CPU parity")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    return MetalRuntime(jax=jax, jnp=jnp, device=jax.devices("cpu")[0])


def test_device_e2_and_pg14_match_float64_scientific_reference() -> None:
    runtime = _cpu_runtime()
    arguments = np.logspace(-6.0, 6.0, 97)
    actual_e2 = np.asarray(exponential_integral_e2_device(runtime.put(arguments)))
    np.testing.assert_allclose(actual_e2, expn(2, arguments), rtol=2.0e-5, atol=2.0e-7)

    pressure = np.logspace(-6.0, 2.0, 80)
    kappa = 0.00316
    gamma1 = 0.12
    gamma2 = 1.7
    irradiation = 1510.0
    alpha = 0.38
    gravity = 5.7
    internal = 100.0
    tau = kappa * pressure * 1.0e5 / gravity

    def eta(gamma: float) -> np.ndarray:
        argument = gamma * tau
        return (
            2.0 / 3.0
            + 2.0 / (3.0 * gamma) * (1.0 + (argument / 2.0 - 1.0) * np.exp(-argument))
            + (2.0 * gamma / 3.0) * (1.0 - 0.5 * tau**2) * expn(2, argument)
        )

    fourth = 0.75 * internal**4 * (2.0 / 3.0 + tau)
    fourth += (
        0.75 * irradiation**4 * ((1.0 - alpha) * eta(gamma1) + alpha * eta(gamma2))
    )
    expected = fourth**0.25
    actual = pg14_temperature_device(
        runtime.put(pressure),
        kappa_ir_m2_kg=kappa,
        gamma1=gamma1,
        gamma2=gamma2,
        irradiation_temperature_k=irradiation,
        alpha=alpha,
        gravity_m_s2=gravity,
        internal_temperature_k=internal,
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=3.0e-5, atol=0.03)


def test_prepared_k_interpolation_and_hydrostatic_tau_match_reference() -> None:
    runtime = _cpu_runtime()
    pressure_grid = np.array([1.0e-5, 1.0e-2, 10.0])
    temperature_grid = np.array([500.0, 1000.0, 1800.0])
    native = 5
    g = 3
    species = 2
    shape = (species, pressure_grid.size, temperature_grid.size, native, g)
    indices = np.indices(shape)
    log_k = (
        -45.0
        + 0.7 * indices[0]
        + 0.4 * indices[1]
        + 0.002 * indices[2]
        + 0.03 * indices[3]
        + 0.1 * indices[4]
    )
    kcoeff = np.exp(log_k)
    selected = np.array([[0, 2, 4], [0, 2, 4]])
    weights = np.array([0.2, 0.3, 0.5])
    prepared = PreparedMetalCorrelatedK.from_arrays(
        kcoeff,
        pressure_grid,
        temperature_grid,
        selected,
        weights,
        runtime,
    )
    pressure = np.array([3.0e-5, 0.2])
    temperature = np.array([720.0, 1400.0])
    actual_k = np.asarray(
        prepared.interpolate(runtime.put(pressure), runtime.put(temperature))
    )

    expected_k = np.empty((species, 2, 3, g))
    for layer in range(2):
        lp = np.log10(pressure[layer])
        p1 = np.searchsorted(np.log10(pressure_grid), lp, side="right")
        p1 = np.clip(p1, 1, pressure_grid.size - 1)
        p0 = p1 - 1
        t1 = np.searchsorted(temperature_grid, temperature[layer], side="right")
        t1 = np.clip(t1, 1, temperature_grid.size - 1)
        t0 = t1 - 1
        wp = (lp - np.log10(pressure_grid[p0])) / np.log10(
            pressure_grid[p1] / pressure_grid[p0]
        )
        wt = (temperature[layer] - temperature_grid[t0]) / (
            temperature_grid[t1] - temperature_grid[t0]
        )
        for item in range(species):
            values = (
                (1.0 - wp) * (1.0 - wt) * log_k[item, p0, t0, selected[item]]
                + wp * (1.0 - wt) * log_k[item, p1, t0, selected[item]]
                + (1.0 - wp) * wt * log_k[item, p0, t1, selected[item]]
                + wp * wt * log_k[item, p1, t1, selected[item]]
            )
            expected_k[item, layer] = np.exp(values)
    np.testing.assert_allclose(actual_k, expected_k, rtol=8.0e-6, atol=0.0)

    vmr = np.array([[1.0e-3, 2.0e-3], [2.0e-4, 3.0e-4]])
    edges = np.array([1.0e-5, 1.0e-3, 1.0])
    mmw = np.array([2.3, 2.4])
    gravity = 6.2
    actual_tau = assemble_gas_optical_depth_device(
        prepared.interpolate(runtime.put(pressure), runtime.put(temperature)),
        runtime.put(vmr),
        runtime.put(edges),
        runtime.put(mmw),
        gravity,
        prepared.g_weights,
        opacity_unit_scale_m2=prepared.opacity_unit_scale_m2,
        random_overlap=False,
    )
    column = np.diff(edges) * 1.0e5 / (mmw * 1.66053906660e-27 * gravity)
    expected_tau = np.einsum("slwg,sl->lwg", expected_k * 1.0e-4, vmr * column[None, :])
    np.testing.assert_allclose(
        np.asarray(actual_tau), expected_tau, rtol=1.5e-5, atol=0.0
    )


def test_k_interpolation_has_explicit_strict_and_clip_coverage_modes() -> None:
    runtime = _cpu_runtime()
    # log(k) corners are [[1, 3], [2, 4]] over pressure and temperature.
    log_k = np.array([[[1.0], [3.0]], [[2.0], [4.0]]])
    kcoeff = np.exp(log_k)[None, :, :, :, None]
    strict = PreparedMetalCorrelatedK.from_arrays(
        kcoeff,
        [1.0e-2, 1.0],
        [500.0, 1000.0],
        [[0]],
        [1.0],
        runtime,
        clip=False,
    )
    clipped = PreparedMetalCorrelatedK.from_arrays(
        kcoeff,
        [1.0e-2, 1.0],
        [500.0, 1000.0],
        [[0]],
        [1.0],
        runtime,
        clip=True,
    )
    out_of_range = strict.interpolate(runtime.put([10.0]), runtime.put([1500.0]))
    assert np.isnan(np.asarray(out_of_range)).all()
    clipped_value = clipped.interpolate(runtime.put([10.0]), runtime.put([1500.0]))
    assert float(np.asarray(clipped_value).reshape(-1)[0]) == pytest.approx(
        np.exp(4.0), rel=2.0e-6
    )


def test_clear_thermal_device_matches_cpu_linear_source() -> None:
    runtime = _cpu_runtime()
    rng = np.random.default_rng(41)
    tau = np.exp(rng.uniform(-8.0, 1.0, size=(4, 6, 3)))
    levels = rng.uniform(1.0, 9.0, size=(5, 6))
    weights = np.array([0.2, 0.3, 0.5])
    paths = np.array([[1.0, 1.0, 1.0, 1.0], [1.4, 1.2, 1.1, 1.05]])
    point_weights = np.array([0.4, 0.6])
    bottom = rng.uniform(5.0, 10.0, size=6)
    visible = np.array([True, False])
    expected = integrate_thermal_emission_spectrum(
        tau,
        levels[:-1],
        weights,
        paths,
        point_weights,
        level_source_ordered=levels,
        bottom_source=bottom,
        bottom_visible=visible,
        backend="numpy",
    ).radiance
    actual = integrate_clear_thermal_device(
        runtime.put(tau),
        runtime.put(levels),
        runtime.put(weights),
        runtime.put(paths),
        runtime.put(point_weights),
        runtime.put(bottom),
        runtime.jax.device_put(visible, runtime.device),
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2.0e-6, atol=2.0e-6)


def test_complete_parameter_to_likelihood_graph_compiles_as_one_jit() -> None:
    runtime = _cpu_runtime()
    pressure_grid = np.array([1.0e-5, 1.0e-2, 1.0])
    temperature_grid = np.array([500.0, 1000.0, 1800.0])
    kcoeff = np.full((1, 3, 3, 2, 2), 2.0e-24)
    prepared = PreparedMetalCorrelatedK.from_arrays(
        kcoeff,
        pressure_grid,
        temperature_grid,
        [[0, 1]],
        [0.4, 0.6],
        runtime,
    )
    projection = MetalLinearProjection.from_matrix(np.eye(2), runtime)
    likelihood = MetalLinearGaussianLikelihood.from_arrays(
        [1.0e8, 8.0e7], [1.0e7, 1.0e7], projection, runtime
    )
    pressure = runtime.put([1.0e-3, 0.1])
    edges = runtime.put([1.0e-5, 1.0e-2, 1.0])
    wavelengths = runtime.put([3.0, 5.0])
    paths = runtime.put([[1.0, 1.0]])
    point_weights = runtime.put([1.0])

    def full_graph(parameters):
        temperature = parameters[:2]
        k_values = prepared.interpolate(pressure, temperature)
        tau = assemble_gas_optical_depth_device(
            k_values,
            runtime.jnp.full((1, 2), runtime.jnp.float32(1.0e-3)),
            edges,
            runtime.jnp.full((2,), runtime.jnp.float32(2.3)),
            runtime.jnp.float32(6.0),
            prepared.g_weights,
            opacity_unit_scale_m2=prepared.opacity_unit_scale_m2,
            random_overlap=True,
        )
        level_temperature = runtime.jnp.array(
            [temperature[0], 0.5 * (temperature[0] + temperature[1]), temperature[1]]
        )
        level_source = planck_radiance_wavelength_device(wavelengths, level_temperature)
        radiance = integrate_clear_thermal_device(
            tau,
            level_source,
            prepared.g_weights,
            paths,
            point_weights,
            level_source[-1],
            runtime.jnp.array([True]),
        )
        return likelihood.loglike(radiance)

    compiled = compile_metal_likelihood(full_graph, runtime)
    value = compiled(runtime.put([900.0, 1300.0]))
    value.block_until_ready()
    assert np.isfinite(float(value))
    jaxpr = str(runtime.jax.make_jaxpr(full_graph)(runtime.put([900.0, 1300.0])))
    assert "searchsorted" in jaxpr
    assert "expm1" in jaxpr
