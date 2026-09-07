"""CPU/JAX parity contracts for the separate opacity-sampling device path."""

from __future__ import annotations

import os

import numpy as np
import pytest

from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.atmosphere.temperature import _parmentier_guillot_eta
from robert_exoplanets.core import PressureGrid
from robert_exoplanets.metal import (
    MetalBinnedProjection,
    MetalIdentityProjection,
    MetalRuntime,
    PreparedMetalOpacitySampling,
    assemble_opacity_sampling_log_optical_depth_device,
    assemble_opacity_sampling_optical_depth_device,
    prepare_opacity_sampling_cloud_free_emission_from_prepared,
)
from robert_exoplanets.opacity import OpacitySamplingProvider
from robert_exoplanets.retrieval import (
    RetrievalParameter,
    RetrievalParameterSet,
    UniformPrior,
)
from robert_exoplanets.retrieval.manifest import build_run_manifest
from robert_exoplanets.rt import (
    assemble_opacity_sampling_gas_optical_depth,
    gauss_legendre_disk_geometry,
)
from robert_exoplanets.rt.emission import solve_emission_spectrum


def _runtime() -> MetalRuntime:
    if os.environ.get("ROBERT_RUN_JAX_CPU_TESTS") != "1":
        pytest.skip("set ROBERT_RUN_JAX_CPU_TESTS=1 for opacity-sampling JAX parity")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    return MetalRuntime(jax=jax, jnp=jnp, device=jax.devices("cpu")[0])


def test_sampled_opacity_interpolation_and_direct_tau_match_cpu(tmp_path) -> None:
    runtime = _runtime()
    provider = OpacitySamplingProvider.from_exomol_paths(
        {
            "H2O": _write_fixture(tmp_path / "H2O.h5", 1.0),
            "CO": _write_fixture(tmp_path / "CO.h5", 2.5),
        },
        checksum=False,
    )
    grid = provider.native_spectral_grid()
    pressure_grid = PressureGrid.logspace(1.0, 10.0, 3)
    prepared = provider.prepare(grid, pressure_grid, ("H2O", "CO"))
    vmr = np.array([1.0e-4, 2.0e-4])
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([610.0, 750.0, 920.0]),
        composition={
            name: np.full(3, value)
            for name, value in zip(prepared.species, vmr, strict=True)
        },
        mean_molecular_weight=np.full(3, 2.3),
    )
    cpu = assemble_opacity_sampling_gas_optical_depth(
        atmosphere, provider, prepared, gravity_m_s2=10.0
    ).total_tau
    device = PreparedMetalOpacitySampling(
        runtime.put(prepared.stacked_log_cross_sections),
        runtime.put(np.log(provider.tables["H2O"].pressure_bar)),
        runtime.put(provider.tables["H2O"].temperature_K),
        1.0e-4,
    )
    actual = assemble_opacity_sampling_optical_depth_device(
        device.interpolate(pressure_grid.centers, atmosphere.temperature),
        runtime.put(np.broadcast_to(vmr[:, None], (2, 3))),
        pressure_grid.edges,
        np.full(3, 2.3),
        10.0,
        opacity_unit_scale_m2=device.opacity_unit_scale_m2,
    )
    actual.block_until_ready()
    np.testing.assert_allclose(np.asarray(actual), cpu, rtol=5.0e-6, atol=1.0e-8)

    # Each retrieved gas must measurably affect the direct sampled optical depth.
    for index in range(vmr.size):
        changed = vmr.copy()
        changed[index] *= 10.0
        perturbed = assemble_opacity_sampling_optical_depth_device(
            device.interpolate(pressure_grid.centers, atmosphere.temperature),
            runtime.put(np.broadcast_to(changed[:, None], (2, 3))),
            pressure_grid.edges,
            np.full(3, 2.3),
            10.0,
            opacity_unit_scale_m2=device.opacity_unit_scale_m2,
        )
        perturbed.block_until_ready()
        assert np.max(np.abs(np.asarray(perturbed) - np.asarray(actual))) > 0.0


def test_complete_sampled_pg14_disk_graph_matches_cpu_and_manifest(tmp_path) -> None:
    runtime = _runtime()
    provider = OpacitySamplingProvider.from_exomol_paths(
        {
            "H2O": _write_fixture(tmp_path / "H2O.h5", 1.0),
            "CO": _write_fixture(tmp_path / "CO.h5", 2.5),
        },
        interpolation="log_pressure_temperature_log_xsec_clip",
        checksum=False,
    )
    spectral_grid = provider.native_spectral_grid()
    pressure_grid = PressureGrid.logspace(1.0, 10.0, 3)
    prepared = provider.prepare(spectral_grid, pressure_grid, ("H2O", "CO"))
    geometry = gauss_legendre_disk_geometry(4)
    parameters = np.array([0.003, 0.2, 1.5, 800.0, 0.4, -4.0, -4.5])

    def cpu_spectrum(values: np.ndarray) -> np.ndarray:
        temperature = _pg14(pressure_grid.centers, values)
        temperature_edges = _pg14(pressure_grid.edges, values)
        trace = 10.0 ** values[5:]
        molecular_weights = np.array([18.01528, 28.0101])
        mmw = (1.0 - trace.sum()) * 2.3 + trace @ molecular_weights
        atmosphere = AtmosphereState(
            pressure_grid=pressure_grid,
            temperature=temperature,
            temperature_edges=temperature_edges,
            composition={
                name: np.full(pressure_grid.n_layers, abundance)
                for name, abundance in zip(prepared.species, trace, strict=True)
            },
            mean_molecular_weight=np.full(pressure_grid.n_layers, mmw),
        )
        tau = assemble_opacity_sampling_gas_optical_depth(
            atmosphere, provider, prepared, gravity_m_s2=10.0
        )
        return np.asarray(
            solve_emission_spectrum(
                tau,
                geometry=geometry,
                planet_radius_m=7.1492e7,
                star_radius_m=6.957e8,
                star_temperature_k=5000.0,
                thermal_integration_backend="numpy",
            ).values
        )

    expected = cpu_spectrum(parameters)
    model = prepare_opacity_sampling_cloud_free_emission_from_prepared(
        provider=provider,
        prepared=prepared,
        runtime=runtime,
        projection=MetalIdentityProjection(spectral_grid.size),
        observation=expected,
        uncertainty=np.maximum(np.abs(expected) * 0.01, 1.0e-12),
        gravity_m_s2=10.0,
        planet_radius_m=7.1492e7,
        star_radius_m=6.957e8,
        star_temperature_k=5000.0,
    )
    actual = np.asarray(model.native_eclipse_depth(runtime.put(parameters)))
    np.testing.assert_allclose(actual, expected, rtol=3.0e-4, atol=1.0e-9)
    assert abs(float(model.loglike(runtime.put(parameters)))) < 1.0e-3

    for gas_index in range(2):
        perturbed = parameters.copy()
        perturbed[5 + gas_index] += 0.5
        expected_perturbed = cpu_spectrum(perturbed)
        actual_perturbed = np.asarray(
            model.native_eclipse_depth(runtime.put(perturbed))
        )
        expected_response = expected_perturbed - expected
        actual_response = actual_perturbed - actual
        assert np.max(np.abs(expected_response)) > 0.0
        np.testing.assert_allclose(
            actual_perturbed, expected_perturbed, rtol=3.0e-4, atol=1.0e-9
        )
        np.testing.assert_allclose(
            actual_response, expected_response, rtol=3.0e-3, atol=2.0e-9
        )

    invalid = parameters.copy()
    invalid[5:] = 0.0
    assert np.isneginf(float(model.loglike(runtime.put(invalid))))

    retrieval_parameters = RetrievalParameterSet(
        tuple(
            RetrievalParameter(name, UniformPrior(lower, upper))
            for name, lower, upper in zip(
                model.parameter_names,
                [1.0e-5, 0.01, 0.01, 300.0, 0.0, -12.0, -12.0],
                [0.1, 10.0, 10.0, 3000.0, 1.0, -0.1, -0.1],
                strict=True,
            )
        )
    )
    problem = model.retrieval_problem(
        name="sampled-manifest", parameters=retrieval_parameters
    )
    manifest = build_run_manifest(
        problem, method="multinest", settings={}, random_seed=7
    )
    assert manifest.likelihood["name"] == "metal-independent-gaussian"
    assert "prepared_cache_key" in manifest.problem_metadata
    assert any(key.endswith(":source_path") for key in manifest.opacity_identifiers)


def test_compact_binned_projection_matches_dense_average() -> None:
    runtime = _runtime()
    values = np.arange(10.0, dtype=np.float32)
    edges = np.array([0, 3, 7, 10])
    projection = MetalBinnedProjection.from_edges(edges, values.size, runtime)
    expected = np.array([values[:3].mean(), values[3:7].mean(), values[7:].mean()])
    actual = np.asarray(projection.project(runtime.put(values)))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-6)
    assert projection.device_bytes < 100


def test_log_tau_preserves_sub_float32_cross_section_after_column_scaling() -> None:
    runtime = _runtime()
    log_xsec = np.full((1, 1, 1), np.log(1.0e-50))
    tau = assemble_opacity_sampling_log_optical_depth_device(
        runtime.put(log_xsec),
        runtime.put([[[1.0e-3]]]).reshape((1, 1)),
        runtime.put([1.0, 2.0]),
        runtime.put([2.3]),
        10.0,
        opacity_unit_scale_m2=1.0e-4,
    )
    value = float(np.asarray(tau)[0, 0, 0])
    expected = 1.0e-50 * 1.0e-3 * 1.0e5 / (2.3 * 1.66053906660e-27 * 10.0) * 1.0e-4
    assert value == pytest.approx(expected, rel=1.0e-5)


def _pg14(pressure_bar: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    kappa, gamma1, gamma2, irradiation, alpha = parameters[:5]
    tau = kappa * np.asarray(pressure_bar) * 1.0e5 / 10.0
    fourth = 0.75 * 100.0**4 * (2.0 / 3.0 + tau)
    fourth += (
        0.75
        * irradiation**4
        * (
            (1.0 - alpha) * _parmentier_guillot_eta(gamma1, tau)
            + alpha * _parmentier_guillot_eta(gamma2, tau)
        )
    )
    return fourth**0.25


def _write_fixture(path, scale: float):
    h5py = pytest.importorskip("h5py")
    pressure = np.array([1.0, 3.0, 10.0])
    temperature = np.array([500.0, 800.0, 1000.0])
    wavenumber = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    xsec = np.empty((pressure.size, temperature.size, wavenumber.size))
    for p, value_p in enumerate(pressure):
        for t, value_t in enumerate(temperature):
            xsec[p, t] = (
                scale
                * value_p
                * np.exp(value_t / 1000.0)
                * np.arange(1.0, 5.0)
                * 1.0e-25
            )
    with h5py.File(path, "w") as handle:
        p = handle.create_dataset("p", data=pressure)
        p.attrs["units"] = "bar"
        t = handle.create_dataset("t", data=temperature)
        t.attrs["units"] = "kelvin"
        w = handle.create_dataset("bin_edges", data=wavenumber)
        w.attrs["units"] = "wavenumbers"
        d = handle.create_dataset("xsecarr", data=xsec)
        d.attrs["units"] = "cm^2/molecule"
    return path
