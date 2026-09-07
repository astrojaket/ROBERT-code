"""Parity contracts for the isolated experimental JAX Metal building blocks."""

from __future__ import annotations

import os
import numpy as np
import pytest

from robert_exoplanets.core import RobertConfigError, RobertValidationError
from robert_exoplanets.metal import (
    MetalLinearGaussianLikelihood,
    MetalLinearProjection,
    MetalMultiDatasetGaussianLikelihood,
    MetalRetrievalProblem,
    MetalRuntime,
    metal_random_overlap_species_tau,
)
from robert_exoplanets.rt.random_overlap import random_overlap_species_tau
from robert_exoplanets.retrieval import (
    RetrievalParameter,
    RetrievalParameterSet,
    UniformPrior,
)


def _cpu_runtime():
    """Exercise device-array math on JAX CPU; this is not a Metal fallback."""

    if os.environ.get("ROBERT_RUN_JAX_CPU_TESTS") != "1":
        pytest.skip("set ROBERT_RUN_JAX_CPU_TESTS=1 to execute optional JAX CPU parity")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", False)
    import jax.numpy as jnp

    return MetalRuntime(jax=jax, jnp=jnp, device=jax.devices("cpu")[0])


def test_device_resident_float32_rorr_matches_cpu_target_bin_definition() -> None:
    runtime = _cpu_runtime()
    rng = np.random.default_rng(101)
    tau = np.exp(rng.uniform(-9.0, 1.0, size=(3, 4, 5, 8)))
    weights = np.arange(1.0, 9.0)
    weights /= weights.sum()

    expected = random_overlap_species_tau(tau, weights, backend="numpy")
    actual = metal_random_overlap_species_tau(runtime.put(tau), runtime.put(weights))
    actual.block_until_ready()

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2.0e-5, atol=2.0e-6)


def test_device_projection_and_normalized_likelihood_match_float32_reference() -> None:
    runtime = _cpu_runtime()
    matrix = np.array([[0.75, 0.25, 0.0], [0.0, 0.4, 0.6]], dtype=np.float32)
    native = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    observation = np.array([1.7, 4.3], dtype=np.float32)
    uncertainty = np.array([0.2, 0.4], dtype=np.float32)
    projection = MetalLinearProjection.from_matrix(matrix, runtime)
    likelihood = MetalLinearGaussianLikelihood.from_arrays(
        observation,
        uncertainty,
        projection,
        runtime,
        mask=[True, False],
        include_normalization=True,
    )

    actual = likelihood.loglike(
        runtime.put(native), offset=np.float32(0.1), jitter=np.float32(0.05)
    )
    actual.block_until_ready()
    prediction = matrix @ native + 0.1
    variance = uncertainty**2 + 0.05**2
    expected = -0.5 * (
        (observation[0] - prediction[0]) ** 2 / variance[0]
        + np.log(2.0 * np.pi * variance[0])
    )
    assert float(actual) == pytest.approx(float(expected), rel=2.0e-6, abs=2.0e-6)


def test_device_likelihood_matches_unnormalized_scaled_cpu_equation() -> None:
    runtime = _cpu_runtime()
    projection = MetalLinearProjection.from_matrix(np.eye(2), runtime)
    likelihood = MetalLinearGaussianLikelihood.from_arrays(
        [1.0, 3.0],
        [0.2, 0.4],
        projection,
        runtime,
        uncertainty_scale=1.2,
    )
    actual = likelihood.loglike(
        runtime.put([0.8, 3.3]),
        offset=np.float32(0.05),
        jitter=np.float32(0.03),
        uncertainty_scale_parameter=np.float32(0.9),
    )
    variance = (np.array([0.2, 0.4]) * 1.2 * 0.9) ** 2 + 0.03**2
    residual = np.array([1.0, 3.0]) - (np.array([0.8, 3.3]) + 0.05)
    expected = -0.5 * np.sum(residual**2 / variance)
    assert float(actual) == pytest.approx(expected, rel=2.0e-6, abs=2.0e-6)


def test_device_likelihood_rejects_invalid_static_and_dynamic_inputs() -> None:
    runtime = _cpu_runtime()
    projection = MetalLinearProjection.from_matrix(np.eye(2), runtime)
    with pytest.raises(RobertValidationError, match="excludes all"):
        MetalLinearGaussianLikelihood.from_arrays(
            [1.0, 2.0], [0.1, 0.2], projection, runtime, mask=[False, False]
        )
    likelihood = MetalLinearGaussianLikelihood.from_arrays(
        [1.0, 2.0], [0.1, 0.2], projection, runtime
    )
    assert np.isneginf(float(likelihood.loglike(runtime.put([1.0, 2.0]), jitter=-1.0)))
    assert np.isneginf(
        float(
            likelihood.loglike(runtime.put([1.0, 2.0]), uncertainty_scale_parameter=0.0)
        )
    )


def test_multi_dataset_likelihood_sums_device_terms() -> None:
    runtime = _cpu_runtime()
    first = MetalLinearGaussianLikelihood.from_arrays(
        [1.0],
        [0.2],
        MetalLinearProjection.from_matrix([[1.0, 0.0]], runtime),
        runtime,
    )
    second = MetalLinearGaussianLikelihood.from_arrays(
        [3.0],
        [0.4],
        MetalLinearProjection.from_matrix([[0.0, 1.0]], runtime),
        runtime,
    )
    likelihood = MetalMultiDatasetGaussianLikelihood((first, second))
    actual = likelihood.loglike(
        (runtime.put([0.8, 3.2]), runtime.put([0.8, 3.2])),
        offsets=runtime.put([0.0, 0.0]),
        jitters=runtime.put([0.0, 0.0]),
        uncertainty_scales=runtime.put([1.0, 1.0]),
    )
    expected = -0.5 * ((0.2 / 0.2) ** 2 + (-0.2 / 0.4) ** 2)
    assert float(actual) == pytest.approx(expected, rel=2.0e-6)


def test_rorr_validates_host_inputs_and_transparent_species() -> None:
    runtime = _cpu_runtime()
    weights = np.array([0.2, 0.3, 0.5])
    transparent = np.zeros((2, 2, 4, 3))
    actual = metal_random_overlap_species_tau(transparent, weights, runtime=runtime)
    np.testing.assert_array_equal(np.asarray(actual), 0.0)
    with pytest.raises(RobertValidationError, match="finite and non-negative"):
        metal_random_overlap_species_tau(
            np.full((1, 1, 1, 3), -1.0), weights, runtime=runtime
        )
    with pytest.raises(RobertValidationError, match="positive sum"):
        metal_random_overlap_species_tau(
            np.zeros((1, 1, 1, 3)), np.zeros(3), runtime=runtime
        )


def test_sampler_facing_metal_problem_synchronizes_only_scalar_boundary() -> None:
    runtime = _cpu_runtime()
    parameters = RetrievalParameterSet(
        (RetrievalParameter("temperature", UniformPrior(500.0, 2000.0)),)
    )
    problem = MetalRetrievalProblem.from_device_graph(
        name="device-test",
        parameters=parameters,
        parameter_to_loglike=lambda vector: (
            -0.5 * ((vector[0] - runtime.jnp.float32(1000.0)) / 100.0) ** 2
        ),
        runtime=runtime,
    )
    assert problem.parameter_names == ("temperature",)
    assert problem.prior_transform([0.5])[0] == pytest.approx(1250.0)
    assert problem.log_likelihood_from_vector([1100.0]) == pytest.approx(-0.5)
    assert problem.metadata["cpu_fallback"] == "forbidden"
    assert problem.metadata["supported_retrieval_methods"] == "multinest"
    with pytest.raises(RobertConfigError, match="MultiNest only"):
        problem.gaussian_inputs_from_vector([1100.0])
    assert np.isneginf(problem.log_likelihood_from_vector([np.nan]))


def test_metal_only_runtime_does_not_allow_cpu_fallback(monkeypatch) -> None:
    import sys
    import types

    import robert_exoplanets.metal.runtime as module
    from robert_exoplanets.core import RobertConfigError

    fake_jax = types.ModuleType("jax")
    fake_jax.config = type("Config", (), {"x64_enabled": False})()
    fake_jax.devices = lambda: (
        type("Device", (), {"platform": "cpu", "device_kind": "test"})(),
    )
    fake_numpy = types.ModuleType("jax.numpy")
    fake_jax.numpy = fake_numpy
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_numpy)
    with pytest.raises(RobertConfigError, match="refusing CPU fallback"):
        module.require_metal_runtime()


def test_cuda_runtime_maps_public_name_to_jax_gpu_backend(monkeypatch) -> None:
    import sys
    import types

    import robert_exoplanets.metal.runtime as module

    client = type("Client", (), {"platform_version": "CUDA 12.4"})()
    device = type(
        "Device",
        (),
        {"platform": "gpu", "device_kind": "NVIDIA A100", "client": client},
    )()
    queries: list[str] = []
    fake_jax = types.ModuleType("jax")

    def devices(query):
        queries.append(query)
        return (device,)

    fake_jax.devices = devices
    fake_numpy = types.ModuleType("jax.numpy")
    fake_numpy.float32 = np.float32
    fake_jax.numpy = fake_numpy
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_numpy)
    runtime = module.require_accelerator_runtime("cuda")
    assert queries == ["gpu"]
    assert runtime.accelerator == "cuda"
    assert runtime.metadata["backend"] == "jax-cuda"
    assert runtime.visible_device_count == 1
    assert queries == ["gpu", "gpu"]
