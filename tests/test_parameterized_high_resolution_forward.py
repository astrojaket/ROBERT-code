"""Tests for retrieval-facing velocity response orchestration."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import Observation, SpectralGrid, Spectrum
from robert_exoplanets.forward.high_resolution import (
    ParameterizedMultiDatasetResponseForwardModel,
    VelocityParameterizedHighResolutionResponse,
)
from robert_exoplanets.instruments import (
    GaussianHighResolutionResponse,
    HighResolutionResponseChain,
    RelativisticDopplerResponse,
    RotationalBroadeningResponse,
)


def _native_spectrum() -> Spectrum:
    wavelength = 2.29 * np.exp(np.arange(5000, dtype=float) / 500_000.0)
    line = np.exp(-0.5 * np.square((wavelength - 2.3000) / 1.2e-5))
    return Spectrum(
        spectral_grid=SpectralGrid.from_array(wavelength, unit="micron"),
        values=1.0 - 0.2 * line,
        unit="eclipse_depth",
        observable="eclipse_depth",
    )


def _observation() -> Observation:
    centers = 2.2995 * np.exp(np.arange(70, dtype=float) / 100_000.0)
    log_edges = np.empty(centers.size + 1)
    log_centers = np.log(centers)
    log_edges[1:-1] = 0.5 * (log_centers[:-1] + log_centers[1:])
    log_edges[0] = log_centers[0] - 0.5 * (log_centers[1] - log_centers[0])
    log_edges[-1] = log_centers[-1] + 0.5 * (
        log_centers[-1] - log_centers[-2]
    )
    return Observation.from_arrays(
        centers,
        np.zeros(centers.size),
        np.ones(centers.size),
        wavelength_bin_edges=np.exp(log_edges),
    )


def test_velocity_parameter_changes_line_position_and_bounds_cache() -> None:
    native = _native_spectrum()
    response = VelocityParameterizedHighResolutionResponse(
        observation=_observation(),
        resolving_power=100_000.0,
        max_cached_responses=2,
    )

    blue = response.observe(native, {"radial_velocity_km_s": -10.0})
    rest = response.observe(native, {"radial_velocity_km_s": 0.0})
    red = response.observe(native, {"radial_velocity_km_s": 10.0})

    assert blue.spectral_grid.values[np.argmin(blue.values)] < (
        rest.spectral_grid.values[np.argmin(rest.values)]
    )
    assert red.spectral_grid.values[np.argmin(red.values)] > (
        rest.spectral_grid.values[np.argmin(rest.values)]
    )
    assert response.cached_response_count == 2
    assert response.cached_native_grid_count == 1


@pytest.mark.parametrize("descending", [False, True])
def test_velocity_optimization_matches_exact_physical_chain(
    descending: bool,
) -> None:
    native = _native_spectrum()
    observation = _observation()
    if descending:
        native = Spectrum(
            spectral_grid=SpectralGrid.from_array(
                native.spectral_grid.values[::-1],
                unit=native.spectral_grid.unit,
            ),
            values=native.values[::-1],
            unit=native.unit,
            observable=native.observable,
        )
        observation = Observation.from_arrays(
            observation.wavelength[::-1],
            observation.flux[::-1],
            observation.uncertainty[::-1],
            wavelength_bin_edges=observation.wavelength_bin_edges[::-1],
        )
    optimized = VelocityParameterizedHighResolutionResponse(
        observation=observation,
        resolving_power=100_000.0,
        projected_rotation_km_s=4.0,
        limb_darkening=0.3,
        max_cached_responses=2,
        max_cached_native_grids=1,
    )

    for velocity in (-10.0, 0.0, 10.0):
        exact = HighResolutionResponseChain(
            (
                RelativisticDopplerResponse(velocity),
                RotationalBroadeningResponse(
                    projected_velocity_km_s=4.0,
                    limb_darkening=0.3,
                ),
                GaussianHighResolutionResponse(resolving_power=100_000.0),
            )
        ).prepare(observation, native.spectral_grid).observe(native)
        actual = optimized.observe(
            native,
            {"radial_velocity_km_s": velocity},
        )

        np.testing.assert_allclose(actual.values, exact.values, rtol=2.0e-12, atol=2.0e-14)
        np.testing.assert_array_equal(
            actual.spectral_grid.values,
            exact.spectral_grid.values,
        )
        assert actual.metadata["physical_response_order"] == (
            "relativistic-doppler,rotational-broadening,"
            "gaussian-high-resolution,pixel-or-observation-mapping"
        )
        assert "scale-equivariant" in actual.metadata["response_optimization"]

    assert optimized.cached_response_count == 2
    assert optimized.cached_native_grid_count == 1


def test_velocity_optimization_matches_center_sampled_chain() -> None:
    native = _native_spectrum()
    binned_observation = _observation()
    observation = Observation.from_arrays(
        binned_observation.wavelength,
        binned_observation.flux,
        binned_observation.uncertainty,
    )
    optimized = VelocityParameterizedHighResolutionResponse(
        observation=observation,
        resolving_power=100_000.0,
        projected_rotation_km_s=4.0,
        limb_darkening=0.3,
        max_cached_responses=2,
        max_cached_native_grids=1,
    )

    for velocity in (-10.0, 0.0, 10.0):
        exact = HighResolutionResponseChain(
            (
                RelativisticDopplerResponse(velocity),
                RotationalBroadeningResponse(
                    projected_velocity_km_s=4.0,
                    limb_darkening=0.3,
                ),
                GaussianHighResolutionResponse(resolving_power=100_000.0),
            )
        ).prepare(observation, native.spectral_grid).observe(native)
        actual = optimized.observe(
            native,
            {"radial_velocity_km_s": velocity},
        )

        np.testing.assert_allclose(actual.values, exact.values, rtol=2.0e-12, atol=2.0e-14)
        np.testing.assert_array_equal(actual.spectral_grid.values, exact.spectral_grid.values)


def test_velocity_cache_key_includes_native_bin_edges() -> None:
    native = _native_spectrum()
    values = native.spectral_grid.values
    edges = np.empty(values.size + 1)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    binned = Spectrum(
        spectral_grid=SpectralGrid(
            values=values,
            bin_edges=edges,
            unit="micron",
        ),
        values=native.values,
        unit=native.unit,
        observable=native.observable,
    )
    response = VelocityParameterizedHighResolutionResponse(
        observation=_observation(),
        resolving_power=100_000.0,
        max_cached_responses=2,
        max_cached_native_grids=2,
    )

    response.observe(native, {"radial_velocity_km_s": 0.0})
    response.observe(binned, {"radial_velocity_km_s": 0.0})

    assert response.cached_response_count == 2


def test_native_grid_eviction_also_releases_dynamic_response_prefixes() -> None:
    native = _native_spectrum()
    values = native.spectral_grid.values
    edges = np.empty(values.size + 1)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    second_native = Spectrum(
        spectral_grid=SpectralGrid(
            values=values,
            bin_edges=edges,
            unit="micron",
        ),
        values=native.values,
        unit=native.unit,
        observable=native.observable,
    )
    response = VelocityParameterizedHighResolutionResponse(
        observation=_observation(),
        resolving_power=100_000.0,
        max_cached_responses=2,
        max_cached_native_grids=1,
    )

    response.observe(native, {"radial_velocity_km_s": 0.0})
    response.observe(second_native, {"radial_velocity_km_s": 0.0})

    assert response.cached_native_grid_count == 1
    assert response.cached_response_count == 1


def test_zero_native_grid_cache_disables_dynamic_response_cache() -> None:
    native = _native_spectrum()
    response = VelocityParameterizedHighResolutionResponse(
        observation=_observation(),
        resolving_power=100_000.0,
        max_cached_responses=2,
        max_cached_native_grids=0,
    )

    response.observe(native, {"radial_velocity_km_s": -10.0})
    response.observe(native, {"radial_velocity_km_s": 10.0})

    assert response.cached_native_grid_count == 0
    assert response.cached_response_count == 0


def test_parameterized_multi_dataset_wrapper_keeps_unlisted_spectrum() -> None:
    native = _native_spectrum()
    low = Spectrum.from_arrays(
        [1.0, 2.0],
        [0.1, 0.2],
        "eclipse_depth",
        "eclipse_depth",
    )
    response = VelocityParameterizedHighResolutionResponse(
        observation=_observation(),
        resolving_power=100_000.0,
    )
    model = ParameterizedMultiDatasetResponseForwardModel(
        native_model=lambda parameters: {"low": low, "high": native},
        runtime_responses={"high": response},
    )

    prediction = model({"radial_velocity_km_s": 3.0})

    assert prediction["low"] is low
    assert prediction["high"].spectral_grid.values.shape == (70,)
    assert model.required_parameters == ("radial_velocity_km_s",)
