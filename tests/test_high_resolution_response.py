"""Tests for the prepared Gaussian high-resolution response."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets.core import RobertCoverageError, RobertValidationError, SpectralGrid, Spectrum
from robert_exoplanets.instruments import Observation
from robert_exoplanets.instruments.high_resolution import (
    DopplerShiftResponse,
    HighResolutionResponseChain,
    GaussianHighResolutionResponse,
    PixelIntegrationResponse,
    PreparedGaussianHighResolutionResponse,
    PreparedPixelIntegrationResponse,
    RelativisticDopplerResponse,
    RotationalBroadeningResponse,
)


def _observation(wavelength: list[float], *, edges: list[float] | None = None) -> Observation:
    return Observation.from_arrays(
        wavelength=wavelength,
        flux=np.zeros(len(wavelength)),
        uncertainty=np.ones(len(wavelength)),
        wavelength_bin_edges=edges,
    )


def _spectrum(grid: SpectralGrid, values: np.ndarray) -> Spectrum:
    return Spectrum(
        spectral_grid=grid,
        values=values,
        unit="eclipse_depth",
        observable="eclipse_depth",
    )


def test_gaussian_response_validates_positive_configuration() -> None:
    with pytest.raises(RobertValidationError, match="resolving_power"):
        GaussianHighResolutionResponse(resolving_power=0.0)
    with pytest.raises(RobertValidationError, match="kernel_support"):
        GaussianHighResolutionResponse(resolving_power=100_000.0, kernel_support=-1.0)
    with pytest.raises(RobertValidationError, match="resolving_power"):
        GaussianHighResolutionResponse(resolving_power=True)


def test_prepared_response_preserves_a_constant_spectrum() -> None:
    native_grid = SpectralGrid.from_array(np.linspace(1.0, 3.0, 1001))
    observation = _observation([1.5, 2.0, 2.5])
    response = GaussianHighResolutionResponse(
        resolving_power=100.0,
        kernel_support=4.0,
    )

    prepared = response.prepare(observation, native_grid)
    observed = prepared.observe(_spectrum(native_grid, np.full(native_grid.size, 2.75)))

    assert isinstance(prepared, PreparedGaussianHighResolutionResponse)
    np.testing.assert_allclose(observed.values, 2.75, rtol=0.0, atol=1.0e-12)
    assert observed.spectral_grid.values.tolist() == observation.wavelength.tolist()
    assert observed.metadata["convolution"] == "piecewise_linear_gaussian"


def test_prepared_response_handles_descending_source_and_observation_grids() -> None:
    ascending_values = np.linspace(1.0, 3.0, 1001)
    ascending_grid = SpectralGrid.from_array(ascending_values)
    descending_grid = SpectralGrid.from_array(ascending_values[::-1])
    ascending_observation = _observation([1.5, 2.0, 2.5])
    descending_observation = _observation([2.5, 2.0, 1.5])
    response = GaussianHighResolutionResponse(resolving_power=100.0)

    ascending = response.prepare(ascending_observation, ascending_grid).observe(
        _spectrum(ascending_grid, ascending_values**2)
    )
    descending = response.prepare(descending_observation, descending_grid).observe(
        _spectrum(descending_grid, ascending_values[::-1] ** 2)
    )

    np.testing.assert_allclose(descending.values, ascending.values[::-1], rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(descending.spectral_grid.values, descending_observation.wavelength)


def test_prepare_rejects_incomplete_gaussian_support_and_bin_coverage() -> None:
    native_grid = SpectralGrid.from_array([1.0, 2.0, 3.0])
    response = GaussianHighResolutionResponse(resolving_power=100.0)

    with pytest.raises(RobertCoverageError, match="kernel support"):
        response.prepare(_observation([0.95]), native_grid)

    with pytest.raises(RobertCoverageError, match="observation bins"):
        response.prepare(_observation([2.0], edges=[0.9, 3.1]), native_grid)


def test_prepared_response_rejects_a_different_native_grid() -> None:
    native_grid = SpectralGrid.from_array(np.linspace(1.0, 3.0, 1001))
    observation = _observation([2.0])
    prepared = GaussianHighResolutionResponse(resolving_power=100.0).prepare(
        observation, native_grid
    )
    different_grid = SpectralGrid.from_array(np.linspace(1.0, 3.0, 1000))

    with pytest.raises(RobertValidationError, match="does not match"):
        prepared.observe(_spectrum(different_grid, np.ones(different_grid.size)))


def test_prepare_accepts_source_grid_alias_and_rejects_missing_grid() -> None:
    native_grid = SpectralGrid.from_array(np.linspace(1.0, 3.0, 101))
    observation = _observation([2.0])
    response = GaussianHighResolutionResponse(resolving_power=100.0)

    prepared = response.prepare(observation, source_grid=native_grid)
    assert prepared.native_grid is prepared.source_grid

    with pytest.raises(RobertValidationError, match="native_grid is required"):
        response.prepare(observation)


def test_relativistic_doppler_shift_uses_wavelength_factor_and_no_extrapolation() -> None:
    native_grid = SpectralGrid.from_array(np.linspace(1.0, 3.0, 1001))
    velocity = 30.0
    response = RelativisticDopplerResponse(velocity_km_s=velocity)
    factor = np.sqrt(
        (1.0 + velocity / 299_792.458) / (1.0 - velocity / 299_792.458)
    )

    shifted = response.prepare(native_grid)
    np.testing.assert_allclose(shifted.target_grid.values, native_grid.values * factor)
    np.testing.assert_allclose(
        shifted.observe(_spectrum(native_grid, np.ones(native_grid.size))).values,
        1.0,
    )

    observation = _observation([0.99, 2.0])
    with pytest.raises(RobertCoverageError, match="Doppler-shifted"):
        response.prepare(observation, native_grid)

    # The alias is the same validated implementation.
    assert isinstance(DopplerShiftResponse(velocity=0.0), RelativisticDopplerResponse)


def test_rotational_broadening_uses_constant_velocity_grid_and_limb_darkening() -> None:
    native_values = np.exp(np.linspace(np.log(1.0), np.log(1.1), 1001))
    native_grid = SpectralGrid.from_array(native_values)
    response = RotationalBroadeningResponse(
        vsini_km_s=20.0,
        linear_limb_darkening=0.5,
        velocity_step_km_s=200.0,
    )

    prepared = response.prepare(native_grid)
    log_step = np.diff(np.log(prepared.target_grid.values))
    np.testing.assert_allclose(log_step, log_step[0], rtol=1.0e-12, atol=0.0)
    observed = prepared.observe(_spectrum(native_grid, np.full(native_grid.size, 4.0)))

    np.testing.assert_allclose(observed.values, 4.0, rtol=0.0, atol=1.0e-12)
    assert observed.metadata["rotation_kernel"] == "gray_linear_limb_darkening"
    assert observed.metadata["limb_darkening"] == "0.5"

    with pytest.raises(RobertCoverageError, match="rotational broadening"):
        response.prepare(_observation([1.00001]), native_grid)


def test_pixel_integration_is_prepared_and_flux_conserving() -> None:
    native_grid = SpectralGrid.from_array(np.linspace(1.0, 3.0, 41))
    observation = _observation([1.5, 2.5], edges=[1.0, 2.0, 3.0])
    response = PixelIntegrationResponse()
    prepared = response.prepare(observation, native_grid)
    spectrum = _spectrum(native_grid, 2.0 * native_grid.values + 1.0)

    observed = prepared.observe(spectrum)

    assert isinstance(prepared, PreparedPixelIntegrationResponse)
    np.testing.assert_allclose(observed.values, [4.0, 6.0], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        prepared.observe(_spectrum(native_grid, np.full(native_grid.size, 3.0))).values,
        3.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert observed.metadata["integration"] == "piecewise_linear_flux_conserving_bins"


def test_response_chain_enforces_order_and_prepares_final_observation_mapping() -> None:
    native_grid = SpectralGrid.from_array(np.linspace(1.0, 3.0, 1001))
    observation = _observation([1.5, 2.0, 2.5], edges=[1.4, 1.8, 2.2, 2.6])
    chain = HighResolutionResponseChain(
        (
            RelativisticDopplerResponse(velocity=0.0),
            RotationalBroadeningResponse(vsini=0.0),
            GaussianHighResolutionResponse(resolving_power=100.0),
        )
    )
    prepared = chain.prepare(observation, native_grid)
    observed = prepared.observe(_spectrum(native_grid, np.full(native_grid.size, 2.0)))

    np.testing.assert_allclose(observed.values, 2.0, rtol=0.0, atol=1.0e-12)
    assert observed.spectral_grid.values.tolist() == observation.wavelength.tolist()
    assert len(prepared.operators) == 4  # final pixel integration is automatic
    assert prepared.operators[-1].name == "pixel-bin-integration"

    with pytest.raises(RobertValidationError, match="ordered"):
        HighResolutionResponseChain(
            (
                PixelIntegrationResponse(),
                GaussianHighResolutionResponse(resolving_power=100.0),
            )
        )

    with pytest.raises(RobertValidationError, match="at most one"):
        HighResolutionResponseChain(
            (
                GaussianHighResolutionResponse(resolving_power=100.0),
                GaussianHighResolutionResponse(resolving_power=200.0),
            )
        )
