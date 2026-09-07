"""Tests for named high-resolution observation preparation."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    HighResolutionObservation,
    HighResolutionOrder,
    LinearDataModelFilter,
    Observation,
    PreparedLinearDataModelFilter,
    SpectralGrid,
    Spectrum,
    VelocityBookkeeping,
)
from robert_exoplanets.core import RobertValidationError


def _observation(
    wavelength: list[float],
    *,
    mask: list[bool] | None = None,
) -> Observation:
    return Observation.from_arrays(
        wavelength=wavelength,
        flux=np.zeros(len(wavelength)),
        uncertainty=np.ones(len(wavelength)),
        mask=mask,
    )


def _native_spectrum(grid: SpectralGrid) -> Spectrum:
    return Spectrum(
        spectral_grid=grid,
        values=np.ones(grid.size),
        unit="eclipse_depth",
        observable="eclipse_depth",
    )


def test_velocity_bookkeeping_has_explicit_additive_sign_convention() -> None:
    bookkeeping = VelocityBookkeeping(
        barycentric_velocity_km_s=12.5,
        systemic_velocity_km_s=-4.0,
    )

    assert bookkeeping.fixed_velocity_km_s == pytest.approx(8.5)
    assert bookkeeping.total_velocity_km_s == pytest.approx(8.5)
    assert bookkeeping.with_additional_velocity(1.5) == pytest.approx(10.0)

    with pytest.raises(RobertValidationError, match="speed of light"):
        VelocityBookkeeping(systemic_velocity_km_s=299_792.458)


def test_order_combines_masks_and_infers_pixel_edges() -> None:
    observation = _observation([1.4, 1.6], mask=[True, False])
    order = HighResolutionOrder(
        name="order-a",
        observation=observation,
        mask=[True, True],
        lsf_resolving_power=100_000.0,
    )

    assert order.valid_mask.tolist() == [True, False]
    assert order.response_observation.wavelength_bin_edges is not None
    np.testing.assert_allclose(
        order.response_observation.wavelength_bin_edges,
        [1.3, 1.5, 1.7],
    )


def test_runtime_velocity_is_added_after_fixed_terms_with_redshift_sign() -> None:
    order = HighResolutionOrder(
        name="order-a",
        observation=_observation([1.4, 1.6]),
        barycentric_velocity_km_s=12.0,
        systemic_velocity_km_s=-4.0,
    )

    chain = order.response_chain_for_velocity(3.0)
    assert chain.stages[0].velocity_km_s == pytest.approx(11.0)
    assert chain.stages[0].doppler_factor > 1.0

    native_grid = SpectralGrid.from_array(np.linspace(1.0, 2.0, 2001))
    prepared = order.prepare_runtime_velocity(native_grid, 3.0)
    assert prepared.response.operators[0].velocity_km_s == pytest.approx(11.0)

    with pytest.raises(RobertValidationError, match="additional_velocity"):
        order.response_chain_for_velocity(np.nan)


def test_collection_prepares_distinct_runtime_velocities_per_order() -> None:
    collection = HighResolutionObservation.from_mapping(
        {"blue": _observation([1.4, 1.6]), "red": _observation([1.4, 1.6])},
        barycentric_velocities_km_s={"blue": 10.0, "red": -10.0},
    )
    native_grid = SpectralGrid.from_array(np.linspace(1.0, 2.0, 2001))
    prepared = collection.prepare(
        {"blue": native_grid, "red": native_grid},
        {"blue": 2.0, "red": -3.0},
    )

    assert prepared.order("blue").response.operators[0].velocity_km_s == pytest.approx(12.0)
    assert prepared.order("red").response.operators[0].velocity_km_s == pytest.approx(-13.0)


def test_named_orders_prepare_order_specific_lsf_and_pixel_responses() -> None:
    observations = {
        "blue": _observation([1.4, 1.6]),
        "red": _observation([1.4, 1.6]),
    }
    collection = HighResolutionObservation.from_mapping(
        observations,
        order_numbers={"blue": 1, "red": 2},
        barycentric_velocities_km_s={"blue": 10.0, "red": 10.0},
        systemic_velocities_km_s={"blue": -2.0, "red": -2.0},
        lsf_resolving_powers={"blue": 100_000.0, "red": 50_000.0},
    )
    native_grid = SpectralGrid.from_array(np.linspace(1.0, 2.0, 2001))
    prepared = collection.prepare({"blue": native_grid, "red": native_grid})

    assert collection.order_names == ("blue", "red")
    assert collection.order("blue").fixed_velocity_km_s == pytest.approx(8.0)
    assert prepared.order_names == ("blue", "red")
    for name, resolving_power in (("blue", 100_000.0), ("red", 50_000.0)):
        operators = prepared.order(name).response.prepared_operators
        gaussian = next(
            operator
            for operator in operators
            if getattr(operator, "metadata", {}).get("convolution")
            == "piecewise_linear_gaussian"
        )
        assert gaussian.metadata["resolving_power"] == f"{resolving_power:g}"
        assert any("pixel" in str(getattr(operator, "name", "")) for operator in operators)

    outputs = prepared.observe({"blue": _native_spectrum(native_grid), "red": _native_spectrum(native_grid)})
    assert set(outputs) == {"blue", "red"}
    assert outputs["blue"].values.shape == (2,)


def test_fixed_filter_uses_the_same_matrix_for_data_and_model_and_mask() -> None:
    observation = _observation([1.0, 2.0, 3.0], mask=[True, False, True])
    operator = LinearDataModelFilter(
        matrix=np.array(
            [
                [1.0, -1.0, 0.0],
                [0.0, 1.0, -1.0],
                [1.0, 0.0, -1.0],
            ]
        )
    )
    prepared = operator.prepare(observation)
    assert isinstance(prepared, PreparedLinearDataModelFilter)
    np.testing.assert_allclose(
        prepared.apply_data_model(
            [1.0, 20.0, 3.0],
            [0.0, 2.0, 1.0],
        )[0],
        [1.0, -3.0, -2.0],
    )
    np.testing.assert_allclose(
        prepared.apply_data_model(
            [1.0, 20.0, 3.0],
            [0.0, 2.0, 1.0],
        )[1],
        [-0.0, -1.0, -1.0],
    )
    assert prepared.output_mask.tolist() == [True, True, True]
    assert prepared.effective_matrix.flags.writeable is False


def test_prepared_filter_binds_observation_grid_and_rejects_wrong_grid() -> None:
    observation = Observation.from_arrays(
        wavelength=[1.0, 2.0, 3.0],
        flux=[0.0, 0.0, 0.0],
        uncertainty=[1.0, 1.0, 1.0],
    )
    prepared = LinearDataModelFilter(np.eye(3)).prepare(observation)
    assert prepared.observation is observation

    matching = _native_spectrum(observation.spectral_grid)
    assert prepared.apply_spectrum(matching).values.shape == (3,)

    wrong_grid = _native_spectrum(SpectralGrid.from_array([1.0, 2.0, 3.01]))
    with pytest.raises(RobertValidationError, match="coordinates"):
        prepared.apply_spectrum(wrong_grid)

    wrong_unit = Spectrum(
        spectral_grid=SpectralGrid.from_array(
            [1.0, 2.0, 3.0],
            unit="cm-1",
        ),
        values=np.ones(3),
        unit="eclipse_depth",
        observable="eclipse_depth",
    )
    with pytest.raises(RobertValidationError, match="wavelength units"):
        prepared.apply_spectrum(wrong_unit)


def test_order_preparation_applies_fixed_filter_after_response() -> None:
    observation = _observation([1.4, 1.6, 1.8])
    filter_operator = LinearDataModelFilter(np.eye(3), name="fixed-pipeline-filter")
    order = HighResolutionOrder(
        name="order-a",
        observation=observation,
        lsf_resolving_power=50_000.0,
        data_model_filter=filter_operator,
    )
    prepared = HighResolutionObservation.from_orders([order]).prepare(
        {"order-a": SpectralGrid.from_array(np.linspace(1.0, 2.0, 2001))}
    )
    output = prepared.observe(
        {"order-a": _native_spectrum(SpectralGrid.from_array(np.linspace(1.0, 2.0, 2001)))}
    )["order-a"]

    assert output.metadata["linear_filter"] == "fixed-pipeline-filter"
    assert output.spectral_grid.size == observation.n_points
    assert prepared.order("order-a").valid_mask.tolist() == [True, True, True]


def test_named_order_validation_rejects_unknown_metadata_and_grid_names() -> None:
    with pytest.raises(RobertValidationError, match="metadata keys"):
        HighResolutionObservation.from_mapping(
            {"order-a": _observation([1.0, 2.0])},
            lsf_resolving_powers={"order-b": 100_000.0},
        )

    collection = HighResolutionObservation.from_mapping(
        {"order-a": _observation([1.0, 2.0])}
    )
    with pytest.raises(RobertValidationError, match="grid names"):
        collection.prepare(
            {"order-b": SpectralGrid.from_array([0.9, 1.0, 2.1])}
        )
