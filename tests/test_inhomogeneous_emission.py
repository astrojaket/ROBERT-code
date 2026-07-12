"""Tests for inhomogeneous dayside emission composition."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    DilutedEmissionModel,
    DiskEmissionModelConfig,
    SpectralGrid,
    Spectrum,
    TwoRegionEmissionModel,
    build_disk_emission_model,
)
from robert_exoplanets.core import RobertValidationError


def _constant(values: list[float]):
    spectrum = Spectrum(
        spectral_grid=SpectralGrid.from_array([2.0, 5.0], unit="micron"),
        values=np.asarray(values),
        unit="eclipse_depth",
        observable="eclipse_depth",
    )
    return lambda parameters: spectrum


def test_two_region_model_is_exact_areal_flux_mixture() -> None:
    model = TwoRegionEmissionModel(_constant([10.0, 20.0]), _constant([2.0, 4.0]))

    spectrum = model({"hot_area_fraction": 0.68})

    np.testing.assert_allclose(spectrum.values, [7.44, 14.88])
    assert spectrum.metadata["source_equation"] == "Schlawin_et_al_2024_equation_1"


def test_two_region_limits_recover_each_region() -> None:
    model = TwoRegionEmissionModel(_constant([10.0, 20.0]), _constant([2.0, 4.0]))

    np.testing.assert_allclose(model({"hot_area_fraction": 1.0}).values, [10.0, 20.0])
    np.testing.assert_allclose(model({"hot_area_fraction": 0.0}).values, [2.0, 4.0])


def test_dilution_scales_observable_not_spectral_coordinate() -> None:
    model = DilutedEmissionModel(_constant([10.0, 20.0]))

    spectrum = model({"dayside_dilution": 0.68})

    np.testing.assert_allclose(spectrum.values, [6.8, 13.6])
    np.testing.assert_allclose(spectrum.spectral_grid.values, [2.0, 5.0])
    assert spectrum.metadata["dilution_definition"] == (
        "fractional_projected_dayside_emitting_area"
    )


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan])
def test_area_fractions_must_be_physical(value: float) -> None:
    model = DilutedEmissionModel(_constant([10.0, 20.0]))

    with pytest.raises(RobertValidationError):
        model({"dayside_dilution": value})


def test_disk_model_configuration_selects_existing_regional_hardware() -> None:
    hot = _constant([10.0, 20.0])
    cold = _constant([2.0, 4.0])

    one_region = build_disk_emission_model(DiskEmissionModelConfig("one_region"), hot)
    diluted = build_disk_emission_model(DiskEmissionModelConfig("diluted"), hot)
    two_region = build_disk_emission_model(
        DiskEmissionModelConfig("2tp"),
        hot,
        secondary_model=cold,
    )

    np.testing.assert_allclose(one_region({}).values, [10.0, 20.0])
    np.testing.assert_allclose(diluted({"dayside_dilution": 0.5}).values, [5.0, 10.0])
    np.testing.assert_allclose(two_region({"hot_area_fraction": 0.5}).values, [6.0, 12.0])
