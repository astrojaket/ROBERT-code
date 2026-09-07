"""Tests for shared continuum optical-depth evaluation paths."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robert_exoplanets import (
    LayerOpticalDepth,
    ParameterizedEmissionModelConfig,
    PressureGrid,
    SpectralGrid,
)
from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.forward import _atmospheric


def test_streamed_additional_optical_depth_preserves_order_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pressure_grid = PressureGrid(
        edges=np.array([0.1, 1.0, 10.0]),
        centers=np.array([0.316227766, 3.16227766]),
        unit="bar",
    )
    spectral_grid = SpectralGrid.from_array(
        [2.0, 2.1, 2.2], unit="micron", role="native"
    )
    gas = SimpleNamespace(pressure_grid=pressure_grid, spectral_grid=spectral_grid)
    contributions = {
        "cia-a": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "cia-b": np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]),
        "hminus": np.array([[0.001, 0.002, 0.003], [0.004, 0.005, 0.006]]),
        "rayleigh": np.array([[0.0001, 0.0002, 0.0003], [0.0004, 0.0005, 0.0006]]),
    }

    def candidate(name: str) -> LayerOpticalDepth:
        return LayerOpticalDepth(
            name=name,
            tau=contributions[name],
            spectral_grid=spectral_grid,
            pressure_grid=pressure_grid,
            kind="scattering_extinction" if name == "rayleigh" else "absorption_continuum",
            metadata={"source": f"fixture-{name}"},
        )

    monkeypatch.setattr(
        _atmospheric,
        "cia_optical_depth",
        lambda _gas, table, **_kwargs: candidate(f"cia-{table}"),
    )
    monkeypatch.setattr(
        _atmospheric,
        "hminus_optical_depth",
        lambda _gas, _config: candidate("hminus"),
    )
    monkeypatch.setattr(
        _atmospheric,
        "rayleigh_scattering_optical_depth",
        lambda _gas: candidate("rayleigh"),
    )

    combined = _atmospheric.evaluate_combined_additional_optical_depth(
        gas,
        cia_tables=("a", "b"),
        hminus_continuum=object(),
        include_rayleigh=True,
    )

    assert combined is not None
    expected = (
        contributions["cia-a"]
        + contributions["cia-b"]
        + contributions["hminus"]
        + contributions["rayleigh"]
    )
    assert combined.tau.dtype == np.float64
    np.testing.assert_array_equal(combined.tau, expected)
    assert combined.name == "cia-a+cia-b+hminus+rayleigh"
    assert combined.kind == "extinction"
    assert combined.metadata["aggregation"] == "streamed_float64_sum"
    assert combined.metadata["source_names"] == combined.name
    assert combined.metadata["source_count"] == "4"
    assert combined.metadata["source_0_source"] == "fixture-cia-a"
    assert combined.metadata["source_3_source"] == "fixture-rayleigh"
    assert not combined.tau.flags.writeable


def test_streamed_additional_optical_depth_returns_none_without_sources() -> None:
    pressure_grid = PressureGrid(
        edges=np.array([0.1, 1.0]),
        centers=np.array([0.316227766]),
        unit="bar",
    )
    spectral_grid = SpectralGrid.from_array([2.0], unit="micron", role="native")
    gas = SimpleNamespace(pressure_grid=pressure_grid, spectral_grid=spectral_grid)

    assert (
        _atmospheric.evaluate_combined_additional_optical_depth(
            gas,
            include_rayleigh=False,
        )
        is None
    )


def test_parameterized_emission_aggregation_is_explicit_and_boolean() -> None:
    default = ParameterizedEmissionModelConfig(opacity_species=("H2O",))
    assert default.aggregate_additional_optical_depths is False
    enabled = ParameterizedEmissionModelConfig(
        opacity_species=("H2O",),
        aggregate_additional_optical_depths=True,
    )
    assert enabled.aggregate_additional_optical_depths is True

    with pytest.raises(
        RobertValidationError,
        match="aggregate_additional_optical_depths must be a boolean",
    ):
        ParameterizedEmissionModelConfig(
            opacity_species=("H2O",),
            aggregate_additional_optical_depths=1,  # type: ignore[arg-type]
        )
