"""Regression tests for the Taylor et al. (2021) Figures 1--2 benchmark inputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _benchmark_module():
    path = (
        Path(__file__).parents[1] / "examples" / "benchmark_taylor2021_figures_1_2.py"
    )
    spec = importlib.util.spec_from_file_location("taylor2021_cloud_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_archived_figure1_forward_models_are_loaded() -> None:
    benchmark = _benchmark_module()
    cases = benchmark.load_nemesis_figure1()

    assert tuple(cases) == tuple(range(9))
    for case in cases.values():
        assert case.wavelength_micron.shape == (233,)
        assert case.eclipse_depth.shape == (233,)
        assert np.all(np.diff(case.wavelength_micron) > 0.0)
        assert np.all(case.eclipse_depth > 0.0)


def test_archived_temperature_profiles_have_expected_gradient_signs() -> None:
    benchmark = _benchmark_module()
    pressure, profiles = benchmark.load_taylor_temperature_profiles()

    assert pressure.shape == (50,)
    assert np.all(np.diff(pressure) > 0.0)
    assert profiles["Non-inverted"][-1] > profiles["Non-inverted"][0]
    np.testing.assert_allclose(profiles["Isothermal"], 1400.0)
    assert profiles["Inverted"][-1] < profiles["Inverted"][0]


def test_pressure_edges_enclose_archived_centers() -> None:
    benchmark = _benchmark_module()
    pressure, _ = benchmark.load_taylor_temperature_profiles()
    grid = benchmark.pressure_grid_from_centers(pressure)

    assert grid.n_layers == 50
    assert np.all(grid.edges[:-1] < grid.centers)
    assert np.all(grid.centers < grid.edges[1:])


def test_temperature_edge_reconstruction_preserves_isothermal_profile() -> None:
    benchmark = _benchmark_module()
    edges = benchmark.temperature_edges_from_centers(np.full(50, 1400.0))

    np.testing.assert_allclose(edges, 1400.0)
    assert edges.shape == (51,)


def test_figure1_reference_end_members_show_expected_dilution() -> None:
    benchmark = _benchmark_module()
    cases = benchmark.load_nemesis_figure1()
    ratio = cases[8].eclipse_depth / cases[0].eclipse_depth

    assert 0.4 < float(np.median(ratio)) < 0.6
