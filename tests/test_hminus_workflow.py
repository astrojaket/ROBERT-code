"""Focused tests for the external H-minus validation workflow."""

from __future__ import annotations

import json
from pathlib import Path
import shlex

import numpy as np
import pytest

from examples import benchmark_hminus_continuum as benchmark
from examples import run_petitradtrans3_hminus_oracle as oracle


def test_report_paths_and_replay_command_are_portable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_file = benchmark.ROOT / "examples" / "benchmark_hminus_continuum.py"
    assert benchmark._display_path(project_file) == (
        "examples/benchmark_hminus_continuum.py"
    )
    assert benchmark._display_path(tmp_path / "external.dat") == str(
        (tmp_path / "external.dat").resolve()
    )

    monkeypatch.setattr(
        "sys.argv",
        ["benchmark_hminus_continuum.py", "--max-memory-gib", "1.8"],
    )
    assert shlex.split(benchmark._reproducible_command()) == [
        "conda",
        "run",
        "-n",
        "robert-exoplanets",
        "python",
        "examples/benchmark_hminus_continuum.py",
        "--max-memory-gib",
        "1.8",
    ]


def test_documented_vmr_states_are_normalised_and_component_isolated() -> None:
    for mode in oracle.MODES:
        state = oracle.vmr_state(mode)
        vmr = state.vmr
        assert sum(vmr.values()) == pytest.approx(1.0)
        if mode == "bound_free":
            assert vmr["H-"] > 0.0
            assert vmr["H"] == 0.0
            assert vmr["e-"] == 0.0
        elif mode == "free_free":
            assert vmr["H-"] == 0.0
            assert vmr["H"] > 0.0
            assert vmr["e-"] > 0.0
        else:
            assert vmr["H-"] > 0.0


def test_pRT_boundary_conversion_is_private_and_conserves_mass() -> None:
    state = oracle.vmr_state("total")
    mass_fractions, mmw = oracle._vmr_to_mass_fractions(state)
    assert mmw > 0.0
    assert sum(mass_fractions.values()) == pytest.approx(1.0, abs=2.0e-15)
    assert set(mass_fractions) == set(state.vmr)
    # The ROBERT-facing state remains the original VMR mapping.
    assert state.vmr["H-"] == oracle.BASE_HMINUS_VMR


def test_opacity_guards_are_strictly_below_one_gib() -> None:
    one_gib = 1024**3
    assert 0 < oracle.MAX_OPACITY_BYTES < one_gib
    assert 0 < benchmark.MAX_OPACITY_BYTES < one_gib


def test_oracle_cli_default_opacity_guard_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_petitradtrans3_hminus_oracle.py"])
    arguments = oracle._parse_args()
    assert arguments.max_opacity_gib < 1.0


def test_temperature_profile_matches_the_documented_formula() -> None:
    profile = oracle.pressure_temperature_profile(3000.0)
    expected = 3000.0 + 100.0 * np.log10(oracle.PRESSURE_CENTERS_BAR)
    np.testing.assert_allclose(profile, expected)
    assert np.all(profile > 0.0)


def test_pRT_geometry_uses_final_projected_disc_weights() -> None:
    geometry = benchmark._geometry()
    expected = 2.0 * benchmark.P_RT_MU * benchmark.P_RT_WEIGHTS
    np.testing.assert_allclose(benchmark.P_RT_DISK_WEIGHTS, expected)
    np.testing.assert_allclose(geometry.emission_angle_weights, expected)
    np.testing.assert_allclose(np.sum(geometry.emission_angle_weights), 1.0)
    assert not np.allclose(geometry.emission_angle_weights, benchmark.P_RT_WEIGHTS)
    assert geometry.name == "petitRADTRANS-3-point-thermal-disc"
    assert geometry.quadrature == "petitRADTRANS_gauss_legendre_projected_disc"
    assert geometry.metadata["final_weight_formula"] == (
        "2 * emission_mu * pRT_integral_weight"
    )
    assert geometry.metadata["raw_gauss_weights_passed"] == "false"


def test_maintained_report_records_final_angular_weights() -> None:
    report_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "data"
        / "hminus_continuum_validation_20260831.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    quadrature = report["angular_quadrature"]
    expected = 2.0 * benchmark.P_RT_MU * benchmark.P_RT_WEIGHTS
    np.testing.assert_allclose(quadrature["emission_mu"], benchmark.P_RT_MU)
    np.testing.assert_allclose(
        quadrature["pRT_integral_weights"], benchmark.P_RT_WEIGHTS
    )
    np.testing.assert_allclose(quadrature["final_disk_weights"], expected)
    assert quadrature["weight_formula"] == "2 * emission_mu * pRT_integral_weight"
    assert quadrature["final_weight_sum"] == pytest.approx(1.0)
    assert quadrature["raw_gauss_weights_passed"] is False
    assert quadrature["robert_receives"] == "final_disk_weights"


def test_john_bound_free_threshold_differs_from_ROBERT_gray_threshold() -> None:
    wavelengths = np.array([1.6, 1.6420, 1.6421, 2.3])
    john = benchmark._john_bound_free_cross_section(wavelengths)
    gray = benchmark._extinction._hminus_bound_free_point_cross_section(
        wavelengths * 1.0e4
    ) * 1.0e4
    assert john[0] > 0.0
    assert john[1] > 0.0
    assert john[2] == 0.0
    assert gray[2] == 0.0
    assert gray[3] == 0.0


def test_john_free_free_is_close_to_ROBERT_gray_fit_at_K_band() -> None:
    wavelength = np.array([2.3])
    john = benchmark._john_free_free_cross_section(wavelength, 3000.0, 1000.0)
    gray = benchmark._extinction._hminus_free_free_cross_section(
        wavelength * 1.0e4,
        3000.0,
        1000.0,
    )
    np.testing.assert_allclose(john, gray, rtol=2.0e-3)


def test_john_free_free_uses_only_the_selected_long_wave_branch() -> None:
    values = benchmark._john_free_free_cross_section(
        np.array([0.2, 0.3644, 0.3645, 2.3]),
        3000.0,
        1000.0,
    )
    assert values[0] == 0.0
    assert values[1] == 0.0
    assert values[2] > 0.0
    assert values[3] > 0.0


def test_john_metadata_has_official_sources_and_no_unsourced_upper_limit() -> None:
    diagnostic = benchmark._poseidon_john_diagnostic()
    assert diagnostic["required_gate"] is False
    assert diagnostic["bound_free_threshold_micron"] == pytest.approx(1.6421)
    assert diagnostic["official_poseidon_absorption_url"].endswith(
        "/POSEIDON/absorption.py"
    )
    assert "1988A%26A...193..189J" in diagnostic["john_source_ads_url"]
    assert "appendix" not in str(diagnostic["source"]).lower()
    assert diagnostic["free_free_upper_validity_sourced"] is False
    assert diagnostic["free_free_upper_validity_micron"] is None


def test_metrics_are_zero_for_identical_spectra() -> None:
    values = np.array([1.0, 2.0, 3.0])
    metrics = benchmark._metrics(values, values)
    assert metrics["rms_relative_error"] == pytest.approx(0.0)
    assert metrics["max_absolute_relative_error"] == pytest.approx(0.0)
    assert metrics["integrated_flux_relative_error"] == pytest.approx(0.0)


def test_metrics_integrate_nonuniform_bins_instead_of_summing_samples() -> None:
    reference = np.ones(3)
    candidate = np.array([1.0, 1.0, 2.0])
    edges = np.array([1.0, 1.1, 3.0, 8.0])
    metrics = benchmark._metrics(
        reference,
        candidate,
        wavelength_bin_edges_micron=edges,
    )
    expected = (0.1 + 1.9 + 2.0 * 5.0) / (0.1 + 1.9 + 5.0) - 1.0
    plain_sum = np.sum(candidate) / np.sum(reference) - 1.0
    assert metrics["integrated_flux_relative_error"] == pytest.approx(expected)
    assert metrics["integrated_flux_relative_error"] != pytest.approx(plain_sum)


def test_metric_gate_checks_both_rms_and_maximum_error() -> None:
    assert benchmark._metric_gate(
        {
            "rms_relative_error": 0.01,
            "max_absolute_relative_error": 0.04,
            "integrated_flux_relative_error": 0.005,
        },
        rms_limit=0.05,
        max_limit=0.20,
    )
    assert not benchmark._metric_gate(
        {
            "rms_relative_error": 0.01,
            "max_absolute_relative_error": 0.21,
            "integrated_flux_relative_error": 0.005,
        },
        rms_limit=0.05,
        max_limit=0.20,
    )


def test_lsf_target_is_inside_the_native_grid() -> None:
    native = np.linspace(2.2988, 2.30378, 2000)
    target = benchmark._lsf_target(native)
    assert target.size > 2
    assert native[0] < target[0] < target[-1] < native[-1]
