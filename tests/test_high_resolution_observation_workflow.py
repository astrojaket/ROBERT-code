"""Focused tests for the synthetic multi-order HRS contract benchmark."""

from __future__ import annotations

import json

import pytest

from examples.benchmark_high_resolution_observation_pipeline import (
    ORDER_SPECS,
    run_benchmark,
    write_report,
)


@pytest.fixture(scope="module")
def benchmark_report() -> dict[str, object]:
    """Run the small deterministic benchmark once for this test module."""

    return run_benchmark()


def test_contract_benchmark_passes_without_science_scope_claims(
    benchmark_report: dict[str, object],
) -> None:
    assert benchmark_report["status"] == "pass"
    assert benchmark_report["passed"] is True
    assert benchmark_report["scope"] == "synthetic operator contract"
    assert benchmark_report["not_real_target_science_validation"] is True
    assert benchmark_report["not_cross_order_covariance"] is True
    assert benchmark_report["opacity_data_used"] is False
    assert benchmark_report["sampler_used"] is False
    assert all(benchmark_report["gates"].values())


def test_two_orders_have_distinct_grids_bins_masks_and_lsf_settings(
    benchmark_report: dict[str, object],
) -> None:
    assert tuple(benchmark_report["orders"]) == ("order_blue", "order_red")
    blue = benchmark_report["orders"]["order_blue"]
    red = benchmark_report["orders"]["order_red"]

    assert blue["wavelengths"] != red["wavelengths"]
    assert blue["pixel_edges"] != red["pixel_edges"]
    assert blue["effective_mask"] != red["effective_mask"]
    assert blue["lsf"]["resolving_power"] != red["lsf"]["resolving_power"]
    assert blue["filter"]["shape"] != red["filter"]["shape"]
    assert blue["order_number"] == 1
    assert red["order_number"] == 2

    assert len(ORDER_SPECS) == 2
    assert ORDER_SPECS[0].name == "order_blue"
    assert ORDER_SPECS[1].name == "order_red"


def test_response_order_velocity_sign_and_runtime_term_are_recorded(
    benchmark_report: dict[str, object],
) -> None:
    expected_order = [
        "relativistic-doppler",
        "gaussian-high-resolution",
        "pixel-bin-integration",
    ]
    blue_velocity = benchmark_report["orders"]["order_blue"]["velocity"]
    red_velocity = benchmark_report["orders"]["order_red"]["velocity"]

    assert benchmark_report["runtime_radial_velocity_km_s"] == pytest.approx(2.5)
    assert blue_velocity["fixed_velocity_km_s"] == pytest.approx(9.0)
    assert blue_velocity["total_velocity_km_s"] == pytest.approx(11.5)
    assert red_velocity["fixed_velocity_km_s"] == pytest.approx(-3.0)
    assert red_velocity["total_velocity_km_s"] == pytest.approx(-0.5)
    assert blue_velocity["observed_doppler_velocity_km_s"] == pytest.approx(11.5)
    assert red_velocity["observed_doppler_velocity_km_s"] == pytest.approx(-0.5)
    assert blue_velocity["doppler_factor"] > 1.0
    assert red_velocity["doppler_factor"] < 1.0
    for order in benchmark_report["orders"].values():
        assert order["response_order"] == expected_order
        assert order["required_response_order"] == expected_order
        assert "positive velocity shifts model wavelengths longer" in order[
            "velocity"
        ]["sign_convention"]


def test_same_response_filter_and_covariance_operator_contract_is_proven(
    benchmark_report: dict[str, object],
) -> None:
    for order in benchmark_report["orders"].values():
        operator_contract = order["operator_contract"]
        covariance = order["covariance"]
        assert operator_contract["same_response_operator_reused"] is True
        assert operator_contract["same_filter_operator_data_model"] is True
        assert operator_contract["filtered_observation_matches_filtered_data"] is True
        assert covariance["propagation_formula"] == "C_filtered = F C_input F.T"
        assert covariance["propagation_passed"] is True


def test_dense_and_psd_ranks_and_independent_correlated_chi_square_match(
    benchmark_report: dict[str, object],
) -> None:
    blue = benchmark_report["orders"]["order_blue"]
    red = benchmark_report["orders"]["order_red"]

    assert blue["covariance"]["operator"] == "DenseCovariance"
    assert blue["covariance"]["retained_rank"] == 4
    assert blue["covariance"]["independent_numpy_rank"] == 4
    assert blue["chi_square"]["method"] == "np.linalg.solve for DenseCovariance"

    assert red["covariance"]["operator"] == "PositiveSemidefiniteCovariance"
    assert red["covariance"]["retained_rank"] == 5
    assert red["covariance"]["independent_numpy_rank"] == 5
    assert red["chi_square"]["method"] == (
        "np.linalg.pinv(rcond=1e-12) for PositiveSemidefiniteCovariance"
    )
    for order in (blue, red):
        assert order["chi_square"]["passed"] is True
        assert order["chi_square"]["absolute_difference"] < 1.0e-10


def test_report_records_independent_order_covariance_and_input_checksums(
    benchmark_report: dict[str, object],
    tmp_path,
) -> None:
    covariance_assumption = benchmark_report["covariance_assumption"]
    assert covariance_assumption["orders_are_independent"] is True
    assert covariance_assumption["cross_order_covariance_included"] is False
    assert "No covariance block connects the two orders" in covariance_assumption[
        "statement"
    ]
    assert len(benchmark_report["input_checksum_sha256"]) == 64
    for order in benchmark_report["orders"].values():
        assert set(order["checksum_inputs"]) >= {
            "observation_wavelengths_sha256",
            "pixel_edges_sha256",
            "observation_mask_sha256",
            "order_mask_sha256",
            "native_grid_sha256",
            "filter_matrix_sha256",
            "input_covariance_sha256",
        }
        assert all(len(value) == 64 for value in order["checksum_inputs"].values())

    path = tmp_path / "contract.json"
    write_report(benchmark_report, path)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["status"] == "pass"
    assert written["input_checksum_sha256"] == benchmark_report["input_checksum_sha256"]


def test_resource_report_is_within_requested_limits(
    benchmark_report: dict[str, object],
) -> None:
    resource = benchmark_report["resource"]
    assert resource["cpu_guard_passed"] is True
    assert resource["rss_guard_passed"] is True
    assert resource["peak_rss_bytes"] < resource["rss_limit_bytes"] < 2 * 1024**3
    assert all(1 <= value <= 3 for value in resource["thread_values"].values())
