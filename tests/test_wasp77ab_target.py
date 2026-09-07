"""Lightweight tests for the WASP-77Ab real-data protocol metadata."""

from __future__ import annotations

import pytest

from examples import wasp77ab_target as target


def test_table_1_target_objects_use_cited_nominal_values() -> None:
    assert target.PLANET.name == "WASP-77Ab"
    assert target.STAR.name == "WASP-77A"
    assert target.STAR_TABLE_1["spectral_type"].value == "G8V"
    assert target.PLANET_TABLE_1["radius"].value == pytest.approx(1.21)
    assert target.PLANET_TABLE_1["mass"].unit == "M_J"
    assert target.PLANET_GRAVITY_M_S2 == pytest.approx(29.796051121185513)
    assert target.PLANET.metadata["source_url"] == target.SMITH_PAPER_URL


def test_default_orbit_is_latest_circular_table_1_choice() -> None:
    assert target.ORBIT_DEFAULT == "circular"
    assert target.ORBITAL_PERIOD_DAY == pytest.approx(1.36002854)
    assert target.TRANSIT_EPOCH_BJD == pytest.approx(2457420.88439)
    assert target.ORBITAL_PERIOD_DAY != target.ORBITAL_PERIOD_DAY_MAXTED
    assert target.TRANSIT_EPOCH_BJD != target.TRANSIT_EPOCH_BJD_MAXTED
    assert target.PLANET_KP_KM_S == pytest.approx(192.0)


def test_primary_and_sensitivity_night_plan_is_explicit() -> None:
    by_label = {night.label: night for night in target.SMITH_NIGHTS}
    assert target.PRIMARY_HRS_NIGHT in by_label
    assert by_label[target.PRIMARY_HRS_NIGHT].frame_count == 79
    assert by_label[target.PRIMARY_HRS_NIGHT].primary_pc_count == 4
    assert target.POST_ECLIPSE_PRIMARY_PC_COUNT == 3
    assert target.HRS_PC_SENSITIVITY_COUNTS == (2, 3, 4, 6, 8)
    assert target.HRS_DISCARDED_EDGE_PIXELS == 200
    assert target.HRS_DISCARDED_ORDER_COUNT == 8


def test_smith_priors_are_vmr_named_and_bounded() -> None:
    assert target.SMITH_PRIORS["log10_h2o_vmr"].lower == -12.0
    assert target.SMITH_PRIORS["log10_co_vmr"].upper == 0.0
    assert target.SMITH_PRIORS["dKp_km_s"].unit == "km s^-1"
    assert target.SMITH_PRIORS["log10_P1_bar"].lower == -5.5
    assert target.SMITH_PRIORS["log10_P3_bar"].upper == 2.0
    assert "mass_fraction" not in " ".join(target.SMITH_PRIORS)


def test_hminus_and_sampler_policies_are_hard_contracts() -> None:
    assert target.HMINUS_POLICY["retrievable"] is True
    assert target.HMINUS_POLICY["state_abundance_type"] == "VMR"
    assert target.HMINUS_POLICY["species"] == ("H-", "H", "e-")
    assert target.HMINUS_POLICY["mass_fraction_parameters_in_robert"] is False
    assert target.SAMPLER_CONTRACT.sampler == "PyMultiNest"
    assert target.SAMPLER_CONTRACT.backend == "MultiNest"
    assert target.SAMPLER_CONTRACT.seeds == (24680, 24681)
    assert target.SAMPLER_CONTRACT.mpi_processes == 1
    assert target.SAMPLER_CONTRACT.max_threads == 3
    assert target.SAMPLER_CONTRACT.process_rss_limit_bytes == 2 * 1024**3
    assert target.SAMPLER_CONTRACT.opacity_rss_limit_bytes == 1023 * 1024**2
    assert len(target.SAMPLER_CONTRACT.thread_variables) == 7
    assert "UltraNest" in target.SAMPLER_POLICY


def test_data_resources_preserve_known_hashes_and_nirspec_count_reconciliation() -> None:
    assert target.SMITH_ZENODO_FILES["cube14v3.pic"].size_bytes == 77_591_894
    assert target.SMITH_ZENODO_FILES["cube14v3.pic"].md5 == (
        "3fd41560b7dbcc60851ef3a0972c852f"
    )
    assert target.NIRSPEC_TABLE.sha256 == (
        "96159824870eb7c8132fd9c8bee30caf9ec95f05e128dbe2fea8627909a40d3f"
    )
    assert target.SMITH_ZENODO_FILES[
        "w77_pre_nirspec_best_fit_R500K_scaled.txt"
    ].md5 == "93841125cab0a2c74ae4730ebf36b7dc"
    assert target.NIRSPEC_PUBLISHED_TABLE_POINTS == 150
    assert target.NIRSPEC_SMITH_TEXT_PREDICTIVE_POINTS == 160
    assert target.NIRSPEC_POINT_COUNT_DISCREPANCY is True
    assert target.NIRSPEC_POINT_COUNT_RECONCILED is True
    assert "reconciled" in target.NIRSPEC_POINT_COUNT_STATUS


def test_comparison_anchors_and_gates_are_cited() -> None:
    joint = target.SMITH_JOINT_ANCHORS
    assert joint["log10_h2o_vmr"].value == pytest.approx(-4.02)
    assert joint["log10_co_vmr"].value == pytest.approx(-3.91)
    assert joint["T0_K"].value == pytest.approx(1400.0)
    assert joint["C_to_O"].value == pytest.approx(0.57)
    assert target.ACCEPTANCE_GATES["resources"].threshold == (
        "Process RSS <2 GiB and LBL opacity RSS <1023 MiB."
    )
    assert "independent ROBERT comparison" in target.INDEPENDENT_COMPARISON_WORDING
    assert len(target.REPRODUCTION_REQUIREMENTS) >= 5
