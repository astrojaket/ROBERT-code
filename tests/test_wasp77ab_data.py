"""Tests for the published WASP-77Ab JWST/NIRSpec spectrum."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets import (
    AUGUST2023_WASP77AB_SHA256,
    WASP77AB_AUGUST2023_SHA256,
    load_august2023_wasp77ab,
)
from robert_exoplanets.core import RobertDataError
from robert_exoplanets.io.configured_tasks import load_observations
from robert_exoplanets.io.task_config import ObservationsConfig
from robert_exoplanets.io.task_config import TaskConfig, load_task_config


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "jwst_emission_spectra"
TABLE = (
    DATA
    / "spectra"
    / "76"
    / "89"
    / "90"
    / "56"
    / "WASP_77_A_b_3.11569_5329_1.tbl"
)


def test_wasp77ab_loader_converts_depth_and_preserves_metadata() -> None:
    collection = load_august2023_wasp77ab(DATA)

    assert collection.names == ("nirspec_g395h_nrs1", "nirspec_g395h_nrs2")
    assert collection.n_points == 150
    assert [dataset.observation.n_points for dataset in collection.datasets] == [
        70,
        80,
    ]
    observation = collection.datasets[0].observation
    np.testing.assert_allclose(observation.flux[:2], [0.001267, 0.00145])
    np.testing.assert_allclose(observation.uncertainty[:2], [8.85e-5, 8.8e-5])
    assert observation.wavelength_unit == "micron"
    assert observation.flux_unit == "eclipse_depth"
    assert observation.observable == "eclipse_depth"
    assert observation.instrument == "JWST/NIRSpec-G395H-NRS1"
    assert observation.metadata["data_producer"] == "August et al. 2023"
    assert observation.metadata["comparison_input"] == "Smith et al. 2024"
    assert (
        observation.metadata["data_role"]
        == "August et al. 2023 / Smith et al. 2024 input"
    )
    assert observation.metadata["smith_produced_data"] == "false"
    assert observation.metadata["checksum_sha256"] == AUGUST2023_WASP77AB_SHA256
    assert collection.metadata["detector_split_index"] == "70"
    assert collection.metadata["detector_split_after_index"] == "69"
    assert collection.metadata["detector_gap_micron"] == "0.119"
    assert collection.metadata["detector_gap_edges_micron"] == "3.712-3.831"


def test_wasp77ab_loader_uses_bandwidth_edges_and_both_error_sides() -> None:
    collection = load_august2023_wasp77ab(TABLE)
    observation = collection.datasets[0].observation
    error_1 = np.fromstring(
        observation.metadata["published_error_1_percent"], sep=","
    )
    error_2 = np.fromstring(
        observation.metadata["published_error_2_percent"], sep=","
    )
    bandwidth = np.fromstring(
        observation.metadata["published_bandwidths_micron"], sep=","
    )
    np.testing.assert_allclose(
        observation.uncertainty,
        0.01 * 0.5 * (np.abs(error_1) + np.abs(error_2)),
    )
    np.testing.assert_allclose(
        observation.wavelength_bin_edges[0],
        observation.wavelength[0] - 0.5 * bandwidth[0],
    )
    np.testing.assert_allclose(
        observation.wavelength_bin_edges[-1],
        observation.wavelength[-1] + 0.5 * bandwidth[-1],
    )
    assert observation.wavelength_bin_edges.size == observation.n_points + 1
    assert np.all(np.diff(observation.wavelength_bin_edges) > 0.0)
    assert bandwidth.size == observation.n_points
    nrs2 = collection.datasets[1].observation
    np.testing.assert_allclose(nrs2.wavelength_bin_edges[0], 3.831)
    np.testing.assert_allclose(nrs2.wavelength_bin_edges[-1], 5.168)
    assert nrs2.metadata["detector"] == "NRS2"
    assert nrs2.metadata["detector_split_index"] == "70"


def test_wasp77ab_loader_accepts_the_new_config_loader_literal() -> None:
    config = ObservationsConfig(
        loader="august2023_wasp77ab",
        path=DATA,
        datasets=("nirspec_g395h_nrs1", "nirspec_g395h_nrs2"),
    )

    assert config.loader == "august2023_wasp77ab"


def test_configured_task_loader_reads_wasp77ab_archive() -> None:
    base = load_task_config(
        ROOT / "configurations" / "targets/WASP-69b/wasp69b_cloud_free_R1000.yaml"
    )
    raw = deepcopy(base.model_dump(mode="python"))
    raw["observations"]["loader"] = "august2023_wasp77ab"
    raw["observations"]["path"] = DATA
    raw["observations"]["datasets"] = (
        "nirspec_g395h_nrs1",
        "nirspec_g395h_nrs2",
    )
    config = TaskConfig.model_validate(raw)

    observations = load_observations(config)

    assert observations.names == ("nirspec_g395h_nrs1", "nirspec_g395h_nrs2")
    assert observations.n_points == 150


def test_wasp77ab_loader_rejects_a_modified_archive_table(tmp_path: Path) -> None:
    modified = tmp_path / TABLE.name
    modified.write_bytes(TABLE.read_bytes() + b"\n")

    with pytest.raises(RobertDataError, match="checksum mismatch"):
        load_august2023_wasp77ab(modified)

    unchecked = load_august2023_wasp77ab(modified, verify_checksum=False)
    assert unchecked.datasets[0].observation.metadata["checksum_sha256"] != (
        WASP77AB_AUGUST2023_SHA256
    )
