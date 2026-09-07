"""Tests for the molecular opacity distributed with ROBERT."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from robert_exoplanets import (
    BUNDLED_K_TABLE_RESOLUTION,
    BUNDLED_K_TABLE_SPECIES,
    bundled_k_table_directory,
    bundled_k_table_paths,
    bundled_opacity_manifest,
)
from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.io.task_config import load_task_config
from robert_exoplanets.opacity import CorrelatedKOpacityProvider, read_kta_header


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configurations" / "targets/WASP-69b/wasp69b_cloud_free_R1000.yaml"


def test_bundled_r100_manifest_matches_every_distributed_table() -> None:
    manifest = bundled_opacity_manifest()
    products = manifest["products"]

    assert BUNDLED_K_TABLE_RESOLUTION == "R100"
    assert tuple(products) == ("CH4", "CO", "CO2", "H2O", "HCN", "NH3")
    assert set(products) == set(BUNDLED_K_TABLE_SPECIES)
    assert manifest["wavelength_min_micron"] == 0.3
    assert manifest["wavelength_max_micron"] == 15.0
    assert manifest["g_points"] == 8
    assert manifest["license"] == "CC-BY-SA-4.0"

    for species, path in bundled_k_table_paths().items():
        header = read_kta_header(path, checksum=True)
        product = products[species]
        assert header.native_shape == (22, 27, 391, 8)
        assert header.file_size_bytes == product["output_size_bytes"]
        assert header.checksum_sha256 == product["output_sha256"]
        assert _sha256(path) == product["output_sha256"]


def test_bundled_r100_table_loads_as_finite_positive_correlated_k() -> None:
    provider = CorrelatedKOpacityProvider.from_exomol_kta_directory(
        bundled_k_table_directory(),
        species=("H2O",),
        resolution="R100",
        interpolation="log_pressure_temperature_log_k",
    )
    table = provider.tables["H2O"]

    assert table.kcoeff.shape == (22, 27, 391, 8)
    assert np.all(np.isfinite(table.kcoeff))
    assert np.all(table.kcoeff > 0.0)
    assert np.isclose(np.sum(table.g_weights), 1.0)
    assert np.min(table.wavelength_micron) > 0.3
    assert np.max(table.wavelength_micron) < 15.0


def test_r100_yaml_uses_bundled_tables_when_external_path_is_omitted(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["paths"].pop("k_table_directory")
    raw["opacity"]["resolution"] = "R100"
    raw["opacity"]["species"] = ["H2O", "CO", "CO2", "CH4", "NH3"]
    path = tmp_path / "r100.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_task_config(path)

    assert config.opacity.path == bundled_k_table_directory()
    assert config.opacity.resolution == "R100"


def test_bundled_path_api_rejects_unavailable_species() -> None:
    with pytest.raises(RobertValidationError, match="SO2"):
        bundled_k_table_paths(["H2O", "SO2"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
