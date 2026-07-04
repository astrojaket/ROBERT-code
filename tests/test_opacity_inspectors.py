"""Tests for lightweight opacity file inspectors."""

from __future__ import annotations

from pathlib import Path

from robert_exoplanets.opacity import (
    OpacityDataSource,
    OpacityMode,
    OpacityStorageFormat,
    inspect_exomol_directory,
    inspect_hitran_cia_file,
    inspect_hitran_par_file,
    inspect_kta_file,
)


def test_inspect_kta_file_records_exomol_op_source_and_nemesis_storage(tmp_path: Path) -> None:
    path = tmp_path / "H2O_emission_R1000.kta"
    path.write_bytes(b"synthetic-k-table")

    product = inspect_kta_file(path)

    assert product.species == ("H2O",)
    assert product.mode == OpacityMode.CORRELATED_K
    assert product.source == OpacityDataSource.EXOMOL_OP
    assert product.storage_format == OpacityStorageFormat.NEMESIS_KTA
    assert product.file_size_bytes == len(b"synthetic-k-table")
    assert product.checksum_sha256


def test_inspect_hitran_par_file_reads_line_center_range(tmp_path: Path) -> None:
    path = tmp_path / "CO.par"
    path.write_text(
        _hitran_line(molecule_id=5, isotopologue_id=1, wavenumber=1234.5)
        + _hitran_line(molecule_id=5, isotopologue_id=1, wavenumber=1250.0),
        encoding="utf-8",
    )

    product = inspect_hitran_par_file(path, species="CO")

    assert product.species == ("CO",)
    assert product.mode == OpacityMode.LINE_BY_LINE
    assert product.source == OpacityDataSource.HITRAN
    assert product.storage_format == OpacityStorageFormat.HITRAN_PAR
    assert product.spectral_coverage is not None
    assert product.spectral_coverage.min_value == 1234.5
    assert product.spectral_coverage.max_value == 1250.0
    assert product.metadata["molecule_ids"] == "5"


def test_inspect_hitran_cia_file_reads_header_ranges(tmp_path: Path) -> None:
    path = tmp_path / "H2-He_2011.cia"
    path.write_text(
        "H2-He 20.0 100.0 200.0 3 7 synthetic header\n"
        "20.0 1.0e-50\n"
        "H2-He 30.0 200.0 300.0 3 7 synthetic header\n"
        "30.0 2.0e-50\n",
        encoding="utf-8",
    )

    product = inspect_hitran_cia_file(path)

    assert product.species == ("H2", "He")
    assert product.mode == OpacityMode.CIA
    assert product.storage_format == OpacityStorageFormat.HITRAN_CIA
    assert product.spectral_coverage is not None
    assert product.spectral_coverage.min_value == 20.0
    assert product.spectral_coverage.max_value == 200.0
    assert product.grid_coverage is not None
    assert product.grid_coverage.temperature_min == 200.0
    assert product.grid_coverage.temperature_max == 300.0
    assert product.metadata["header_sets"] == "2"


def test_inspect_exomol_directory_discovers_line_list_and_k_table(tmp_path: Path) -> None:
    (tmp_path / "H2O.states").write_text("states", encoding="utf-8")
    (tmp_path / "H2O.trans").write_text("trans", encoding="utf-8")
    (tmp_path / "H2O.pf").write_text("pf", encoding="utf-8")
    (tmp_path / "H2O_R1000.kta").write_bytes(b"kta")

    database = inspect_exomol_directory(tmp_path, species="H2O")

    modes = {product.mode for product in database.products}
    assert OpacityMode.LINE_BY_LINE in modes
    assert OpacityMode.CORRELATED_K in modes
    k_products = [product for product in database.products if product.mode == OpacityMode.CORRELATED_K]
    assert k_products[0].source == OpacityDataSource.EXOMOL_OP
    assert k_products[0].storage_format == OpacityStorageFormat.NEMESIS_KTA


def _hitran_line(*, molecule_id: int, isotopologue_id: int, wavenumber: float) -> str:
    return f"{molecule_id:>2}{isotopologue_id:1d}{wavenumber:12.6f}{1.0e-20:10.3E}\n"
