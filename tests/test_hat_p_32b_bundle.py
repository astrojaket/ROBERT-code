"""Regression checks for the clone-local HAT-P-32b benchmark bundle."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from examples.hat_p_32b_fastchem_config import (
    BUNDLE_ROOT,
    FASTCHEM_PATH,
    OBSERVATION_NPZ,
    OPACITY_ARCHIVE_DIR,
    OPACITY_SPECIES,
    make_model_config,
)
from robert_exoplanets import CorrelatedKOpacityProvider


def test_hat_p_32b_bundle_checksums_are_complete_and_valid() -> None:
    checksum_path = BUNDLE_ROOT / "checksums.sha256"
    records = [line.split("  ", maxsplit=1) for line in checksum_path.read_text().splitlines()]

    assert records
    for expected, relative_path in records:
        path = BUNDLE_ROOT / relative_path
        assert path.is_file(), relative_path
        assert _file_sha256(path) == expected


def test_hat_p_32b_configuration_uses_bundled_fastchem_and_opacities() -> None:
    assert OBSERVATION_NPZ.is_file()
    assert (FASTCHEM_PATH / "input/element_abundances/asplund_2009.dat").is_file()
    assert (FASTCHEM_PATH / "input/logK/logK.dat").is_file()
    assert (FASTCHEM_PATH / "FASTCHEM_LICENSE.md").is_file()

    config = make_model_config()

    assert isinstance(config.opacity_source, CorrelatedKOpacityProvider)
    assert config.opacity_binning is None
    assert config.opacity_source.species == OPACITY_SPECIES
    for species, table in config.opacity_source.tables.items():
        assert table.kcoeff.shape == (22, 27, 117, 20), species
        assert Path(table.metadata["source_path"]).name == f"{species}.robert-opacity.npz"
        assert (OPACITY_ARCHIVE_DIR / f"{species}.robert-opacity.npz").is_file()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
