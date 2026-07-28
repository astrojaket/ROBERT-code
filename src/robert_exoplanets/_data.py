"""Locations and provenance for data distributed with ROBERT."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from robert_exoplanets.core import RobertDataError, RobertValidationError


BUNDLED_K_TABLE_RESOLUTION = "R100"
BUNDLED_K_TABLE_SPECIES = ("H2O", "CO", "CO2", "CH4", "NH3", "HCN")


def bundled_k_table_directory() -> Path:
    """Return the installed directory containing ROBERT's R=100 k-tables."""

    resource = files("robert_exoplanets").joinpath(
        f"data/opacities/{BUNDLED_K_TABLE_RESOLUTION}"
    )
    directory = Path(str(resource))
    if not directory.is_dir():
        raise RobertDataError(
            "the installed robert-exoplanets distribution does not contain "
            f"the bundled {BUNDLED_K_TABLE_RESOLUTION} opacity directory"
        )
    return directory


def bundled_k_table_paths(
    species: tuple[str, ...] | list[str] | None = None,
) -> Mapping[str, Path]:
    """Return installed R=100 k-table paths for selected bundled molecules."""

    selected = BUNDLED_K_TABLE_SPECIES if species is None else tuple(species)
    if not selected or any(not str(item).strip() for item in selected):
        raise RobertValidationError("bundled opacity species must not be empty")
    unsupported = tuple(
        item for item in selected if item not in BUNDLED_K_TABLE_SPECIES
    )
    if unsupported:
        raise RobertValidationError(
            "no bundled R=100 opacity table for: " + ", ".join(unsupported)
        )
    directory = bundled_k_table_directory()
    paths = {
        item: directory / f"{item}_{BUNDLED_K_TABLE_RESOLUTION}.kta"
        for item in selected
    }
    missing = tuple(item for item, path in paths.items() if not path.is_file())
    if missing:
        raise RobertDataError(
            "the installed R=100 opacity set is incomplete: " + ", ".join(missing)
        )
    return MappingProxyType(paths)


def bundled_opacity_manifest() -> Mapping[str, object]:
    """Load the immutable provenance manifest for bundled molecular opacity."""

    path = bundled_k_table_directory() / "provenance.json"
    if not path.is_file():
        raise RobertDataError("the bundled opacity provenance manifest is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RobertDataError("the bundled opacity provenance manifest is invalid")
    return MappingProxyType(payload)


__all__ = [
    "BUNDLED_K_TABLE_RESOLUTION",
    "BUNDLED_K_TABLE_SPECIES",
    "bundled_k_table_directory",
    "bundled_k_table_paths",
    "bundled_opacity_manifest",
]
