"""Memory-lifecycle checks for configured KTA opacity preparation."""

from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np

from robert_exoplanets import CorrelatedKOpacityProvider, CorrelatedKTable
from robert_exoplanets.instruments import (
    Observation,
    ObservationCollection,
    ObservationDataset,
)
from robert_exoplanets.io import configured_tasks
from robert_exoplanets.io.configured_tasks import prepare_opacity
from robert_exoplanets.io.task_config import load_task_config


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configurations" / "emission.yaml"


def test_kta_preparation_releases_each_native_provider_and_reuses_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_task_config(EXAMPLE)
    cache = tmp_path / "opacity_cache"
    config = config.model_copy(
        update={
            "opacity": config.opacity.model_copy(
                update={
                    "species": ("H2O", "CO"),
                    "path": tmp_path / "source_tables",
                    "cache_directory": cache,
                }
            )
        }
    )
    observation = Observation.from_arrays(
        wavelength=[1.25, 2.25],
        wavelength_bin_edges=[1.0, 1.5, 2.5],
        flux=[0.01, 0.02],
        uncertainty=[0.001, 0.001],
        flux_unit="eclipse_depth",
        observable="eclipse_depth",
    )
    observations = ObservationCollection(
        datasets=(
            ObservationDataset(name="synthetic", observation=observation),
        )
    )
    provider_refs: list[weakref.ReferenceType] = []
    table_refs: list[weakref.ReferenceType] = []
    factory_calls: list[tuple[str, bool]] = []
    binning_calls: list[str] = []

    def fake_from_kta_directory(
        directory,
        *,
        species,
        resolution,
        name,
        interpolation,
        nonfinite_policy,
        cache_log_kcoeff,
    ):
        gc.collect()
        if provider_refs:
            assert provider_refs[-1]() is None
            assert table_refs[-1]() is None
        assert cache_log_kcoeff is False
        species_name = species[0]
        factory_calls.append((species_name, cache_log_kcoeff))
        table = CorrelatedKTable(
            species=species_name,
            pressure_bar=np.array([1.0e-6, 1.0]),
            temperature_K=np.array([500.0, 1500.0]),
            wavenumber_cm_inverse=np.array([5000.0, 10000.0]),
            g_samples=np.array([0.25, 0.75]),
            g_weights=np.array([0.5, 0.5]),
            kcoeff=np.full((2, 2, 2, 2), 1.0e-24),
            metadata={
                "source_path": str(tmp_path / f"{species_name}_R1000.kta"),
                "checksum_sha256": f"source-{species_name}",
            },
        )
        provider = CorrelatedKOpacityProvider(
            tables={species_name: table},
            name=name,
            interpolation=interpolation,
            cache_log_kcoeff=cache_log_kcoeff,
        )
        assert not provider._log_kcoeff
        provider_refs.append(weakref.ref(provider))
        table_refs.append(weakref.ref(table))
        return provider

    def fake_bin_to_spectral_grid(self, spectral_grid, **kwargs):
        species_name = next(iter(self.tables))
        binning_calls.append(species_name)
        return SimpleNamespace(tables=self.tables)

    monkeypatch.setattr(
        configured_tasks.CorrelatedKOpacityProvider,
        "from_exomol_kta_directory",
        staticmethod(fake_from_kta_directory),
    )
    monkeypatch.setattr(
        CorrelatedKOpacityProvider,
        "bin_to_spectral_grid",
        fake_bin_to_spectral_grid,
    )

    prepare_opacity(config, observations)

    expected_paths = [
        cache / "R1000" / f"synthetic_{species}.npz"
        for species in config.opacity.species
    ]
    for species, path in zip(config.opacity.species, expected_paths, strict=True):
        assert path.is_file()
        with np.load(path, allow_pickle=False) as saved:
            assert str(saved["species"]) == species
            assert str(saved["source_sha256"]) == f"source-{species}"
            assert str(saved["spectral_preparation"]) == "exo_k_bin_down_cp"
            assert saved["kcoeff"].shape == (2, 2, 2, 2)
            assert int(saved["opacity_cache_schema_version"]) == 2
    assert binning_calls == ["H2O", "CO"]
    mtimes = [path.stat().st_mtime_ns for path in expected_paths]

    prepare_opacity(config, observations)

    assert binning_calls == ["H2O", "CO"]
    assert [path.stat().st_mtime_ns for path in expected_paths] == mtimes
    assert factory_calls == [
        ("H2O", False),
        ("CO", False),
        ("H2O", False),
        ("CO", False),
    ]
