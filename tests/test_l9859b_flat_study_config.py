"""Configuration contract for the L 98-59 b flat-spectrum CLR study."""

from pathlib import Path

from robert_exoplanets.io.task_config import load_task_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "studies" / "l98_59b_flat_clr" / "configs"


def test_flat_clr_ensembles_share_setup_and_use_physical_h2s_closure() -> None:
    ensemble_a = load_task_config(CONFIGS / "ensemble_a.yaml")
    ensemble_b = load_task_config(CONFIGS / "ensemble_b.yaml")

    assert ensemble_a.bodies == ensemble_b.bodies
    assert ensemble_a.observations.loader == ensemble_b.observations.loader == "robert_npz"
    assert ensemble_a.observations.datasets == ensemble_b.observations.datasets == ("synthetic",)
    assert ensemble_a.atmosphere.chemistry.background_species == ("H2S",)
    assert ensemble_b.atmosphere.chemistry.background_species == ("H2S",)
    assert ensemble_a.atmosphere.chemistry.phantom_species is None
    assert ensemble_b.atmosphere.chemistry.phantom_species is None
    assert ensemble_a.opacity.species == ("H2O", "CO2", "CO", "H2S", "SO2")
    assert ensemble_b.opacity.species == ("H2O", "CO2", "CO", "H2S")
    assert ensemble_a.sampler.engine == ensemble_b.sampler.engine == "multinest"
    assert ensemble_a.sampler.live_points == ensemble_b.sampler.live_points == 50
    assert ensemble_a.runtime.mpi_processes == ensemble_b.runtime.mpi_processes == 3


def test_flat_clr_config_templates_do_not_embed_a_machine_specific_path() -> None:
    for path in CONFIGS.glob("ensemble_*.yaml"):
        contents = path.read_text(encoding="utf-8")
        assert "/Users/" not in contents
        assert "Dropbox" not in contents
        assert "ROBERT_K_TABLE_DIRECTORY" in contents
