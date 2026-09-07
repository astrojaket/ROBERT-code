"""Contracts for the new-user R=100 injection-recovery workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robert_exoplanets import bundled_k_table_directory
from robert_exoplanets.io.task_config import load_task_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGURATIONS = {
    "emission": ROOT / "configurations" / "quickstart/r100_emission_validation.yaml",
    "transmission": ROOT / "configurations" / "quickstart/r100_transmission_validation.yaml",
}
REFERENCE = ROOT / "data" / "validation" / "r100_quickstart"
NOTEBOOK = (
    ROOT / "examples" / "notebooks" / "r100_emission_transmission_validation.ipynb"
)


@pytest.mark.parametrize("model", ("emission", "transmission"))
def test_r100_validation_configuration_is_local_and_identifiable(model: str) -> None:
    config = load_task_config(CONFIGURATIONS[model])

    assert config.radiative_transfer.model == model
    assert config.opacity.resolution == "R100"
    assert config.opacity.path == bundled_k_table_directory()
    assert config.opacity.species == ("H2O",)
    assert tuple(item.name for item in config.parameters) == ("log_H2O",)
    assert config.sampler.engine == "multinest"
    assert config.sampler.live_points == 60
    assert config.sampler.multinest_max_iterations == 0
    assert config.runtime.mpi_processes == "auto"


@pytest.mark.parametrize("model", ("emission", "transmission"))
def test_r100_reference_recovery_contains_truth_at_95_percent(model: str) -> None:
    report = json.loads(
        (REFERENCE / f"{model}_recovery.json").read_text(encoding="utf-8")
    )
    recovery = report["parameter_recoveries"]["log_H2O"]

    assert report["passed"] is True
    assert report["inference_converged"] is True
    assert report["fit_passed"] is True
    assert recovery["q02_5"] <= recovery["truth"] <= recovery["q97_5"]


def test_r100_validation_notebook_is_valid_python() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", ())) for cell in notebook["cells"]]
    code = [
        source
        for cell, source in zip(notebook["cells"], sources, strict=True)
        if cell["cell_type"] == "code"
    ]

    for index, source in enumerate(code):
        compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")
