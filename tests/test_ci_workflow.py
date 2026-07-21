"""Static checks for maintained paths in the GitHub Actions workflow."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def test_ci_example_smoke_checks_reference_maintained_files() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    example_paths = re.findall(r"python (examples/[^\s]+\.py)", text)
    assert example_paths
    for relative_path in example_paths:
        assert (ROOT / relative_path).is_file(), relative_path

    assert "Depreciated_Benchmarks" not in text
    assert "examples/build_atmosphere_from_config.py" not in text


def test_ci_validates_the_generalized_yaml_workflow() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "python run_retrieval.py" in text
    assert "configurations/wasp69b_cloud_free_R1000.yaml" in text
    assert "--validate-only" in text


def test_ci_quality_job_exercises_optional_diagnostics() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    quality_job = text.split("  test:", maxsplit=1)[0]
    assert "[dev,perf,opacity,diagnostics]" in quality_job
