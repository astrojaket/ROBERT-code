"""Production-safety tests for long-running retrievals."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robert_exoplanets import (
    Observation,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    Spectrum,
    UniformPrior,
    load_retrieval_status,
    run_retrieval,
)
from robert_exoplanets.core import RobertConfigError, RobertDataError
from robert_exoplanets.retrieval.manifest import build_run_manifest
from robert_exoplanets.retrieval.runner import (
    _acquire_run_directory_lock,
    _prepare_manifest,
    _release_run_directory_lock,
)
from robert_exoplanets.retrieval.status import (
    append_retrieval_attempt_event,
    main as status_main,
    write_retrieval_status,
)


def _problem(*, name: str = "production-test") -> RetrievalProblem:
    observation = Observation.from_arrays(
        wavelength=[1.0, 2.0],
        flux=[1.0, 1.0],
        uncertainty=[0.1, 0.1],
    )
    parameters = RetrievalParameterSet(
        (RetrievalParameter("baseline", UniformPrior(0.0, 2.0)),)
    )
    return RetrievalProblem(
        name=name,
        observation=observation,
        parameters=parameters,
        forward_model=lambda p: Spectrum.from_arrays(
            observation.wavelength,
            np.full(observation.n_points, p["baseline"]),
            unit=observation.flux_unit,
            observable=observation.observable,
            wavelength_unit=observation.wavelength_unit,
        ),
        opacity_identifiers={"H2O": "sha256:test"},
    )


def _manifest(problem: RetrievalProblem, *, max_ncalls: int, floor: float = -1.0e100):
    return build_run_manifest(
        problem,
        method="ultranest",
        settings={
            "method": "ultranest",
            "max_ncalls": max_ncalls,
            "min_num_live_points": 40,
            "resume": "resume",
            "invalid_loglike_floor": floor,
        },
        random_seed=42,
    )


def test_run_directory_lock_rejects_concurrent_writer(tmp_path) -> None:
    first = _acquire_run_directory_lock(tmp_path)
    try:
        with pytest.raises(RobertConfigError, match="already in use"):
            _acquire_run_directory_lock(tmp_path)
    finally:
        _release_run_directory_lock(first)

    second = _acquire_run_directory_lock(tmp_path)
    _release_run_directory_lock(second)


def test_resume_preserves_manifest_but_records_changed_call_budget(tmp_path) -> None:
    problem = _problem()
    original = _prepare_manifest(
        _manifest(problem, max_ncalls=500),
        tmp_path,
        resume="resume",
        is_nested=True,
    )
    resumed = _prepare_manifest(
        _manifest(problem, max_ncalls=1000),
        tmp_path,
        resume="resume",
        is_nested=True,
    )

    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    attempts = sorted((tmp_path / "attempts").glob("attempt-*.json"))
    assert resumed.config_hash == original.config_hash
    assert saved["settings"]["max_ncalls"] == 500
    assert len(attempts) == 2
    latest = json.loads(attempts[-1].read_text(encoding="utf-8"))
    assert latest["settings"]["max_ncalls"] == 1000
    assert latest["original_config_hash"] == original.config_hash


def test_resume_rejects_changed_scientific_definition(tmp_path) -> None:
    problem = _problem()
    _prepare_manifest(
        _manifest(problem, max_ncalls=500),
        tmp_path,
        resume="resume",
        is_nested=True,
    )

    with pytest.raises(RobertConfigError, match="invalid_loglike_floor"):
        _prepare_manifest(
            _manifest(problem, max_ncalls=1000, floor=-1.0e90),
            tmp_path,
            resume="resume",
            is_nested=True,
        )


def test_status_is_atomic_journalled_and_cli_readable(tmp_path, capsys) -> None:
    append_retrieval_attempt_event(tmp_path, {"attempt_id": "one", "event": "started"})
    write_retrieval_status(
        tmp_path,
        {
            "attempt_id": "one",
            "state": "running",
            "ncall": 250,
            "ncall_this_attempt": 50,
            "calls_per_second": 2.5,
        },
    )

    status = load_retrieval_status(tmp_path)
    assert status["state"] == "running"
    assert status["attempt_count"] == 1
    assert status["ncall"] == 250
    assert status_main([str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "State:        running" in output
    assert "Calls:        250" in output


def test_status_rejects_empty_directory(tmp_path) -> None:
    with pytest.raises(RobertDataError, match="no retrieval status"):
        load_retrieval_status(tmp_path)


def test_real_ultranest_checkpoint_can_be_resumed(tmp_path) -> None:
    pytest.importorskip("ultranest")
    problem = _problem(name="real-resume-test")
    first = run_retrieval(
        problem,
        method="ultranest",
        output_dir=tmp_path,
        min_num_live_points=40,
        max_ncalls=200,
        dlogz=1.5,
        resume="overwrite",
        show_status=False,
        seed=7,
    )
    first_calls = int(first.metadata["ncall"])

    second = run_retrieval(
        problem,
        method="ultranest",
        output_dir=tmp_path,
        min_num_live_points=40,
        max_ncalls=first_calls + 200,
        dlogz=1.5,
        resume="resume",
        show_status=False,
        seed=7,
    )
    status = load_retrieval_status(tmp_path)

    assert int(second.metadata["ncall"]) > first_calls
    assert int(second.metadata["ncall_start"]) == first_calls
    assert status["attempt_count"] == 2
    assert status["ncall_checkpointed"] >= int(second.metadata["ncall"])
