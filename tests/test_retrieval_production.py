"""Production-safety tests for long-running retrievals."""

from __future__ import annotations

import json
import os
from typing import cast

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
from robert_exoplanets.retrieval.priors import Prior
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


def _problem(
    *,
    name: str = "production-test",
    flux: tuple[float, ...] = (1.0, 1.0),
) -> RetrievalProblem:
    observation = Observation.from_arrays(
        wavelength=[1.0, 2.0],
        flux=flux,
        uncertainty=[0.1, 0.1],
    )
    parameters = RetrievalParameterSet(
        (
            RetrievalParameter(
                "baseline",
                cast(Prior, UniformPrior(0.0, 2.0)),
            ),
        )
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


def _manifest(
    problem: RetrievalProblem,
    *,
    max_iter: int,
    floor: float = -1.0e100,
    n_live_points: int = 40,
    multimodal: bool = True,
    evidence_tolerance: float = 0.5,
    verbose: bool = True,
    n_iter_before_update: int = 100,
):
    return build_run_manifest(
        problem,
        method="multinest",
        settings={
            "method": "multinest",
            "max_iter": max_iter,
            "n_live_points": n_live_points,
            "multimodal": multimodal,
            "evidence_tolerance": evidence_tolerance,
            "verbose": verbose,
            "n_iter_before_update": n_iter_before_update,
            "resume": True,
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


def test_resume_preserves_manifest_but_records_changed_iteration_budget(tmp_path) -> None:
    problem = _problem()
    original = _prepare_manifest(
        _manifest(problem, max_iter=500),
        tmp_path,
        resume=True,
        is_nested=True,
    )
    resumed = _prepare_manifest(
        _manifest(problem, max_iter=1000),
        tmp_path,
        resume=True,
        is_nested=True,
    )

    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    attempts = sorted((tmp_path / "attempts").glob("attempt-*.json"))
    assert resumed.config_hash == original.config_hash
    assert saved["settings"]["max_iter"] == 500
    assert len(attempts) == 2
    latest = json.loads(attempts[-1].read_text(encoding="utf-8"))
    assert latest["settings"]["max_iter"] == 1000
    assert latest["original_config_hash"] == original.config_hash


def test_resume_rejects_changed_scientific_definition(tmp_path) -> None:
    problem = _problem()
    _prepare_manifest(
        _manifest(problem, max_iter=500),
        tmp_path,
        resume=True,
        is_nested=True,
    )

    with pytest.raises(RobertConfigError, match="invalid_loglike_floor"):
        _prepare_manifest(
            _manifest(problem, max_iter=1000, floor=-1.0e90),
            tmp_path,
            resume=True,
            is_nested=True,
        )


def test_resume_rejects_changed_observation_data(tmp_path) -> None:
    original_problem = _problem()
    original = _prepare_manifest(
        _manifest(original_problem, max_iter=500),
        tmp_path,
        resume=True,
        is_nested=True,
    )
    changed_problem = _problem(flux=(1.0, 1.1))
    changed = _manifest(changed_problem, max_iter=1000)

    assert changed.config_hash != original.config_hash
    with pytest.raises(RobertConfigError, match="observation_identity"):
        _prepare_manifest(
            changed,
            tmp_path,
            resume=True,
            is_nested=True,
        )


def test_resume_normalizes_omitted_and_explicit_multinest_defaults(tmp_path) -> None:
    problem = _problem()
    omitted = build_run_manifest(
        problem,
        method="multinest",
        settings={"method": "multinest"},
        random_seed=42,
    )
    explicit = build_run_manifest(
        problem,
        method="nested",
        settings={
            "method": "nested",
            "n_live_points": 400,
            "max_iter": 0,
            "evidence_tolerance": 0.5,
            "sampling_efficiency": 0.8,
            "resume": True,
            "verbose": True,
            "mpi_nprocs": None,
            "seed": None,
            "invalid_loglike_floor": -1.0e100,
            "importance_nested_sampling": True,
            "multimodal": True,
            "n_iter_before_update": 100,
        },
        random_seed=42,
    )

    assert omitted.settings == explicit.settings
    assert omitted.config_hash == explicit.config_hash
    _prepare_manifest(omitted, tmp_path, resume=True, is_nested=True)
    _prepare_manifest(explicit, tmp_path, resume=True, is_nested=True)
    assert len(tuple((tmp_path / "attempts").glob("attempt-*.json"))) == 2


def test_resume_rejects_changed_multinest_structure(tmp_path) -> None:
    problem = _problem()
    _prepare_manifest(
        _manifest(problem, max_iter=0, n_live_points=40),
        tmp_path,
        resume=True,
        is_nested=True,
    )

    with pytest.raises(RobertConfigError, match="settings.n_live_points"):
        _prepare_manifest(
            _manifest(problem, max_iter=100, n_live_points=50),
            tmp_path,
            resume=True,
            is_nested=True,
        )

    with pytest.raises(RobertConfigError, match="settings.multimodal"):
        _prepare_manifest(
            _manifest(problem, max_iter=100, n_live_points=40, multimodal=False),
            tmp_path,
            resume=True,
            is_nested=True,
        )


def test_resume_allows_per_attempt_stopping_and_logging_settings(tmp_path) -> None:
    problem = _problem()
    original = _prepare_manifest(
        _manifest(problem, max_iter=500),
        tmp_path,
        resume=True,
        is_nested=True,
    )
    resumed = _prepare_manifest(
        _manifest(
            problem,
            max_iter=1000,
            evidence_tolerance=1.0,
            verbose=False,
            n_iter_before_update=200,
        ),
        tmp_path,
        resume=True,
        is_nested=True,
    )

    assert resumed.config_hash == original.config_hash
    attempts = sorted((tmp_path / "attempts").glob("attempt-*.json"))
    assert len(attempts) == 2
    latest = json.loads(attempts[-1].read_text(encoding="utf-8"))
    assert latest["settings"]["evidence_tolerance"] == 1.0
    assert latest["settings"]["verbose"] is False
    assert latest["settings"]["n_iter_before_update"] == 200

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


@pytest.mark.skipif(
    os.environ.get("ROBERT_RUN_MULTINEST_SMOKE") != "1",
    reason="set ROBERT_RUN_MULTINEST_SMOKE=1 for the native conda smoke test",
)
def test_real_multinest_checkpoint_can_be_resumed(tmp_path) -> None:
    pytest.importorskip("pymultinest")
    problem = _problem(name="real-resume-test")
    first = run_retrieval(
        problem,
        output_dir=tmp_path,
        n_live_points=30,
        max_iter=120,
        evidence_tolerance=1.0,
        resume=False,
        verbose=False,
        seed=7,
    )
    second = run_retrieval(
        problem,
        output_dir=tmp_path,
        n_live_points=30,
        max_iter=240,
        evidence_tolerance=1.0,
        resume=True,
        verbose=False,
        seed=7,
    )
    status = load_retrieval_status(tmp_path)

    assert second.metadata["attempt_id"] != first.metadata["attempt_id"]
    assert status["attempt_count"] == 2
