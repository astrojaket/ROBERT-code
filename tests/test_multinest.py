"""Tests for the optional PyMultiNest adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robert_exoplanets import (
    Observation,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    Spectrum,
    UniformPrior,
    run_multinest,
)
from robert_exoplanets.core import RobertConfigError, RobertDataError
from robert_exoplanets.retrieval.samplers.callback_supervisor import (
    CALLBACK_FAILURE_MARKER_ENV,
    CALLBACK_SUPERVISED_ENV,
)
from robert_exoplanets.retrieval.samplers.multinest import (
    MULTINEST_MAX_SEED,
    _effective_multinest_seed,
    _global_evidence,
    _native_likelihood_evaluations,
)


def _problem() -> RetrievalProblem:
    observation = Observation.from_arrays(
        wavelength=[1.0, 2.0],
        flux=[1.0, 1.0],
        uncertainty=[0.1, 0.1],
    )

    def forward(parameters):
        return Spectrum.from_arrays(
            observation.wavelength,
            np.full(observation.n_points, parameters["level"]),
            unit="eclipse_depth",
            observable="eclipse_depth",
        )

    return RetrievalProblem(
        name="multinest-smoke",
        observation=observation,
        parameters=RetrievalParameterSet(
            (RetrievalParameter("level", UniformPrior(0.0, 2.0)),)
        ),
        forward_model=forward,
    )


class _Analyzer:
    def __init__(self, n_params, outputfiles_basename, verbose):
        assert n_params == 1
        assert outputfiles_basename == "chains/1-"

    def get_data(self):
        return np.array([[0.25, 2.0, 0.8], [0.75, 0.0, 1.0]])

    def get_stats(self):
        return {"global evidence": -1.2, "global evidence error": 0.1}

    def get_best_fit(self):
        return {"log_likelihood": 0.0, "parameters": [1.0]}


def test_multinest_adapter_transforms_prior_and_returns_common_result(
    monkeypatch, tmp_path
) -> None:
    calls = {}

    def fake_run(loglike, prior, ndim, **kwargs):
        cube = [0.5]
        prior(cube, ndim, ndim)
        calls["cube"] = tuple(cube)
        calls["loglike"] = loglike(cube, ndim, ndim, -1.0e100)
        calls["kwargs"] = kwargs
        kwargs["dump_callback"](
            2, 20, 1, np.empty((20, 2)), np.empty((2, 3)), (), 0.0, -1.2, 0.1, 0
        )

    monkeypatch.setitem(
        sys.modules,
        "pymultinest",
        SimpleNamespace(run=fake_run, Analyzer=_Analyzer),
    )

    result = run_multinest(
        _problem(),
        output_dir=tmp_path,
        n_live_points=20,
        mpi_nprocs=1,
        seed=4,
        verbose=False,
    )

    assert calls["cube"] == (1.0,)
    assert calls["loglike"] == 0.0
    assert calls["kwargs"]["seed"] == 4
    assert result.method == "multinest"
    assert result.converged
    assert result.log_evidence == -1.2
    assert result.best_fit_parameters == {"level": 1.0}
    assert result.metadata["likelihood_evaluations"] == "1"
    np.testing.assert_allclose(result.weights, [0.25, 0.75])
    assert (tmp_path / "chains" / "1-params.json").is_file()
    assert (tmp_path / "sampler_status.json").is_file()


@pytest.mark.parametrize("fatal_site", ["prior", "likelihood", "dump"])
def test_multinest_writes_callback_marker_and_discards_partial_run(
    monkeypatch, tmp_path, fatal_site
) -> None:
    marker = tmp_path / "callback-failure.json"
    calls = {"likelihood": None, "analyzer": False}

    class FailingProblem:
        parameter_names = ("x",)
        ndim = 1

        def prior_transform(self, cube):
            if fatal_site == "prior":
                raise RuntimeError("prior callback failed")
            return np.asarray(cube, dtype=float)

        def log_likelihood_from_vector(self, vector):
            del vector
            if fatal_site == "likelihood":
                raise RuntimeError("likelihood callback failed")
            return -1.0

    def fake_run(loglike, prior, ndim, **kwargs):
        cube = [0.5]
        prior(cube, ndim, ndim)
        calls["likelihood"] = loglike(cube, ndim, ndim, -1.0e100)
        if fatal_site == "dump":
            kwargs["dump_callback"](
                1,
                10,
                1,
                np.empty((10, 2)),
                np.empty((1, 3)),
                (),
                -1.0,
                -1.0,
                0.1,
                0,
            )

    class UnexpectedAnalyzer:
        def __init__(self, **kwargs):
            del kwargs
            calls["analyzer"] = True

    monkeypatch.setitem(
        sys.modules,
        "pymultinest",
        SimpleNamespace(run=fake_run, Analyzer=UnexpectedAnalyzer),
    )
    monkeypatch.setenv(CALLBACK_FAILURE_MARKER_ENV, str(marker))
    monkeypatch.setenv(CALLBACK_SUPERVISED_ENV, "1")

    def failing_dump_callback(*args):
        del args
        raise RuntimeError("dump callback failed")

    run_kwargs = {"dump_callback": failing_dump_callback} if fatal_site == "dump" else {}
    with pytest.raises(RobertDataError, match="callback failed"):
        run_multinest(
            FailingProblem(),
            output_dir=tmp_path / "multinest",
            n_live_points=10,
            mpi_nprocs=1,
            verbose=False,
            **run_kwargs,
        )

    expected_loglike = -1.0 if fatal_site == "dump" else -1.0e100
    assert calls["likelihood"] == expected_loglike
    assert calls["analyzer"] is False
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["exception_type"] == "RuntimeError"
    assert payload["output_dir"] == str((tmp_path / "multinest").resolve())
    status = json.loads(
        (tmp_path / "multinest" / "sampler_status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "failed"
    assert status["multinest_resume_safe"] is False
    assert status["partial_chains_preserved"] is True
    assert not (tmp_path / "multinest" / "result.json").exists()


def test_multinest_refuses_to_resume_after_callback_failure(tmp_path) -> None:
    output_dir = tmp_path / "multinest"
    output_dir.mkdir()
    (output_dir / "sampler_status.json").write_text(
        json.dumps(
            {
                "state": "failed",
                "callback_failure": True,
                "multinest_resume_safe": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RobertConfigError, match="unsafe to resume"):
        run_multinest(
            _problem(),
            output_dir=output_dir,
            mpi_nprocs=1,
        )


def test_multinest_rejects_negative_seed(tmp_path) -> None:
    with pytest.raises(RobertConfigError, match="non-negative"):
        run_multinest(
            _problem(),
            output_dir=tmp_path,
            seed=-1,
        )


def test_multinest_rejects_unsupported_resume_mode(tmp_path) -> None:
    with pytest.raises(RobertConfigError, match="resume must be"):
        run_multinest(
            _problem(),
            output_dir=tmp_path,
            resume="resume-similar",
        )


def test_multinest_maps_requested_seed_into_native_randomns_range() -> None:
    assert _effective_multinest_seed(None) == -1
    assert _effective_multinest_seed(0) == 0
    assert _effective_multinest_seed(MULTINEST_MAX_SEED) == MULTINEST_MAX_SEED
    assert _effective_multinest_seed(MULTINEST_MAX_SEED + 1) == 0
    assert _effective_multinest_seed(358_737_206) == 21_281


def test_multinest_global_evidence_ignores_malformed_mode_errors(
    tmp_path,
) -> None:
    stats = tmp_path / "1-stats.dat"
    stats.write_text(
        "Nested Sampling Global Log-Evidence : -1.3 +/- 0.2\n"
        "Nested Importance Sampling Global Log-Evidence : -1.2 +/- 0.1\n"
        "Strictly Local Log-Evidence : -1.1 +/- 0.197626258336498618-322\n",
        encoding="utf-8",
    )

    class BrokenModeAnalyzer:
        def get_stats(self):
            raise AssertionError("per-mode statistics must not be parsed")

    evidence, error = _global_evidence(BrokenModeAnalyzer(), Path(stats))

    assert evidence == pytest.approx(-1.2)
    assert error == pytest.approx(0.1)


@pytest.mark.parametrize(
    "importance_line",
    [
        "Nested Importance Sampling Global Log-Evidence: "
        "0.101855797978190647-311 +/- 0.695273390074996372-309",
        "Nested Importance Sampling Global Log-Evidence: "
        "0.101855797978190647-311 +/- 0.1",
        "Nested Importance Sampling Global Log-Evidence: "
        "2300.66686647193183 +/- 0.695273390074996372-309",
    ],
)
def test_multinest_global_evidence_rejects_subnormal_importance_values(
    tmp_path, importance_line
) -> None:
    stats = tmp_path / "1-stats.dat"
    stats.write_text(
        "Nested Sampling Global Log-Evidence: "
        "2300.66686647193183 +/- 0.435589723883032853\n"
        f"{importance_line}\n",
        encoding="utf-8",
    )

    class AnalyzerMustNotBeUsed:
        def get_stats(self):
            raise AssertionError("stats file should be read directly")

    with pytest.raises(RobertConfigError, match="subnormal"):
        _global_evidence(AnalyzerMustNotBeUsed(), Path(stats))


def test_multinest_reads_native_likelihood_count_from_resume_header(
    tmp_path,
) -> None:
    resume = tmp_path / "1-resume.dat"
    resume.write_text(
        " F\n         500         514           1         400\n",
        encoding="utf-8",
    )

    assert _native_likelihood_evaluations(resume) == 514


@pytest.mark.skipif(
    os.environ.get("ROBERT_RUN_MULTINEST_SMOKE") != "1",
    reason="set ROBERT_RUN_MULTINEST_SMOKE=1 for the native conda smoke test",
)
def test_real_conda_multinest_smoke_retrieval(tmp_path) -> None:
    pytest.importorskip("pymultinest")

    result = run_multinest(
        _problem(),
        output_dir=tmp_path,
        n_live_points=30,
        evidence_tolerance=1.0,
        mpi_nprocs=1,
        seed=11,
        verbose=False,
    )

    assert result.converged
    assert result.samples.shape[1] == 1
    assert result.samples.shape[0] > 0
    assert np.isfinite(result.log_evidence)
    assert abs(result.best_fit_parameters["level"] - 1.0) < 0.05
