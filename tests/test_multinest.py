"""Tests for the optional PyMultiNest adapter."""

from __future__ import annotations

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
from robert_exoplanets.retrieval.samplers.multinest import (
    MULTINEST_MAX_SEED,
    _effective_multinest_seed,
    _global_evidence,
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
    np.testing.assert_allclose(result.weights, [0.25, 0.75])
    assert (tmp_path / "chains" / "1-params.json").is_file()
    assert (tmp_path / "sampler_status.json").is_file()


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
