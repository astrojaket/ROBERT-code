"""Lightweight contract tests for the real-data ROBERT WASP-77Ab workflow."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples import run_wasp77ab_robert_joint_pymultinest as workflow
from robert_exoplanets import (
    GaussianLikelihood,
    HeterogeneousLikelihood,
    HeterogeneousLikelihoodComponent,
    HeterogeneousRetrievalProblem,
    HighResolutionEmissionTemplate,
    Observation,
    RetrievalParameter,
    RetrievalParameterSet,
    Spectrum,
    TimeResolvedHighResolutionLikelihood,
    TimeResolvedHighResolutionObservation,
    UniformPrior,
)


def _synthetic_operator_problem() -> HeterogeneousRetrievalProblem:
    wavelengths = np.linspace(2.0, 2.12, 32)
    phases = np.linspace(0.34, 0.46, 5)
    flux = 1.0 + 0.01 * np.sin(np.arange(5 * wavelengths.size).reshape(5, wavelengths.size))
    observation = TimeResolvedHighResolutionObservation.from_arrays(
        wavelengths[None, :],
        flux[None, :, :],
        phases,
        np.zeros(phases.size),
        np.arange(phases.size, dtype=float) + 2_459_197.5,
    )
    template = HighResolutionEmissionTemplate(
        wavelength=np.linspace(1.98, 2.14, 400),
        planet_flux=1.0 + 0.02 * np.sin(np.linspace(0.0, 15.0, 400)),
        stellar_flux=np.ones(400),
        flux_ratio_scale=0.01,
    )
    lrs_observation = Observation.from_arrays(
        [2.02, 2.06],
        [0.01, 0.01],
        [0.001, 0.001],
        wavelength_bin_edges=[2.0, 2.04, 2.08],
    )
    lrs_spectrum = Spectrum.from_arrays(
        lrs_observation.wavelength,
        [0.01, 0.01],
        unit="eclipse_depth",
        observable="eclipse_depth",
        wavelength_unit="micron",
    )
    hrs_likelihood = TimeResolvedHighResolutionLikelihood(
        n_components=2,
        kp_parameter=workflow.HRS_KP_PARAMETER,
        dVsys_parameter=workflow.HRS_DVSYS_PARAMETER,
        dphi_parameter="fixed_phase",
        scale_parameter=workflow.HRS_SCALE_PARAMETER,
    ).prepare(observation)
    parameters = RetrievalParameterSet(
        (
            RetrievalParameter(workflow.HRS_KP_PARAMETER, UniformPrior(180.0, 200.0)),
            RetrievalParameter(workflow.HRS_DVSYS_PARAMETER, UniformPrior(-5.0, 5.0)),
            RetrievalParameter(workflow.HRS_SCALE_PARAMETER, UniformPrior(-1.0, 1.0)),
            RetrievalParameter(workflow.LRS_SCALE_PARAMETER, UniformPrior(0.5, 1.5)),
        )
    )
    likelihood = HeterogeneousLikelihood(
        (
            HeterogeneousLikelihoodComponent(
                name="hrs",
                prediction_key="hrs",
                likelihood=hrs_likelihood,
            ),
            HeterogeneousLikelihoodComponent(
                name="lrs",
                prediction_key="lrs",
                likelihood=GaussianLikelihood(include_normalization=False),
                observation=lrs_observation,
            ),
        )
    )

    def forward(parameters):
        del parameters
        return {"hrs": template, "lrs": lrs_spectrum}

    return HeterogeneousRetrievalProblem(
        name="synthetic-wasp77ab-operator-check",
        parameters=parameters,
        forward_model=forward,
        likelihood=likelihood,
    )


def test_sampler_and_resource_contract_is_laptop_safe() -> None:
    assert workflow.SEEDS == (24680, 24681)
    assert workflow.N_LIVE_POINTS == 64
    assert workflow.MAX_ITER == 0
    assert workflow.MPI_PROCESSES == 1
    assert workflow.OPACITY_MEMORY_LIMIT_BYTES == 1023 * 1024**2
    assert workflow.PROCESS_RSS_HARD_LIMIT_BYTES == 2 * 1024**3
    assert tuple(workflow._thread_values()) == workflow.THREAD_VARIABLES
    assert workflow.MAX_THREADS == 6
    assert all(1 <= value <= 6 for value in workflow._thread_values().values())
    record = workflow.resource_record(check_peak=False)
    assert record["gate_passed"] is True
    assert record["mpi_world_size_observed"] == 1
    assert record["thread_limit"] == 6


def test_shared_builder_is_vmr_only_and_hminus_is_explicit() -> None:
    builder = workflow.build_atmosphere_builder()
    chemistry = builder.chemistry_model
    assert chemistry.convention == "volume_mixing_ratio"
    assert chemistry.active_species == ("H2O", "CO", "H-", "H", "e-")
    assert builder.opacity_free_species == ("H-", "H", "e-")
    assert chemistry.required_parameters() == (
        "log10_H2O_VMR",
        "log10_CO_VMR",
        "log10_Hminus_VMR",
        "log10_electron_VMR",
    )
    parameters = {
        "temperature_K": 2200.0,
        "log10_H2O_VMR": -4.0,
        "log10_CO_VMR": -4.0,
        "log10_Hminus_VMR": -8.0,
        "log10_electron_VMR": -8.0,
    }
    state = builder.build(parameters)
    assert state.composition_convention == "volume_mixing_ratio"
    assert np.allclose(
        sum(np.asarray(value) for value in state.composition.values()),
        1.0,
    )
    assert "mass_fraction_parameters" in chemistry.metadata
    assert chemistry.metadata["mass_fraction_parameters"] == "none"
    assert workflow.HMINUS_CONFIG.metadata["bound_free_cutoff_micron"] == "1.6421"
    assert workflow.HMINUS_CONFIG.metadata["strict_fit_temperature_min_K"] == "2500"
    assert workflow.HMINUS_CONFIG.temperature_extrapolation == "raise"
    assert workflow.HMINUS_CONFIG.spectral_extrapolation == "raise"
    assert np.allclose(state.composition["H"], workflow.FIXED_NEUTRAL_H_VMR)


def test_k_band_selection_preserves_real_operator_metadata() -> None:
    wavelengths = np.array(
        [
            np.linspace(2.0, 2.1, 8),
            np.linspace(2.3, 2.4, 8),
            np.linspace(1.7, 1.8, 8),
        ]
    )
    flux = np.ones((3, 4, 8))
    observation = TimeResolvedHighResolutionObservation.from_arrays(
        wavelengths,
        flux,
        np.linspace(0.3, 0.4, 4),
        np.zeros(4),
        np.arange(4, dtype=float),
        metadata={
            "source": "synthetic",
            "cube_size_bytes": "123",
            "cube_sha256": "cube-hash",
            "info_size_bytes": "456",
            "info_sha256": "info-hash",
        },
    )
    selected, indices = workflow.select_k_band_observation(observation)
    assert indices == (0, 1)
    assert selected.n_orders == 2
    assert selected.metadata["k_band_subset"] == "1.90-2.45 micron"
    assert selected.metadata["selected_order_indices"] == "0,1"
    assert selected.metadata["cube_sha256"] == "cube-hash"
    assert selected.metadata["info_sha256"] == "info-hash"


def test_hrs_and_lrs_providers_share_immutable_lbl_tables(tmp_path: Path) -> None:
    source = tmp_path / "shared-lbl.h5"
    source.write_bytes(b"metadata-only fixture")
    tables = {
        species: workflow.LineByLineTable(
            species=species,
            path=source,
            pressure_bar=[1.0, 10.0],
            temperature_K=[500.0, 1000.0],
            wavelength_micron=[2.0, 2.1],
        )
        for species in ("H2O", "CO")
    }
    hrs_provider = workflow._provider(
        h2o_table=source,
        co_table=source,
        bounds=(2.0, 2.1),
        stride=2,
        tables=tables,
    )
    lrs_provider = workflow._provider(
        h2o_table=source,
        co_table=source,
        bounds=(2.0, 2.1),
        stride=100,
        tables=tables,
    )
    assert hrs_provider.tables["H2O"] is lrs_provider.tables["H2O"]
    assert hrs_provider.tables["CO"] is lrs_provider.tables["CO"]


def test_hrs_response_query_bounds_cover_the_prior_rectangle() -> None:
    wavelengths = np.array(
        [
            np.linspace(2.0, 2.1, 8),
            np.linspace(2.3, 2.4, 8),
        ]
    )
    observation = TimeResolvedHighResolutionObservation.from_arrays(
        wavelengths,
        np.ones((2, 4, 8)),
        np.array([0.2, 0.3, 0.7, 0.8]),
        np.zeros(4),
        np.arange(4, dtype=float),
    )
    lower, upper = workflow._hrs_velocity_bounds(observation)
    query_lower, query_upper = workflow._hrs_query_wavelength_bounds(observation)
    assert lower < upper
    assert query_lower < query_upper
    assert query_lower < float(wavelengths.min())
    assert query_upper > float(wavelengths.max())


def test_real_operator_preflight_checks_model_and_null() -> None:
    problem = _synthetic_operator_problem()
    report = workflow.run_operator_preflight(problem)
    assert report["status"] == "pass"
    assert report["finite_prediction"] is True
    assert report["finite_injection_operator"] is True
    assert report["finite_null"] is True
    assert report["injection_operator"]["applicable"] is True
    assert set(report["prediction_loglike_by_component"]) == {"hrs", "lrs"}
    checkpoints = report["memory_checkpoints"]
    assert {
        "prediction",
        "hrs_loglike",
        "lrs_loglike",
        "null_prediction",
        "null_hrs_loglike",
        "null_lrs_loglike",
    }.issubset(checkpoints)
    assert all(isinstance(value, int) and value >= 0 for value in checkpoints.values())


def test_real_operator_preflight_reuses_streaming_hrs_path(monkeypatch) -> None:
    problem = _synthetic_operator_problem()
    hrs_likelihood = problem.likelihood.components[0].likelihood

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("full HRS diagnostic cubes are forbidden in preflight")

    monkeypatch.setattr(type(hrs_likelihood), "evaluate_model", forbidden)
    report = workflow.run_operator_preflight(problem)
    assert report["status"] == "pass"
    assert report["injection_operator"]["operator"] == (
        "streaming exact Smith/Brogi-Line loglike"
    )
    assert report["injection_operator"]["full_diagnostic_cubes_stored"] is False


def test_null_template_has_zero_planet_to_star_signal() -> None:
    template = HighResolutionEmissionTemplate(
        wavelength=np.linspace(2.0, 2.1, 8),
        planet_flux=np.ones(8),
        stellar_flux=np.ones(8),
    )
    null = workflow._null_template(template)
    assert np.allclose(null.flux_ratio, 0.0)


def test_sampler_stops_after_first_resource_violation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    problem = _synthetic_operator_problem()
    resource_calls = 0
    sampler_seeds: list[int] = []

    def fake_resource_record(**kwargs):
        del kwargs
        nonlocal resource_calls
        resource_calls += 1
        passed = resource_calls == 1
        return {
            "gate_passed": passed,
            "resource_gate_passed": passed,
            "peak_rss_bytes": 100 if passed else 2_000,
        }

    def fake_sampler(problem, **kwargs):
        sampler_seeds.append(int(kwargs["seed"]))
        return SimpleNamespace(
            samples=np.array([[190.0, 0.0, 0.0, 1.0]]),
            parameter_names=problem.parameter_names,
            weights=np.array([1.0]),
            converged=True,
            message="converged",
            log_evidence=-1.0,
            log_evidence_error=0.1,
            best_fit_parameters={},
            metadata={"likelihood_evaluations": "1"},
        )

    monkeypatch.setattr(workflow, "resource_record", fake_resource_record)
    record = workflow.run_sampler_pair(
        problem,
        mode=workflow.JOINT_MODE,
        output_root=tmp_path,
        sampler_runner=fake_sampler,
    )
    assert sampler_seeds == [24680]
    assert len(record["runs"]) == 1
    assert record["runs"][0]["resource_gate_passed"] is False


def test_dry_run_does_not_load_external_data(tmp_path: Path) -> None:
    report_path = tmp_path / "contract.json"
    report = workflow.run_workflow(
        modes=(workflow.JOINT_MODE,),
        report_path=report_path,
        dry_run=True,
        preflight=False,
        run_sampler=False,
    )
    assert report["status"] == "dry_run"
    assert report["real_data_loaded"] is False
    assert report["composition_policy"]["mass_fraction_parameters"] is False
    assert report["hminus_policy"]["bound_free_in_window"] == "zero"
    assert "mean of absolute published asymmetric errors" in report["opacity_policy"]["lrs_uncertainty_model"]
    assert report["diagnostics"]["real_data_truth_recovery"] is False
    assert "exact child-likelihood weighted" in report["diagnostics"]["lrs_residuals"]
    assert "not produced" not in report["diagnostics"]["residual_products"]
    assert "recorded per seed" in report["diagnostics"]["hrs_statistic"]
    assert report["problems"][workflow.JOINT_MODE]["component_names"] == [
        "hrs",
        "nirspec_g395h_nrs1",
        "nirspec_g395h_nrs2",
    ]
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "dry_run"


def test_preflight_report_does_not_rebuild_real_problem_for_parameter_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_calls: list[str] = []
    fake_inputs = SimpleNamespace(
        memory_checkpoints={},
        source_hashes={},
        hrs_order_indices=(),
        hrs_stride=2,
        lrs_stride=100,
        stride_sources={"hrs": "measured", "lrs": "measured"},
        hrs_observation=SimpleNamespace(n_points=1),
        nirspec_observations=SimpleNamespace(n_points=2),
        metadata={
            "hrs_velocity_prior_bounds_km_s": "0,1",
            "hrs_template_query_bounds_micron": "2,3",
            "lrs_accuracy_preferred_stride": "25",
            "lrs_production_stride": "100",
            "lrs_stride_policy": "fixture",
            "lrs_stride_report": "fixture.json",
            "lrs_stride_candidate_rms_sigma": "0.1",
            "lrs_stride_candidate_max_absolute_sigma": "0.2",
            "lrs_stride_candidate_estimate_bytes": "10",
            "lrs_stride_estimated_savings_bytes": "20",
        },
    )

    monkeypatch.setattr(workflow, "load_real_inputs", lambda **_kwargs: fake_inputs)

    def fake_build(_inputs, *, mode, hminus_enabled):
        del _inputs, hminus_enabled
        build_calls.append(mode)
        return SimpleNamespace()

    monkeypatch.setattr(workflow, "build_real_problem", fake_build)
    monkeypatch.setattr(
        workflow,
        "run_operator_preflight",
        lambda _problem: {"status": "pass", "memory_checkpoints": {}},
    )
    report = workflow.run_workflow(
        modes=(workflow.HRS_MODE,),
        report_path=tmp_path / "preflight.json",
        preflight=True,
        run_sampler=False,
        max_memory_bytes=workflow.PROCESS_RSS_HARD_LIMIT_BYTES - 1,
        hminus_enabled=False,
    )

    assert build_calls == [workflow.HRS_MODE]
    assert report["status"] == "preflight_pass"
    assert report["problems"][workflow.HRS_MODE]["parameter_names"] == [
        parameter.name for parameter in workflow._parameter_specs(workflow.HRS_MODE)
    ]


def test_stride_fallback_is_explicit_and_does_not_claim_measurement(tmp_path: Path) -> None:
    stride, source = workflow._stride_from_report(tmp_path / "missing.json", mode=workflow.HRS_MODE)
    assert stride == workflow.SAFE_HRS_LBL_STRIDE
    assert source.startswith("safe_default")


def test_stride_reader_does_not_accept_unmeasured_aliases(tmp_path: Path) -> None:
    report_path = tmp_path / "old-report.json"
    report_path.write_text(
        json.dumps({"hrs": {"selected_stride": 8}, "lrs": {"selected_stride": 9}}),
        encoding="utf-8",
    )
    hrs_stride, hrs_source = workflow._stride_from_report(
        report_path,
        mode=workflow.HRS_MODE,
    )
    lrs_stride, lrs_source = workflow._stride_from_report(
        report_path,
        mode=workflow.LRS_MODE,
    )
    assert hrs_stride == workflow.SAFE_HRS_LBL_STRIDE
    assert lrs_stride == workflow.SAFE_LRS_LBL_STRIDE
    assert hrs_source.startswith("safe_default")
    assert lrs_source.startswith("safe_default")


def test_stride_reader_rejects_selection_from_failed_report(tmp_path: Path) -> None:
    report_path = tmp_path / "failed-report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "fail",
                "hrs": {"selected_finest_passing_stride": 2},
                "lrs": {"selected_finest_passing_stride": 25},
            }
        ),
        encoding="utf-8",
    )
    hrs_stride, hrs_source = workflow._stride_from_report(
        report_path,
        mode=workflow.HRS_MODE,
    )
    lrs_stride, lrs_source = workflow._stride_from_report(
        report_path,
        mode=workflow.LRS_MODE,
    )
    assert hrs_stride == workflow.SAFE_HRS_LBL_STRIDE
    assert lrs_stride == workflow.SAFE_LRS_LBL_STRIDE
    assert hrs_source == "safe_default_stride_report_not_passing"
    assert lrs_source == "safe_default_stride_report_not_passing"


def test_stride_reader_rejects_malformed_numeric_selection(tmp_path: Path) -> None:
    report_path = tmp_path / "malformed-report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "record_type": "wasp77ab_real_grid_lbl_sampling_convergence",
                "target": {"name": "WASP-77Ab"},
                "hrs": {"selected_finest_passing_stride": True},
            }
        ),
        encoding="utf-8",
    )
    stride, source = workflow._stride_from_report(
        report_path,
        mode=workflow.HRS_MODE,
    )
    assert stride == workflow.SAFE_HRS_LBL_STRIDE
    assert source == "safe_default_stride_report_has_no_selection"


def test_source_record_fails_closed_on_missing_or_wrong_file(tmp_path: Path) -> None:
    with np.testing.assert_raises(FileNotFoundError):
        workflow._source_record(
            tmp_path / "missing.bin",
            name="missing.bin",
            role="test",
        )
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    with np.testing.assert_raises(ValueError):
        workflow._source_record(
            source,
            name="source.bin",
            role="test",
            expected_sha256="0" * 64,
        )


def test_phoenix_subset_records_exact_verified_files(tmp_path: Path, monkeypatch) -> None:
    payload = b"minimal PHOENIX fixture"
    spec = SimpleNamespace(
        name="catalog.fits",
        size_bytes=len(payload),
        sha256=workflow.sha256(payload).hexdigest(),
        role="fixture PHOENIX catalogue",
        content_url="https://ssb.stsci.edu/cdbs/grid/phoenix/catalog.fits",
    )
    target = tmp_path / "grid" / "phoenix" / spec.name
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    monkeypatch.setattr(workflow, "PHOENIX_FILES", (spec,))

    records = workflow._phoenix_source_records(tmp_path)

    assert set(records) == {"phoenix_catalog"}
    record = records["phoenix_catalog"]
    assert record["verified"] is True
    assert record["size_bytes"] == len(payload)
    assert record["expected_size_bytes"] == len(payload)
    assert record["relative_path"] == "grid/phoenix/catalog.fits"
    assert record["official_url"] == spec.content_url


def test_phoenix_subset_fails_closed_before_model_loading(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="PHOENIX subset file"):
        workflow._phoenix_source_records(tmp_path)


def test_explicit_blackbody_does_not_require_phoenix_root(tmp_path: Path) -> None:
    report = workflow.run_workflow(
        modes=(workflow.HRS_MODE,),
        stellar_spectrum_model="blackbody",
        pysyn_cdbs_root=tmp_path / "missing-pysyn-cdbs",
        dry_run=True,
    )

    assert report["status"] == "dry_run"
    assert report["stellar_policy"]["model"] == "blackbody"
    assert report["stellar_policy"]["pysyn_cdbs_required"] is False
    assert report["stellar_policy"]["phoenix_subset"] == (
        "not required for explicit blackbody mode"
    )


def test_stride_report_selects_finest_passing_real_grid_values() -> None:
    report = Path(workflow.DEFAULT_STRIDE_REPORT)
    if not report.is_file():
        return
    assert workflow._stride_from_report(report, mode=workflow.HRS_MODE)[0] == 2
    assert workflow._stride_from_report(report, mode=workflow.LRS_MODE)[0] == 25


def test_lrs_production_stride_uses_report_validated_memory_safe_candidate() -> None:
    report = Path(workflow.DEFAULT_STRIDE_REPORT)
    if not report.is_file():
        return
    preferred, preferred_source = workflow._stride_from_report(
        report,
        mode=workflow.LRS_MODE,
    )
    stride, source, metadata = workflow._lrs_memory_safe_stride_from_report(
        report,
        accuracy_preferred_stride=preferred,
        accuracy_source=preferred_source,
    )
    assert preferred == 25
    assert stride == workflow.LRS_MEMORY_SAFE_STRIDE == 100
    assert "report_validated_laptop_memory_safe_override:25->100" in source
    assert metadata["policy"] == "report_validated_laptop_memory_safe_override"
    assert float(metadata["candidate_rms_sigma"]) == pytest.approx(
        0.07970951513528207
    )
    assert float(metadata["candidate_max_absolute_sigma"]) == pytest.approx(
        0.2570330977642441
    )
    assert int(metadata["candidate_estimate_bytes"]) == 38_390_400
    assert int(metadata["estimated_savings_bytes_vs_accuracy_preferred"]) == (
        153_543_024 - 38_390_400
    )


def test_lrs_production_stride_fails_closed_without_report(tmp_path: Path) -> None:
    with pytest.raises(workflow.RobertValidationError, match="requires a passing report"):
        workflow._lrs_memory_safe_stride_from_report(
            tmp_path / "missing.json",
            accuracy_preferred_stride=25,
            accuracy_source="safe_default",
        )


def test_lrs_production_stride_rejects_failed_candidate_gate(tmp_path: Path) -> None:
    report = tmp_path / "failed-lrs-candidate.json"
    report.write_text(
        json.dumps(
            {
                "status": "pass",
                "record_type": "wasp77ab_real_grid_lbl_sampling_convergence",
                "target": {"name": "WASP-77Ab"},
                "acceptance": {
                    "lrs_finest_passing_stride_selected": True,
                    "lrs_reference_and_candidates_complete": True,
                    "opacity_estimates_below_1023_MiB": True,
                    "process_rss_below_1_9_GiB": True,
                    "thread_values_between_1_and_3": True,
                },
                "lrs": {
                    "candidate_strides": [25, 100],
                    "metrics": {
                        "100": {
                            "gate_pass": False,
                            "rms_sigma": 0.3,
                            "max_absolute_sigma": 0.8,
                            "gate_rms_limit_sigma": 0.25,
                            "gate_max_limit_sigma": 1.0,
                        }
                    },
                    "max_preparation_estimate_bytes": {"100": 38_390_400},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(workflow.RobertValidationError, match="does not pass its benchmark gate"):
        workflow._lrs_memory_safe_stride_from_report(
            report,
            accuracy_preferred_stride=25,
            accuracy_source="measured",
        )


def test_hminus_off_keeps_same_parameterisation_for_evidence_comparison(
    tmp_path: Path,
) -> None:
    report = workflow.run_workflow(
        modes=(workflow.HRS_MODE,),
        report_path=tmp_path / "hminus-off.json",
        dry_run=True,
        hminus_enabled=False,
    )
    contract = report["problems"][workflow.HRS_MODE]
    assert report["hminus_enabled"] is False
    assert report["hminus_policy"]["enabled"] is False
    assert contract["hminus"]["enabled"] is False
    assert "log10_Hminus_VMR" in contract["parameter_names"]


def test_hminus_metadata_false_text_is_not_truthy() -> None:
    assert workflow._metadata_boolean(False, name="hminus_enabled") is False
    assert workflow._metadata_boolean(
        "false", name="hminus_enabled"
    ) is False
    assert workflow._metadata_boolean("true", name="hminus_enabled") is True
    with np.testing.assert_raises(ValueError):
        workflow._metadata_boolean("enabled", name="hminus_enabled")


class _FakeHrsLikelihood:
    effective_residual_rank = 9

    def velocity(self, parameters):
        del parameters
        return np.array([1.0, 2.0])

    def cross_correlation(self, prediction, parameters):
        del prediction, parameters
        return np.array([0.1, -0.2])


class _FakeExactGaussianLikelihood:
    def __init__(self, chi_square: float = 17.0) -> None:
        self.chi_square_value = chi_square
        self.chi_square_calls = 0

    def effective_inputs(self, prediction, observation, parameters):
        del prediction, observation, parameters
        return (
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 4.0, 3.0]),
            np.ones(3),
        )

    def chi_square(self, prediction, observation, parameters):
        del prediction, observation, parameters
        self.chi_square_calls += 1
        return self.chi_square_value

    def effective_residual_rank(self, observation):
        del observation
        return 2


class _FakeDiagnosticLikelihood:
    def __init__(self, lrs_likelihood) -> None:
        self.lrs_likelihood = lrs_likelihood
        self.components = (
            SimpleNamespace(
                name="hrs",
                prediction_key="hrs",
                likelihood=_FakeHrsLikelihood(),
                observation=None,
            ),
            SimpleNamespace(
                name="nirspec_g395h_nrs1",
                prediction_key="nirspec_g395h_nrs1",
                likelihood=lrs_likelihood,
                observation=object(),
            ),
        )

    def loglike_by_component(self, prediction, parameters):
        del prediction, parameters
        return {"hrs": -12.5, "nirspec_g395h_nrs1": -3.25}


class _FakeDiagnosticProblem:
    def __init__(self, lrs_likelihood) -> None:
        self.likelihood = _FakeDiagnosticLikelihood(lrs_likelihood)

    def predict(self, parameters):
        del parameters
        return {"hrs": object(), "nirspec_g395h_nrs1": object()}


def test_post_run_diagnostics_use_exact_component_statistics() -> None:
    lrs_likelihood = _FakeExactGaussianLikelihood(chi_square=17.0)
    problem = _FakeDiagnosticProblem(lrs_likelihood)
    posterior = {
        "log10_H2O_VMR": {"median": -4.0, "lower_90": -4.5, "upper_90": -3.5},
        "log10_CO_VMR": {"median": -3.8, "lower_90": -4.2, "upper_90": -3.4},
        "log10_Hminus_VMR": {"median": -8.0, "lower_90": -9.0, "upper_90": -7.0},
        "log10_electron_VMR": {"median": -7.0, "lower_90": -8.0, "upper_90": -6.0},
    }
    diagnostics = workflow.summarize_post_run_diagnostics(
        problem,
        best_fit_parameters={
            "Kp": 191.0,
            "dVsys": -5.0,
            "log10_H2O_VMR": -4.0,
            "log10_CO_VMR": -3.8,
        },
        posterior=posterior,
    )
    assert diagnostics["status"] == "pass"
    assert diagnostics["best_fit_log_likelihood_by_component"] == {
        "hrs": -12.5,
        "nirspec_g395h_nrs1": -3.25,
    }
    hrs = diagnostics["hrs"]
    assert hrs["statistic_name"] == "smith_brogi_line_log_likelihood"
    assert hrs["statistic_value"] == -12.5
    assert hrs["chi_square"] is None
    assert hrs["chi_square_defined"] is False
    assert hrs["effective_retained_rank"] == 9
    assert diagnostics["rv"]["Kp_km_s"] == 191.0
    assert diagnostics["rv"]["dVsys_km_s"] == -5.0
    lrs = diagnostics["lrs"]["nirspec_g395h_nrs1"]
    assert lrs["count"] == 3
    assert lrs["effective_retained_rank"] == 2
    assert lrs["weighted_chi_square"] == 17.0
    assert lrs["residual_rms"] == np.sqrt(5.0 / 3.0)
    assert lrs["diagonal_fallback_used"] is False
    assert lrs_likelihood.chi_square_calls == 1
    assert diagnostics["abundance_posterior_intervals"]["H2O"]["median"] == -4.0
    assert diagnostics["abundance_posterior_intervals"]["e-"]["upper_90"] == -6.0
    assert diagnostics["full_arrays_stored"] is False


def test_post_run_diagnostics_reject_missing_exact_lrs_statistic() -> None:
    class MissingChiSquare:
        def effective_inputs(self, prediction, observation, parameters):
            del prediction, observation, parameters
            return np.ones(2), np.ones(2), np.ones(2)

        def effective_residual_rank(self, observation):
            del observation
            return 2

    diagnostics = workflow.summarize_post_run_diagnostics(
        _FakeDiagnosticProblem(MissingChiSquare()),
        best_fit_parameters={"Kp": 191.0, "dVsys": -5.0},
    )
    assert diagnostics["status"] == "fail"
    residual = diagnostics["lrs"]["nirspec_g395h_nrs1"]
    assert residual["status"] == "fail"
    assert residual["diagonal_fallback_used"] is False
    assert "diagonal fallback is forbidden" in residual["error"]


def test_sampler_record_contains_post_run_diagnostics(tmp_path: Path) -> None:
    problem = _synthetic_operator_problem()
    result = SimpleNamespace(
        samples=np.array(
            [[190.0, 0.0, 0.0, 1.0], [191.0, -1.0, 0.1, 1.1]],
        ),
        parameter_names=problem.parameter_names,
        weights=np.array([0.4, 0.6]),
        converged=True,
        method="multinest",
        message="converged",
        log_evidence=-10.0,
        log_evidence_error=0.1,
        best_fit_parameters={
            workflow.HRS_KP_PARAMETER: 190.0,
            workflow.HRS_DVSYS_PARAMETER: 0.0,
            workflow.HRS_SCALE_PARAMETER: 0.0,
            workflow.LRS_SCALE_PARAMETER: 1.0,
        },
        metadata={"likelihood_evaluations": 12},
    )
    record = workflow._run_record(
        result,
        problem=problem,
        seed=24680,
        elapsed=1.0,
        resources={"peak_rss_bytes": 100, "gate_passed": True},
        output_dir=tmp_path,
    )
    assert record["status"] == "pass"
    assert record["diagnostics_gate_passed"] is True
    assert record["diagnostics"]["status"] == "pass"
    assert set(record["diagnostics"]["best_fit_log_likelihood_by_component"]) == {
        "hrs",
        "lrs",
    }
