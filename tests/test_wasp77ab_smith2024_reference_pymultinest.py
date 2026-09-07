"""Lightweight tests for the bounded WASP-77Ab reference workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets.core import RobertCoverageError
from robert_exoplanets.instruments import (
    HighResolutionEmissionTemplate,
    Observation,
    ObservationCollection,
    ObservationDataset,
    TimeResolvedHighResolutionObservation,
)
from robert_exoplanets.retrieval.samplers import NestedSamplerResult

from examples import run_wasp77ab_smith2024_reference_pymultinest as workflow


def _reference_inputs() -> workflow.ReferenceInputs:
    order_wavelengths = np.linspace(2.01, 2.08, 8)[None, :]
    phases = np.linspace(0.34, 0.46, 5)
    flux = np.ones((1, 5, 8), dtype=float)
    flux += 0.01 * np.sin(np.arange(flux.size, dtype=float).reshape(flux.shape))
    hrs = TimeResolvedHighResolutionObservation.from_arrays(
        order_wavelengths=order_wavelengths,
        flux=flux,
        phase=phases,
        fixed_velocity_km_s=np.zeros(5),
        time_bjd=np.arange(5, dtype=float) + 2_459_197.5,
        wavelength_unit="micron",
        flux_unit="detector_counts",
        observable="normalized_high_resolution_flux",
    )
    template = workflow.HighResolutionEmissionTemplate(
        wavelength=np.linspace(2.0, 2.12, 200),
        planet_flux=1.0 + 0.1 * np.sin(np.linspace(0.0, 8.0, 200)),
        stellar_flux=np.ones(200),
        wavelength_unit="micron",
        flux_ratio_scale=0.01,
    )
    datasets = []
    for name, detector, centres, edges in (
        (
            "nirspec_g395h_nrs1",
            "NRS1",
            [2.02, 2.04],
            [2.01, 2.03, 2.05],
        ),
        (
            "nirspec_g395h_nrs2",
            "NRS2",
            [2.08, 2.10],
            [2.07, 2.09, 2.11],
        ),
    ):
        observation = Observation.from_arrays(
            wavelength=centres,
            flux=[0.01, 0.01],
            uncertainty=[0.001, 0.001],
            wavelength_unit="micron",
            flux_unit="eclipse_depth",
            observable="eclipse_depth",
            wavelength_bin_edges=edges,
        )
        datasets.append(
            ObservationDataset(
                name=name,
                observation=observation,
                metadata={"detector": detector},
            )
        )
    nirspec = ObservationCollection(tuple(datasets))
    spectra = workflow._lrs_template_spectra(template, nirspec)
    return workflow.ReferenceInputs(
        hrs_observation=hrs,
        nirspec_observations=nirspec,
        template=template,
        hrs_template=workflow._prepare_hrs_template(template, hrs),
        lrs_template_spectra=spectra,
        source_hashes={"synthetic": {"verified": False}},
    )


def test_sampler_contract_and_thread_environment_are_strict() -> None:
    assert workflow.SEEDS == (24680, 24681)
    assert workflow.N_LIVE_POINTS == 64
    assert workflow.MAX_ITER == 0
    assert workflow.EVIDENCE_TOLERANCE == 0.5
    assert workflow.SAMPLING_EFFICIENCY == 0.8
    assert workflow.MPI_PROCESSES == 1
    assert workflow.PROCESS_RSS_LIMIT_BYTES == 2 * 1024**3
    assert all(1 <= value <= 3 for value in workflow._thread_values().values())
    assert tuple(workflow._thread_values()) == workflow.THREAD_VARIABLES


def test_reference_problems_have_required_parameters_and_terms() -> None:
    problems = workflow.build_reference_problems(_reference_inputs())

    assert tuple(problems) == workflow.ALL_MODES
    assert problems[workflow.HRS_MODE].parameter_names == (
        "Kp",
        "dVsys",
        "log10_a_hrs",
    )
    assert problems[workflow.HRS_MODE].parameters.bounds == (
        (172.0, 212.0),
        (-20.0, 20.0),
        (-2.0, 2.0),
    )
    assert problems[workflow.LRS_MODE].parameter_names == (
        "lrs_template_scale",
    )
    assert problems[workflow.LRS_MODE].parameters.bounds == ((0.5, 1.5),)
    assert problems[workflow.JOINT_MODE].parameter_names == (
        "Kp",
        "dVsys",
        "log10_a_hrs",
        "lrs_template_scale",
    )
    assert problems[workflow.HRS_MODE].likelihood.component_names == ("hrs",)
    assert problems[workflow.LRS_MODE].likelihood.component_names == (
        "nirspec_g395h_nrs1",
        "nirspec_g395h_nrs2",
    )
    assert problems[workflow.JOINT_MODE].likelihood.component_names == (
        "hrs",
        "nirspec_g395h_nrs1",
        "nirspec_g395h_nrs2",
    )
    assert problems[workflow.JOINT_MODE].likelihood.components[0].observation is None
    assert all(
        component.observation is not None
        for component in problems[workflow.JOINT_MODE].likelihood.components[1:]
    )
    assert "not a claim of exact Smith likelihood reproduction" in str(
        problems[workflow.HRS_MODE].likelihood.components[0].metadata["statistic"]
    )
    assert "R=45,000 LSF" in str(problems[workflow.HRS_MODE].metadata["model_response"])
    assert (
        problems[workflow.HRS_MODE].likelihood.components[0].metadata[
            "template_response"
        ]
        .lower()
        .startswith("official r=500,000 template convolved")
    )


def test_lrs_template_scale_is_explicit_and_shared_by_both_detectors() -> None:
    inputs = _reference_inputs()
    problem = workflow.build_lrs_problem(inputs)

    unit_scale = problem.predict([1.0])
    double_scale = problem.predict([2.0])
    for name in ("nirspec_g395h_nrs1", "nirspec_g395h_nrs2"):
        np.testing.assert_allclose(
            double_scale[name].values,
            2.0 * unit_scale[name].values,
        )
    assert problem.metadata["template_scale_prior"] == workflow.LRS_SCALE_PRIOR_DESCRIPTION
    assert "VMR" in problem.metadata["abundance_state"]


def test_hrs_response_order_coverage_and_line_shape() -> None:
    order_wavelengths = np.linspace(2.01, 2.08, 32)[None, :]
    observation = TimeResolvedHighResolutionObservation.from_arrays(
        order_wavelengths=order_wavelengths,
        flux=np.ones((1, 3, order_wavelengths.shape[1])),
        phase=np.array([0.34, 0.40, 0.46]),
        fixed_velocity_km_s=np.zeros(3),
        time_bjd=np.arange(3, dtype=float) + 2_459_197.5,
        wavelength_unit="micron",
        flux_unit="detector_counts",
        observable="normalized_high_resolution_flux",
    )
    source_wavelength = np.linspace(2.0, 2.12, 5_001)
    source_planet = np.ones(source_wavelength.size)
    source_planet[np.argmin(np.abs(source_wavelength - 2.045))] += 1.0
    source = HighResolutionEmissionTemplate(
        wavelength=source_wavelength,
        planet_flux=source_planet,
        stellar_flux=np.ones(source_wavelength.size),
        wavelength_unit="micron",
        flux_ratio_scale=0.01,
    )

    prepared = workflow._prepare_hrs_template(source, observation)
    metadata = prepared.metadata
    assert metadata["hrs_response_stage_order"] == "gray_rotation_then_gaussian_lsf"
    assert metadata["hrs_pixel_integration"] == (
        "not applied; Smith interpolates the convolved model"
    )
    assert metadata["hrs_smith_model_resolution"] == (
        "Smith used R=250,000; this public template carrier is R=500,000"
    )
    query_lower, query_upper = workflow._hrs_query_wavelength_bounds(observation)
    assert prepared.wavelength[0] <= query_lower
    assert prepared.wavelength[-1] >= query_upper
    source_at_output = np.interp(
        prepared.wavelength,
        source.wavelength,
        source.planet_flux,
    )
    assert np.max(np.abs(prepared.planet_flux - source_at_output)) > 1.0e-3

    narrow_source = HighResolutionEmissionTemplate(
        wavelength=np.linspace(2.01, 2.08, 2_001),
        planet_flux=np.ones(2_001),
        stellar_flux=np.ones(2_001),
        wavelength_unit="micron",
    )
    with pytest.raises(RobertCoverageError, match="support margins"):
        workflow._prepare_hrs_template(narrow_source, observation)


def test_dry_run_writes_contract_without_loading_external_data(tmp_path: Path) -> None:
    report_path = tmp_path / "reference.json"
    report = workflow.run_workflow(
        modes=(workflow.HRS_MODE,),
        report_path=report_path,
        dry_run=True,
        run_sampler=False,
        preflight=False,
    )

    assert report["status"] == "dry_run"
    assert report["truth_recovery_claim"] is False
    assert report["sampler_policy"]["seeds"] == [24680, 24681]
    assert report["problems"][workflow.HRS_MODE]["parameter_names"] == [
        "Kp",
        "dVsys",
        "log10_a_hrs",
    ]
    assert report["data"]["source_hashes"]["hrs_cube"]["verified"] is False
    assert report["data"]["hrs_response_template"]["stage_order"] == (
        "gray_rotation_then_gaussian_lsf"
    )
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "dry_run"


def test_two_seed_runner_uses_fixed_pymultinest_settings_and_reports_components(
    tmp_path: Path,
) -> None:
    problem = workflow.build_hrs_problem(_reference_inputs())
    calls: list[dict[str, object]] = []

    def fake_sampler(problem, **kwargs):
        calls.append(kwargs)
        parameters = np.array(
            [
                [192.0, -5.0, 0.0],
                [193.0, -4.0, 0.1],
                [191.0, -6.0, -0.1],
            ]
        )
        return NestedSamplerResult(
            method="multinest",
            parameter_names=problem.parameter_names,
            samples=parameters,
            log_likelihood=[-1.0, -0.5, -0.8],
            weights=[0.2, 0.5, 0.3],
            log_evidence=-12.0 + 0.1 * len(calls),
            log_evidence_error=0.1,
            best_fit_parameters={
                "Kp": 192.0,
                "dVsys": -5.0,
                "log10_a_hrs": 0.0,
            },
            metadata={"likelihood_evaluations": "123"},
            converged=True,
            message="converged",
        )

    record = workflow.run_sampler_pair(
        problem,
        mode=workflow.HRS_MODE,
        output_root=tmp_path,
        sampler_runner=fake_sampler,
    )

    assert [call["seed"] for call in calls] == [24680, 24681]
    assert all(call["n_live_points"] == 64 for call in calls)
    assert all(call["max_iter"] == 0 for call in calls)
    assert all(call["evidence_tolerance"] == 0.5 for call in calls)
    assert all(call["sampling_efficiency"] == 0.8 for call in calls)
    assert all(call["mpi_nprocs"] == 1 for call in calls)
    assert record["gate_passed"] is True
    assert record["evidence_comparison"]["delta_log_evidence"] == pytest.approx(0.1)
    assert record["runs"][0]["calls"] == 123
    assert np.isfinite(record["runs"][0]["per_component"]["hrs"]["loglike"])
    assert record["runs"][0]["per_component"]["hrs"]["chi_square"] is None
    assert record["runs"][0]["posterior"]["Kp"]["median"] == pytest.approx(
        192.2857142857
    )


def test_source_hash_record_fails_closed_on_wrong_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"fixed source")
    with np.testing.assert_raises(ValueError):
        workflow._verified_source_record(
            name="source.bin",
            path=source,
            role="test",
            url=None,
            expected_size=999,
        )


def test_evidence_gate_fails_closed_for_nonfinite_values() -> None:
    records = [
        {
            "seed": 24680,
            "converged": True,
            "resource_gate_passed": True,
            "log_evidence": float("nan"),
            "log_evidence_error": 0.1,
        },
        {
            "seed": 24681,
            "converged": True,
            "resource_gate_passed": True,
            "log_evidence": -12.0,
            "log_evidence_error": 0.1,
        },
    ]
    comparison = workflow._evidence_comparison(records)
    assert comparison["gate_passed"] is False
    assert comparison["finite_evidence"] is False
    assert comparison["delta_log_evidence"] is None


def test_sampler_stops_before_second_seed_after_resource_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = workflow.build_hrs_problem(_reference_inputs())
    calls: list[int] = []
    resource_calls = 0

    def fake_resource_record(**kwargs):
        del kwargs
        nonlocal resource_calls
        resource_calls += 1
        passed = resource_calls not in {3, 4}
        return {
            "gate_passed": passed,
            "peak_rss_bytes": 100 if passed else 2_000,
        }

    def fake_sampler(problem, **kwargs):
        del problem
        calls.append(int(kwargs["seed"]))
        return NestedSamplerResult(
            method="multinest",
            parameter_names=("Kp", "dVsys", "log10_a_hrs"),
            samples=np.array([[192.0, -5.0, 0.0]]),
            log_likelihood=np.array([-1.0]),
            weights=np.array([1.0]),
            log_evidence=-12.0,
            log_evidence_error=0.1,
            best_fit_parameters={"Kp": 192.0, "dVsys": -5.0, "log10_a_hrs": 0.0},
            converged=True,
        )

    monkeypatch.setattr(workflow, "resource_record", fake_resource_record)
    record = workflow.run_sampler_pair(
        problem,
        mode=workflow.HRS_MODE,
        output_root=tmp_path,
        sampler_runner=fake_sampler,
    )
    assert calls == [24680]
    assert len(record["runs"]) == 1
    assert record["runs"][0]["resource_gate_passed"] is False
