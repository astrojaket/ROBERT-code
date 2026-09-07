"""Lightweight contracts for the real-data ROBERT operator injection test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from examples import validate_wasp77ab_robert_injection as workflow
from robert_exoplanets import (
    HeterogeneousLikelihood,
    HeterogeneousLikelihoodComponent,
    HeterogeneousRetrievalProblem,
    HighResolutionEmissionTemplate,
    Observation,
    ObservationCollection,
    ObservationDataset,
    RetrievalParameter,
    RetrievalParameterSet,
    Spectrum,
    TimeResolvedHighResolutionLikelihood,
    TimeResolvedHighResolutionObservation,
    UniformPrior,
)


def _small_hrs() -> tuple[
    TimeResolvedHighResolutionObservation,
    HighResolutionEmissionTemplate,
]:
    wavelengths = np.exp(np.linspace(np.log(1.95), np.log(2.15), 96))
    phases = np.linspace(0.32, 0.46, 6)
    flux = 1.0 + 0.01 * np.sin(np.arange(6 * 96).reshape(1, 6, 96))
    observation = TimeResolvedHighResolutionObservation.from_arrays(
        wavelengths[None, :],
        flux,
        phases,
        np.linspace(-1.0, 1.0, 6),
        2_459_197.5 + np.arange(6, dtype=float) * 0.01,
        metadata={"source": "synthetic fixture"},
    )
    template_wavelength = np.exp(np.linspace(np.log(1.85), np.log(2.25), 1601))
    line = np.exp(-0.5 * ((template_wavelength - 2.05) / 2.0e-4) ** 2)
    template = HighResolutionEmissionTemplate(
        wavelength=template_wavelength,
        planet_flux=1.0 + line,
        stellar_flux=np.ones(template_wavelength.size),
        # Use a deliberately visible fixture signal.  The real workflow uses
        # the ROBERT eclipse-depth ratio, which is much smaller.
        flux_ratio_scale=0.2,
        name="synthetic fixture HRS template",
    )
    return observation, template


def _small_lrs() -> tuple[ObservationCollection, dict[str, Spectrum]]:
    first = Observation.from_arrays(
        [2.81, 2.83, 2.85],
        [0.01, 0.01, 0.01],
        [0.001, 0.001, 0.001],
        wavelength_bin_edges=[2.80, 2.82, 2.84, 2.86],
        instrument="NRS1",
    )
    second = Observation.from_arrays(
        [3.84, 3.88, 3.92],
        [0.02, 0.02, 0.02],
        [0.002, 0.002, 0.002],
        wavelength_bin_edges=[3.82, 3.86, 3.90, 3.94],
        instrument="NRS2",
    )
    collection = ObservationCollection(
        (
            ObservationDataset("nrs1", first),
            ObservationDataset("nrs2", second),
        ),
        name="synthetic NIRSpec fixture",
    )
    predictions = {
        "nrs1": Spectrum.from_arrays(
            first.wavelength,
            [0.011, 0.012, 0.013],
            unit=first.flux_unit,
            observable=first.observable,
        ),
        "nrs2": Spectrum.from_arrays(
            second.wavelength,
            [0.019, 0.020, 0.021],
            unit=second.flux_unit,
            observable=second.observable,
        ),
    }
    return collection, predictions


def test_dry_run_has_no_external_data_or_sampler(tmp_path: Path) -> None:
    report_path = tmp_path / "operator-injection.json"
    report = workflow.run_validation(report_path=report_path)
    assert report["status"] == "dry_run"
    assert report["real_data_loaded"] is False
    assert report["sampler"]["used"] is False
    assert report["composition_policy"]["robert_convention"] == "volume_mixing_ratio"
    assert report["composition_policy"]["mass_fraction_parameters_in_robert"] is False
    assert report["generative_model"]["synthetic_spectra_written"] is False
    assert report["truth_parameters"] == {
        name: float(value)
        for name, value in workflow.DEFAULT_TRUTH_PARAMETERS.items()
    }
    assert report["generative_model"]["hrs_noise_calibration"] is None
    assert report_path.is_file()


def test_hrs_generation_uses_prepared_data_scale_and_ratio() -> None:
    observation, template = _small_hrs()
    prepared = TimeResolvedHighResolutionLikelihood(
        n_components=2,
        kp_parameter="Kp",
        dVsys_parameter="dVsys",
        dphi_parameter="fixed_phase",
        scale_parameter="log10_a",
    ).prepare(observation)
    parameters = {"Kp": 190.0, "dVsys": 0.0, "log10_a": 0.0}
    expected = np.array(
        prepared.evaluate_model(template, parameters).injected_model,
        copy=True,
    )
    scales = np.sqrt(np.mean(np.square(prepared.data_residual), axis=(1, 2)))
    generator = np.random.default_rng(workflow.HRS_NOISE_SEED)
    for order, scale in enumerate(scales):
        expected[order] += generator.standard_normal(expected[order].shape) * scale
    generated = workflow.generate_synthetic_hrs_observation(
        observation,
        prepared,
        template,
        parameters,
    )
    assert np.array_equal(generated.flux, expected)
    assert generated.metadata["synthetic_generation_equation"] == workflow.HRS_GENERATIVE_EQUATION
    assert generated.metadata["synthetic_noise"] == "fixed-seed independent Gaussian detector noise"
    assert generated.metadata["synthetic_noise_seed"] == str(workflow.HRS_NOISE_SEED)
    assert generated.metadata["synthetic_noise_scale_definition"] == (
        "per-order RMS of prepared Smith data_residual over valid pixels"
    )
    assert generated.metadata["synthetic_noise_chunking"] == "one order at a time"
    assert generated.metadata["synthetic_noise_scale_sha256"] == workflow._array_sha256(scales)
    assert generated.metadata["source_data_scale"] == "prepared Smith low-rank PCA scale"
    assert generated.metadata["source_data_residual"] == "prepared Smith clipped PCA residual"
    calibration = workflow._hrs_noise_calibration_record(generated)
    assert calibration["seed"] == workflow.HRS_NOISE_SEED
    assert calibration["n_orders"] == 1
    assert len(calibration["sigma_by_order"]) == 1
    assert calibration["sigma_sha256"] == generated.metadata["synthetic_noise_scale_sha256"]
    assert calibration["sigma_summary"]["min"] == calibration["sigma_by_order"][0]
    with pytest.raises(workflow.RobertValidationError, match="order count"):
        workflow._hrs_noise_calibration_record(generated, expected_n_orders=19)


def test_hrs_generation_seed_and_scale_metadata_are_deterministic() -> None:
    observation, template = _small_hrs()
    prepared = TimeResolvedHighResolutionLikelihood(
        n_components=2,
        kp_parameter="Kp",
        dVsys_parameter="dVsys",
        dphi_parameter="fixed_phase",
        scale_parameter="log10_a",
    ).prepare(observation)
    parameters = {"Kp": 190.0, "dVsys": 0.0, "log10_a": 0.0}
    first = workflow.generate_synthetic_hrs_observation(
        observation,
        prepared,
        template,
        parameters,
        seed=41,
    )
    second = workflow.generate_synthetic_hrs_observation(
        observation,
        prepared,
        template,
        parameters,
        seed=41,
    )
    third = workflow.generate_synthetic_hrs_observation(
        observation,
        prepared,
        template,
        parameters,
        seed=42,
    )
    assert np.array_equal(first.flux, second.flux)
    assert not np.array_equal(first.flux, third.flux)
    assert first.metadata["synthetic_noise_seed"] == "41"
    assert first.metadata["synthetic_noise_scale_sha256"] == (
        second.metadata["synthetic_noise_scale_sha256"]
    )


def test_real_report_schema_persists_truth_and_hrs_calibration(monkeypatch) -> None:
    calibration = {
        "seed": workflow.HRS_NOISE_SEED,
        "scale_definition": "per-order RMS of prepared Smith data_residual over valid pixels",
        "distribution": "fixed-seed independent Gaussian detector noise",
        "chunking": "one order at a time",
        "n_orders": 1,
        "sigma_by_order": [0.25],
        "sigma_sha256": workflow._array_sha256([0.25]),
        "sigma_summary": {"min": 0.25, "max": 0.25, "mean": 0.25, "rms": 0.25},
    }
    check = {
        "finite": True,
        "truth_loglike": 3.0,
        "null_loglike": 1.0,
        "off_velocity_loglike": 2.0,
        "truth_minus_null": 2.0,
        "truth_minus_off_velocity": 1.0,
        "truth_preferred_to_null": True,
        "truth_preferred_to_off_velocity": True,
    }
    fake = workflow.InjectionRunResult(
        truth_parameters=dict(workflow.DEFAULT_TRUTH_PARAMETERS),
        hrs_noise_calibration=calibration,
        source_hashes={},
        grid_record={},
        on_checks={mode: check for mode in workflow.ALL_MODES},
        off_checks={mode: check for mode in workflow.ALL_MODES},
        resources={"gate_passed": True},
    )
    monkeypatch.setattr(
        workflow.robert_workflow,
        "resource_record",
        lambda **_: {"gate_passed": True},
    )
    monkeypatch.setattr(workflow, "_run_real_validation", lambda **_: fake)
    report = workflow.run_validation(run_real=True, dry_run=False)
    assert report["truth_parameters"] == {
        name: float(value)
        for name, value in workflow.DEFAULT_TRUTH_PARAMETERS.items()
    }
    assert report["generative_model"]["hrs_noise_calibration"] == calibration
    assert report["generative_model"]["hrs_noise_calibration"]["sigma_by_order"] == [0.25]
    assert report["status"] == "pass"


def test_lrs_generation_is_seeded_and_preserves_bins() -> None:
    observations, predictions = _small_lrs()
    first = workflow.generate_synthetic_lrs_observations(observations, predictions, seed=41)
    second = workflow.generate_synthetic_lrs_observations(observations, predictions, seed=41)
    third = workflow.generate_synthetic_lrs_observations(observations, predictions, seed=42)
    for name in observations.names:
        first_observation = first.datasets[observations.names.index(name)].observation
        second_observation = second.datasets[observations.names.index(name)].observation
        third_observation = third.datasets[observations.names.index(name)].observation
        original = observations.datasets[observations.names.index(name)].observation
        assert np.array_equal(first_observation.flux, second_observation.flux)
        assert not np.array_equal(first_observation.flux, third_observation.flux)
        assert np.array_equal(first_observation.wavelength_bin_edges, original.wavelength_bin_edges)
        assert first_observation.metadata["synthetic_noise_seed"] == "41"
    assert first.metadata["synthetic_noise_seed"] == "41"


def test_truth_null_off_velocity_contract_is_deterministic() -> None:
    observation, template = _small_hrs()
    prepared = TimeResolvedHighResolutionLikelihood(
        n_components=2,
        kp_parameter="Kp",
        dVsys_parameter="dVsys",
        dphi_parameter="fixed_phase",
        scale_parameter="log10_a",
    ).prepare(observation)
    parameters = RetrievalParameterSet(
        (
            RetrievalParameter("Kp", UniformPrior(170.0, 215.0)),
            RetrievalParameter("dVsys", UniformPrior(-20.0, 20.0)),
            RetrievalParameter("log10_a", UniformPrior(-2.0, 2.0)),
        )
    )
    lrs_obs, lrs_predictions = _small_lrs()
    lrs_synthetic = workflow.generate_synthetic_lrs_observations(
        lrs_obs,
        lrs_predictions,
        seed=20260831,
    )
    template_prediction = template

    def forward(values):
        del values
        return {
            "hrs": template_prediction,
            "nrs1": lrs_predictions["nrs1"],
            "nrs2": lrs_predictions["nrs2"],
        }

    real_problem = HeterogeneousRetrievalProblem(
        name="synthetic fixture problem",
        parameters=parameters,
        forward_model=forward,
        likelihood=HeterogeneousLikelihood(
            (
                HeterogeneousLikelihoodComponent(
                    name="hrs",
                    prediction_key="hrs",
                    likelihood=prepared,
                ),
            )
        ),
    )
    synthetic_problem = workflow.build_synthetic_problem(
        real_problem,
        workflow.generate_synthetic_hrs_observation(
            observation,
            prepared,
            template,
            {"Kp": 190.0, "dVsys": 0.0, "log10_a": 0.0},
        ),
        lrs_synthetic,
        mode=workflow.HRS_MODE,
    )
    result = workflow.evaluate_truth_null_off_velocity(
        synthetic_problem,
        {"Kp": 190.0, "dVsys": 0.0, "log10_a": 0.0},
        mode=workflow.HRS_MODE,
    )
    assert result["finite"] is True
    assert result["truth_preferred_to_null"] is True
    assert result["truth_preferred_to_off_velocity"] is True


def test_resource_policy_is_strict_and_thread_capped() -> None:
    assert workflow.MAX_THREADS == 6
    assert workflow.MAX_MEMORY_BYTES < workflow.HARD_MEMORY_LIMIT_BYTES
    assert "return" in workflow._clamp_thread_environment.__annotations__
    assert all(1 <= value <= 6 for value in workflow.robert_workflow._thread_values().values())
    report = workflow.run_validation()
    assert report["resource_policy"]["max_threads"] == 6
