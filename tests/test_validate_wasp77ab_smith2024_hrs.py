"""Dry-run and integrity tests for the Smith WASP-77Ab HRS workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets.instruments.time_resolved_high_resolution import (
    HighResolutionEmissionTemplate,
    TimeResolvedHighResolutionObservation,
)

from examples import validate_wasp77ab_smith2024_hrs as workflow


def test_thread_configuration_is_clamped_to_three(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in workflow.THREAD_VARIABLES:
        monkeypatch.setenv(name, "99")
    workflow._configure_thread_limits()
    assert workflow._thread_values() == {
        name: 3 for name in workflow.THREAD_VARIABLES
    }


def test_dry_run_writes_a_no_io_contract_report(tmp_path: Path) -> None:
    report_path = tmp_path / "smith_hrs_dry_run.json"
    report = workflow.run_validation(
        data_root=tmp_path / "does_not_exist",
        report_path=report_path,
        plot_path=None,
        dry_run=True,
    )
    assert report["status"] == "dry_run"
    assert report["inputs"]["loaded"] is False
    assert report["paper"]["ccf_snr_anchors"] == {
        "full": 9.4,
        "h2o_only": 9.2,
        "co_only": 3.6,
    }
    assert report["method"]["response_chain"] == [
        "rotational-broadening",
        "gaussian-high-resolution",
    ]
    assert report["response"]["operator_family"] == (
        "ROBERT sparse spectral response operators"
    )
    assert report["method"]["pixel_integration"] is False
    assert report_path.is_file()


def test_file_identity_rejects_wrong_sha_before_scientific_load(tmp_path: Path) -> None:
    path = tmp_path / workflow.CUBE_FILENAME
    path.write_bytes(b"not a Smith cube")
    with pytest.raises(ValueError, match="size mismatch"):
        workflow._verify_file(path, workflow._FILE_SPECS[workflow.CUBE_FILENAME])


def test_two_gib_process_limit_is_strict() -> None:
    with pytest.raises(ValueError, match="strictly below 2 GiB"):
        workflow.run_validation(
            report_path=None,
            plot_path=None,
            max_memory_bytes=workflow.MAX_MEMORY_BYTES,
            dry_run=True,
        )


def test_real_diagnostic_uses_smith_pre_eclipse_table3_anchor() -> None:
    assert workflow.PAPER_PRE_PARAMETERS == {
        "Kp": pytest.approx(190.74),
        "dVsys": pytest.approx(-5.27),
        "dphi": pytest.approx(0.0),
        "log10_a": pytest.approx(0.15),
    }
    assert workflow.LOCAL_SCAN_KP == (190.74,)
    assert workflow.LOCAL_SCAN_DVSYS == (-7.27, -5.27, -3.27)


def test_optional_plot_writer_is_small_and_hashed(tmp_path: Path) -> None:
    plot_path = tmp_path / "smith_hrs_ccf.png"
    digest = workflow._write_plot(
        plot_path,
        {
            "dVsys_km_s": [-2.0, 0.0, 2.0],
            "raw_summed_ccf": [[1.0, 2.0, 1.5]],
        },
    )
    assert plot_path.is_file()
    assert len(digest) == 64


def _synthetic_inputs() -> tuple[
    TimeResolvedHighResolutionObservation,
    HighResolutionEmissionTemplate,
]:
    """Build a small, dense HRS grid for response-operator tests."""

    observation_wavelengths = np.array(
        [
            np.linspace(1.94, 1.98, 12),
            np.linspace(2.02, 2.06, 12),
        ]
    )
    observation = TimeResolvedHighResolutionObservation.from_arrays(
        order_wavelengths=observation_wavelengths,
        flux=np.zeros((2, 3, 12)),
        phase=np.array([0.10, 0.30, 0.50]),
        fixed_velocity_km_s=np.array([1.0, -2.0, 0.5]),
        time_bjd=np.array([2_459_197.5, 2_459_197.6, 2_459_197.7]),
    )
    wavelength = np.exp(np.linspace(np.log(1.85), np.log(2.15), 1601))
    narrow_line = np.exp(-0.5 * ((wavelength - 2.0) / 2.0e-5) ** 2)
    template = HighResolutionEmissionTemplate(
        wavelength=wavelength,
        planet_flux=1.0 + narrow_line,
        stellar_flux=np.ones(wavelength.size),
        wavelength_unit="micron",
        flux_ratio_scale=1.0,
        name="synthetic-r250000-template",
    )
    return observation, template


def test_response_chain_is_gray_then_gaussian_without_pixel_integration() -> None:
    observation, template = _synthetic_inputs()
    processed, metadata = workflow._prepare_template_for_hrs(template, observation)

    assert workflow.HRS_RESPONSE_STAGES == (
        "rotational-broadening",
        "gaussian-high-resolution",
    )
    assert workflow.HRS_RESPONSE_PIXEL_INTEGRATION is False
    assert metadata["stage_order"] == list(workflow.HRS_RESPONSE_STAGES)
    assert metadata["applied_operator_names"] == list(workflow.HRS_RESPONSE_STAGES)
    assert metadata["pixel_integration"] is False
    assert processed.metadata["response_pixel_integration"] == "false"
    assert "pixel-bin-integration" not in metadata["applied_operator_names"]


def test_processed_template_covers_prior_and_table3_anchor_velocities() -> None:
    observation, template = _synthetic_inputs()
    processed, metadata = workflow._prepare_template_for_hrs(template, observation)
    velocity_metadata = metadata["velocity_coverage"]
    prior_lower, prior_upper = velocity_metadata["prior_velocity_bounds_km_s"]
    anchor_lower, anchor_upper = velocity_metadata["anchor_velocity_bounds_km_s"]
    assert prior_lower <= anchor_lower <= anchor_upper <= prior_upper

    trial_velocities = (
        prior_lower,
        prior_upper,
        anchor_lower,
        anchor_upper,
        workflow.PAPER_PRE_PARAMETERS["dVsys"],
    )
    for velocity in trial_velocities:
        for wavelengths in observation.order_wavelengths:
            ratio = processed.interpolate_flux_ratio(
                wavelengths,
                velocity_km_s=float(velocity),
                doppler_mode="smith_nonrelativistic",
            )
            assert np.all(np.isfinite(ratio))


def test_gray_and_gaussian_response_changes_a_narrow_line_shape() -> None:
    observation, template = _synthetic_inputs()
    processed, metadata = workflow._prepare_template_for_hrs(template, observation)
    raw_ratio = np.interp(
        processed.wavelength,
        template.wavelength,
        template.flux_ratio,
    )
    difference = np.max(np.abs(processed.flux_ratio - raw_ratio))
    assert difference > 1.0e-3
    assert processed.flux_ratio.max() < template.flux_ratio.max()
    assert metadata["rotation_vsini_km_s"] == pytest.approx(4.2)
    assert metadata["gaussian_lsf_resolving_power"] == pytest.approx(45_000.0)
