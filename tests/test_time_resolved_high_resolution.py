"""Tests for the Smith/Brogi-Line time-resolved HRS likelihood."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.instruments import (
    HighResolutionEmissionTemplate,
    TimeResolvedHighResolutionObservation,
)
from robert_exoplanets.likelihoods import (
    TimeResolvedHighResolutionLikelihood,
)


def _observation(
    flux: np.ndarray,
    *,
    phase: np.ndarray | None = None,
    fixed_velocity: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> TimeResolvedHighResolutionObservation:
    n_orders, n_frames, n_pixels = flux.shape
    wavelengths = np.tile(
        np.linspace(1.0, 1.0 + 0.01 * (n_pixels - 1), n_pixels),
        (n_orders, 1),
    )
    return TimeResolvedHighResolutionObservation.from_arrays(
        order_wavelengths=wavelengths,
        flux=flux,
        phase=np.zeros(n_frames) if phase is None else phase,
        fixed_velocity_km_s=(
            np.zeros(n_frames) if fixed_velocity is None else fixed_velocity
        ),
        time_bjd=2_459_000.0 + np.arange(n_frames),
        mask=mask,
    )


def _template(
    wavelength: np.ndarray,
    ratio: np.ndarray,
) -> HighResolutionEmissionTemplate:
    return HighResolutionEmissionTemplate(
        wavelength=wavelength,
        planet_flux=ratio,
        stellar_flux=np.ones_like(ratio),
    )


def test_observation_is_immutable_and_combines_masks() -> None:
    flux = np.ones((2, 3, 4))
    observation = _observation(
        flux,
        mask=np.ones((2, 4), dtype=bool),
    )
    assert observation.mask.shape == flux.shape
    assert observation.mask.flags.writeable is False
    assert observation.n_points == flux.size
    with pytest.raises(ValueError):
        observation.mask[0, 0, 0] = False

    with pytest.raises(RobertValidationError, match="strictly monotonic"):
        TimeResolvedHighResolutionObservation.from_arrays(
            order_wavelengths=[[1.0, 1.1, 1.05, 1.2]],
            flux=[[[1.0, 1.0, 1.0, 1.0]]],
            phase=[0.0],
            fixed_velocity_km_s=[0.0],
            time_bjd=[1.0],
        )


def test_brogi_line_formula_matches_hand_calculation() -> None:
    flux = np.asarray(
        [
            [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0]],
        ]
    )
    observation = _observation(flux, phase=np.asarray([0.0, 0.25]))
    wavelength = observation.order_wavelengths[0]
    template = _template(wavelength, np.full(wavelength.size, 0.1))
    prepared = TimeResolvedHighResolutionLikelihood(n_components=0).prepare(observation)

    actual = prepared.loglike(template)
    expected = 0.0
    for values in flux[0]:
        f2 = float(np.dot(values, values))
        # With zero removed components the Smith low-rank data scale is zero,
        # so the trial model is zero and the statistic reduces to the raw
        # PCA-residual norm. Only the model is mean-subtracted.
        expected += -0.5 * values.size * np.log(f2 / values.size)
    assert actual == pytest.approx(expected, rel=1.0e-12)
    assert prepared.loglike_terms(template).shape == (1, 2)


def test_model_reprocessing_uses_exact_same_component_count() -> None:
    flux = np.asarray(
        [
            [
                [1.0, 2.0, 3.0, 5.0],
                [2.0, 1.0, 4.0, 3.0],
                [3.0, 5.0, 1.0, 2.0],
            ]
        ]
    )
    observation = _observation(flux)
    wavelength = observation.order_wavelengths[0]
    template = _template(wavelength, np.full(wavelength.size, 0.2))
    prepared = TimeResolvedHighResolutionLikelihood(n_components=1).prepare(observation)
    model = prepared.evaluate_model(template, {"log10_a": np.log10(2.0)})

    injected = (1.0 + 0.2) * prepared.data_scale[0]
    u, singular_values, vt = np.linalg.svd(injected, full_matrices=False)
    expected_scaling = (u[:, :1] * singular_values[:1]) @ vt[:1]
    np.testing.assert_allclose(model.injected_model[0], injected, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        model.scaling_matrix[0],
        expected_scaling,
        rtol=0.0,
        atol=1.0e-12,
    )
    for frame in range(flux.shape[1]):
        expected_filtered = injected[frame] - expected_scaling[frame]
        expected_filtered -= np.mean(expected_filtered)
        np.testing.assert_allclose(
            model.centered_model[0, frame],
            2.0 * expected_filtered,
            rtol=0.0,
            atol=1.0e-12,
        )


def test_velocity_formula_and_positive_redshift_sign() -> None:
    flux = np.ones((1, 2, 7))
    observation = _observation(
        flux,
        phase=np.asarray([0.0, 0.25]),
        fixed_velocity=np.asarray([10.0, -5.0]),
    )
    prepared = TimeResolvedHighResolutionLikelihood(n_components=0).prepare(observation)
    velocity = prepared.velocity({"Kp": 20.0, "dVsys": 3.0, "dphi": 0.0})
    np.testing.assert_allclose(velocity, [13.0, 18.0], atol=1.0e-12)

    template_grid = np.linspace(1.0, 1.01, 1001)
    ratio = np.exp(-0.5 * ((template_grid - 1.005) / 0.00004) ** 2)
    zero = _template(template_grid, ratio)
    rest = zero.interpolate_flux_ratio(template_grid)
    red = zero.interpolate_flux_ratio(template_grid, velocity_km_s=100.0)
    assert int(np.nanargmax(red)) > int(np.nanargmax(rest))


def test_ccf_diagnostics_and_scale_parameter_alias() -> None:
    flux = np.asarray([[[1.0, 2.0, 4.0, 8.0], [2.0, 3.0, 5.0, 9.0]]])
    # Distinct frame velocities preserve model structure after PCA filtering.
    observation = _observation(flux, fixed_velocity=np.asarray([0.0, 1000.0]))
    template_wavelength = np.linspace(0.9, 1.1, 1001)
    template_ratio = 0.1 + 0.3 * np.exp(
        -0.5 * ((template_wavelength - 1.015) / 0.005) ** 2
    )
    template = _template(template_wavelength, template_ratio)
    prepared = TimeResolvedHighResolutionLikelihood(n_components=1).prepare(observation)
    model = prepared.evaluate_model(template, {"a": 0.0})
    assert np.all(np.ptp(model.centered_model, axis=-1) > 0.0)
    ccf = prepared.ccf(template, {"a": 0.0})
    assert ccf.shape == (1, 2)
    assert np.all(np.isfinite(ccf))
    mapped = prepared.ccf_map(template, [-10.0, 10.0], [-2.0, 2.0])
    assert mapped.shape == (2, 2)


def test_ccf_is_nan_for_an_exact_zero_variance_model() -> None:
    flux = np.asarray([[[1.0, 2.0, 4.0, 8.0], [2.0, 3.0, 5.0, 9.0]]])
    observation = _observation(flux)
    wavelength = observation.order_wavelengths[0]
    template = _template(wavelength, np.zeros(wavelength.size))
    prepared = TimeResolvedHighResolutionLikelihood(n_components=0).prepare(observation)

    ccf = prepared.ccf(template, {"a": 0.0})

    assert ccf.shape == (1, 2)
    assert np.all(np.isnan(ccf))


def test_global_clip_zero_fills_data_but_retains_model_pixel_count() -> None:
    flux = np.asarray(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1000.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    mask = np.ones_like(flux, dtype=bool)
    mask[0, 0, 0] = False
    observation = _observation(flux, mask=mask)
    prepared = TimeResolvedHighResolutionLikelihood(n_components=0).prepare(observation)

    raw = flux[0]
    expected_median = np.median(raw[mask[0]])
    expected_std = np.std(raw[mask[0]])
    expected_clipped = mask[0] & (
        np.abs(raw - expected_median) > 3.0 * expected_std
    )
    assert expected_clipped[0, 5]
    np.testing.assert_array_equal(prepared.clipped_mask[0], expected_clipped)
    assert prepared.n_active_by_order_frame.tolist() == [[5, 6]]
    assert prepared.data_residual[0, 0, 5] == 0.0
    # A clipped data pixel remains a valid model pixel.  With zero removed
    # components this is visible through the retained frame count in the
    # exact Brogi-Line term.
    assert prepared.loglike_terms(
        _template(observation.order_wavelengths[0], np.ones(observation.n_pixels))
    )[0, 0] != 0.0


def test_scalar_loglike_uses_order_streaming_path(monkeypatch: pytest.MonkeyPatch) -> None:
    flux = np.asarray([[[1.0, 2.0, 4.0, 8.0], [2.0, 3.0, 5.0, 9.0]]])
    observation = _observation(flux)
    wavelength = observation.order_wavelengths[0]
    template = _template(wavelength, np.linspace(0.1, 0.4, wavelength.size))
    prepared = TimeResolvedHighResolutionLikelihood(n_components=1).prepare(observation)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("scalar loglike must not allocate evaluate_model diagnostics")

    monkeypatch.setattr(type(prepared), "evaluate_model", forbidden)
    assert np.isfinite(prepared.loglike(template))


def test_direct_numpy_transcription_matches_masked_pca_and_model_reprocessing() -> None:
    """Compare the prepared statistic with a literal Smith-style NumPy calculation."""

    flux = np.asarray(
        [
            [
                [1.0, 2.0, 4.0, 5.0, 3.0],
                [2.0, 5.0, 1.0, 4.0, 3.0],
                [3.0, 1.0, 5.0, 2.0, 4.0],
            ]
        ]
    )
    mask = np.ones_like(flux, dtype=bool)
    mask[0, 1, 2] = False
    observation = _observation(flux, mask=mask)
    wavelength = observation.order_wavelengths[0]
    ratio = np.asarray([0.10, 0.25, 0.05, 0.30, 0.15])
    template = _template(wavelength, ratio)
    parameters = {"log10_a": np.log10(1.3)}
    prepared = TimeResolvedHighResolutionLikelihood(n_components=1).prepare(observation)

    # Data preparation: one SVD per order, then one order-wide median/std clip.
    u, singular_values, vt = np.linalg.svd(flux[0], full_matrices=False)
    data_scale = (u[:, :1] * singular_values[:1]) @ vt[:1]
    data_residual = flux[0] - data_scale
    residual_values = data_residual[mask[0]]
    residual_median = np.median(residual_values)
    residual_std = np.std(residual_values)
    clipped = np.zeros_like(mask[0])
    if residual_std > 0.0 and np.isfinite(residual_std):
        clipped = mask[0] & (
            np.abs(data_residual - residual_median)
            > 3.0 * residual_std
        )
        data_residual[clipped] = 0.0
    expected_f2 = np.asarray(
        [
            np.dot(data_residual[frame, mask[0, frame]], data_residual[frame, mask[0, frame]])
            for frame in range(flux.shape[1])
        ]
    )
    expected_n = np.asarray([np.count_nonzero(mask[0, frame]) for frame in range(flux.shape[1])])

    # Model preparation: inject into the low-rank data scale, re-SVD with the
    # same component count, and subtract the model mean on active pixels only.
    injected = (1.0 + ratio[None, :]) * data_scale
    model_u, model_s, model_vt = np.linalg.svd(injected, full_matrices=False)
    model_scaling = (model_u[:, :1] * model_s[:1]) @ model_vt[:1]
    model_filtered = injected - model_scaling
    model_centered = np.zeros_like(model_filtered)
    for frame in range(flux.shape[1]):
        valid = mask[0, frame]
        values = model_filtered[frame, valid]
        model_centered[frame, valid] = 1.3 * (values - np.mean(values))

    expected_terms = np.empty(flux.shape[1])
    for frame in range(flux.shape[1]):
        valid = mask[0, frame]
        model_values = model_centered[frame, valid]
        data_values = data_residual[frame, valid]
        m2 = np.dot(model_values, model_values)
        cross = np.dot(data_values, model_values)
        expected_terms[frame] = -0.5 * expected_n[frame] * np.log(
            (m2 + expected_f2[frame] - 2.0 * cross) / expected_n[frame]
        )

    np.testing.assert_allclose(prepared.data_scale[0], data_scale, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(prepared.data_residual[0], data_residual, rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(prepared.clipped_mask[0], clipped)
    np.testing.assert_allclose(prepared.f2_by_order_frame[0], expected_f2, rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(prepared.n_active_by_order_frame[0], expected_n)
    assert prepared.data_scale is prepared.pca_scaling
    assert prepared.data_centered is prepared.data_residual

    model = prepared.evaluate_model(template, parameters)
    np.testing.assert_allclose(model.injected_model[0], injected, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(model.scaling_matrix[0], model_scaling, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(model.centered_model[0], model_centered, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        prepared.loglike_terms(template, parameters)[0],
        expected_terms,
        rtol=0.0,
        atol=1.0e-12,
    )
