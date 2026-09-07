"""Tests for public high-resolution and combined-resolution API exports."""

from __future__ import annotations

from typing import get_args

import robert_exoplanets as robert
import robert_exoplanets.forward as forward
import robert_exoplanets.instruments as instruments
import robert_exoplanets.likelihoods as likelihoods
import robert_exoplanets.opacity as opacity
import robert_exoplanets.validation as validation


def test_high_resolution_symbols_are_exported_at_subpackage_and_root_levels() -> None:
    modules = (instruments, forward, likelihoods, opacity, validation)
    for module in modules:
        assert not [name for name in module.__all__ if not hasattr(module, name)]

    expected = {
        "DopplerShiftResponse",
        "HighResolutionResponseChain",
        "VelocityParameterizedHighResolutionResponse",
        "ParameterizedMultiDatasetResponseForwardModel",
        "LineByLineOpacitySource",
        "CalibratedDirectFluxLikelihood",
        "PolynomialContinuumLikelihood",
        "CrossCorrelationLikelihood",
        "MixedMultiDatasetLikelihood",
        "evaluate_multi_dataset_injection_recovery",
        "inject_spectrum_collection",
    }
    assert expected.issubset(set(robert.__all__))
    assert all(hasattr(robert, name) for name in expected)


def test_public_aliases_refer_to_their_canonical_implementations() -> None:
    assert instruments.DopplerShiftResponse is instruments.RelativisticDopplerResponse
    assert instruments.ResponseChain is instruments.HighResolutionResponseChain
    assert likelihoods.DirectFluxLikelihood is likelihoods.CalibratedDirectFluxLikelihood
    assert (
        likelihoods.ProfiledPolynomialContinuumLikelihood
        is likelihoods.PolynomialContinuumLikelihood
    )
    assert opacity.PreparedLineByLineOpacity in get_args(opacity.PreparedOpacity)
