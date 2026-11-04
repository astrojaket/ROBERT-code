"""Strict error classification for Virga retrievals and native callbacks."""

from __future__ import annotations

import pytest

from robert_exoplanets import (
    MultiDatasetRetrievalProblem,
    Observation,
    ObservationCollection,
    ObservationDataset,
    RetrievalParameter,
    RetrievalParameterSet,
    UniformPrior,
)
from robert_exoplanets.core import (
    RobertConfigError,
    RobertCoverageError,
    RobertDataError,
    RobertValidationError,
)
from robert_exoplanets.forward.virga_cloud import (
    VirgaConfigurationError,
    VirgaInvalidProposalError,
    _is_virga_202_low_kzz_no_solution,
    _virga_compute_failure,
)


VIRGA_METADATA = {"cloud_model": "primary:virga_ackerman_marley_eddy_sed"}
INVALID_FLOOR = -123.0


def _multi_dataset_problem(error, *, metadata=None, strict_virga_errors=None):
    observation = Observation.from_arrays(
        wavelength=[1.0],
        flux=[1.0],
        uncertainty=[0.1],
    )

    def forward_model(parameters):
        raise error

    return MultiDatasetRetrievalProblem(
        name="error-policy-test",
        observations=ObservationCollection(
            (ObservationDataset("test", observation),)
        ),
        parameters=RetrievalParameterSet(
            (RetrievalParameter("x", UniformPrior(0.0, 1.0)),)
        ),
        forward_model=forward_model,
        invalid_loglike=INVALID_FLOOR,
        metadata={} if metadata is None else metadata,
        strict_virga_errors=strict_virga_errors,
    )


def test_virga_problem_selects_strict_policy_from_cloud_model_metadata() -> None:
    problem = _multi_dataset_problem(
        RuntimeError("not evaluated"), metadata=VIRGA_METADATA
    )

    assert problem.strict_virga_errors is True
    assert problem.metadata["virga_error_policy"] == (
        "strict_known_invalid_proposals_only"
    )


@pytest.mark.parametrize(
    "error",
    [
        VirgaInvalidProposalError("unsupported low-Kzz fall-speed solution"),
    ],
)
def test_strict_virga_policy_maps_explicit_invalid_proposals_to_floor(error) -> None:
    problem = _multi_dataset_problem(error, metadata=VIRGA_METADATA)

    assert problem.log_likelihood_from_vector([0.5]) == INVALID_FLOOR


@pytest.mark.parametrize(
    "error",
    [
        RobertConfigError("missing configured opacity input"),
        VirgaConfigurationError("missing Virga parameter mapping"),
        RobertDataError("corrupt Mie table"),
        RobertCoverageError("fixed wavelength coverage is insufficient"),
        RobertValidationError("unclassified validation failure"),
        FloatingPointError("unclassified numerical failure"),
        OverflowError("unclassified numerical overflow"),
        RuntimeError("unclassified library failure"),
        ValueError("unclassified Python failure"),
    ],
)
def test_strict_virga_policy_keeps_config_data_and_unknown_errors_fatal(error) -> None:
    problem = _multi_dataset_problem(error, metadata=VIRGA_METADATA)

    with pytest.raises(type(error), match=str(error)):
        problem.log_likelihood_from_vector([0.5])


def test_strict_policy_can_be_overridden_and_legacy_behavior_is_preserved() -> None:
    strict_off = _multi_dataset_problem(
        RobertDataError("legacy invalid model"),
        metadata=VIRGA_METADATA,
        strict_virga_errors=False,
    )
    non_virga = _multi_dataset_problem(ValueError("legacy invalid model"))

    assert strict_off.metadata["virga_error_policy"] == "legacy_invalid_model_errors"
    assert strict_off.log_likelihood_from_vector([0.5]) == INVALID_FLOOR
    assert non_virga.log_likelihood_from_vector([0.5]) == INVALID_FLOOR


@pytest.mark.parametrize(
    "message",
    [
        "Warning: Could not find a physical solution that balances w_convect "
        "(Kzz is so low that particles would need to be smaller than gas atoms "
        "to stay in the pressure layer). Please try using a higher value Kzz.",
        "Warning: Could not find a physical solution for SPHERES (needed in the "
        "alpha calculation) that balances w_convect (Kzz is so low that particles "
        "would need to be smaller than gas atoms to stay in the pressure layer). "
        "Please try using a higher value Kzz.",
    ],
)
def test_only_known_virga_202_low_kzz_failures_are_proposal_invalid(message) -> None:
    error = Exception(message)

    assert _is_virga_202_low_kzz_no_solution(error, "2.0.2")
    assert isinstance(
        _virga_compute_failure(
            error,
            virga_version="2.0.2",
            condensibles=("MgSiO3",),
        ),
        VirgaInvalidProposalError,
    )
    assert not _is_virga_202_low_kzz_no_solution(error, "2.0.3")
    assert isinstance(
        _virga_compute_failure(
            error,
            virga_version="2.0.3",
            condensibles=("MgSiO3",),
        ),
        RobertDataError,
    )


def test_unrecognized_virga_compute_failure_remains_fatal() -> None:
    failure = _virga_compute_failure(
        Exception("could not read mie table"),
        virga_version="2.0.2",
        condensibles=("MgSiO3",),
    )

    assert isinstance(failure, RobertDataError)
    assert "MgSiO3" in str(failure)


def test_virga_failure_match_rejects_extra_message_text() -> None:
    known = (
        "Warning: Could not find a physical solution that balances w_convect "
        "(Kzz is so low that particles would need to be smaller than gas atoms "
        "to stay in the pressure layer). Please try using a higher value Kzz."
    )

    assert not _is_virga_202_low_kzz_no_solution(
        Exception(known + " Invalid Mie input."), "2.0.2"
    )
