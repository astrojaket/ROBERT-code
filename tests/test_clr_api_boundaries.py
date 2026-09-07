"""CLR transforms must not be used as scalar densities or Gaussian priors."""

import numpy as np
import pytest

from robert_exoplanets import (
    CenteredLogRatioPrior,
    Observation,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    run_optimal_estimation,
)
from robert_exoplanets.core import RobertConfigError


@pytest.fixture
def clr_problem() -> RetrievalProblem:
    def unused_forward(parameters):
        pytest.fail("unsupported CLR API reached the forward model")

    return RetrievalProblem(
        name="clr-boundary",
        observation=Observation.from_arrays(
            wavelength=[1.0], flux=[1.0], uncertainty=[0.1]
        ),
        parameters=RetrievalParameterSet(
            (RetrievalParameter("log_A", CenteredLogRatioPrior()),)
        ),
        forward_model=unused_forward,
    )


def test_clr_nested_transform_remains_available(clr_problem) -> None:
    vector = clr_problem.prior_transform([0.5])
    np.testing.assert_allclose(vector, [np.log10(0.5)])
    assert clr_problem.parameter_mapping(vector)["log_A"] == vector[0]


@pytest.mark.parametrize("method", ["log_prior_from_vector", "log_posterior_from_vector"])
def test_clr_density_apis_reject_unsupported_joint_density(clr_problem, method) -> None:
    with pytest.raises(RobertConfigError, match="CLR prior density is not implemented"):
        getattr(clr_problem, method)(clr_problem.prior_transform([0.5]))


def test_direct_python_oe_rejects_clr_before_forward_evaluation(clr_problem) -> None:
    with pytest.raises(RobertConfigError, match="CLR priors require direct nested sampling"):
        run_optimal_estimation(clr_problem)
