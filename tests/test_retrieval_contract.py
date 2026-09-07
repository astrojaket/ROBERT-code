"""Tests for the shared retrieval problem contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from robert_exoplanets import (
    Observation,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    Spectrum,
    UniformPrior,
)
from robert_exoplanets.retrieval.heterogeneous import HeterogeneousRetrievalProblem
from robert_exoplanets.retrieval.multi_dataset import MultiDatasetRetrievalProblem
from robert_exoplanets.retrieval.priors import Prior
from robert_exoplanets.retrieval.protocols import (
    OptimalEstimationProblem,
    SamplerRetrievalProblem,
)

if TYPE_CHECKING:
    from robert_exoplanets.metal.problem import MetalRetrievalProblem


if TYPE_CHECKING:

    def _static_contract_conformance(
        cpu: RetrievalProblem,
        mixed: MultiDatasetRetrievalProblem,
        heterogeneous: HeterogeneousRetrievalProblem,
        device: MetalRetrievalProblem,
    ) -> None:
        """Keep concrete problem implementations assignable to the protocols."""

        sampler_problems: tuple[
            SamplerRetrievalProblem,
            SamplerRetrievalProblem,
            SamplerRetrievalProblem,
            SamplerRetrievalProblem,
        ] = (cpu, mixed, heterogeneous, device)
        oe_cpu: OptimalEstimationProblem = cpu
        oe_mixed: OptimalEstimationProblem = mixed
        del sampler_problems, oe_cpu, oe_mixed


def _problem() -> RetrievalProblem:
    observation = Observation.from_arrays(
        wavelength=[1.0, 2.0],
        flux=[1.0, 1.0],
        uncertainty=[0.1, 0.1],
    )
    parameters = RetrievalParameterSet(
        (RetrievalParameter("level", cast(Prior, UniformPrior(0.0, 2.0))),)
    )
    return RetrievalProblem(
        name="protocol-test",
        observation=observation,
        parameters=parameters,
        forward_model=lambda values: Spectrum.from_arrays(
            observation.wavelength,
            np.full(observation.n_points, values["level"]),
            unit=observation.flux_unit,
            observable=observation.observable,
            wavelength_unit=observation.wavelength_unit,
        ),
    )


def test_cpu_problem_exposes_sampler_and_oe_contracts() -> None:
    problem = _problem()

    assert isinstance(problem, SamplerRetrievalProblem)
    assert isinstance(problem, OptimalEstimationProblem)
    assert problem.parameter_names == ("level",)
    np.testing.assert_allclose(problem.prior_transform([0.5]), [1.0])
    assert problem.log_likelihood_from_vector([1.0]) == 0.0
