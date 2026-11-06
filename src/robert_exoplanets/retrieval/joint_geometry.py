"""Compose retrieval components that use separate atmospheric geometries.

This module provides a small Python API for joint models such as transmission
and thermal emission. Each component keeps its own forward model, observation,
and likelihood. Explicit parameter bindings connect component-local names to
one shared retrieval parameter set.

Theory-informed limits belong in the supplied normalized prior set. This
module does not reject likelihood points by a theory predicate, because that
would leave nested-sampling evidence normalized to the broader prior.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, cast

from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.retrieval.heterogeneous import (
    HeterogeneousLikelihood,
    HeterogeneousLikelihoodComponent,
    HeterogeneousRetrievalProblem,
)
from robert_exoplanets.retrieval.priors import RetrievalParameterSet


JointForwardEvaluator = Callable[[Mapping[str, float]], object]


def _names(
    values: Sequence[str],
    *,
    label: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise RobertValidationError(f"{label} must be a sequence of names")
    result = tuple(str(value).strip() for value in values)
    if (not allow_empty and not result) or any(not name for name in result):
        raise RobertValidationError(f"{label} must contain non-empty names")
    if len(set(result)) != len(result):
        raise RobertValidationError(f"{label} must not contain duplicates")
    return result


def _opacity_identifiers(component: "JointGeometryComponent") -> Mapping[str, str]:
    values = getattr(component.forward_model, "opacity_identifiers", {})
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise RobertValidationError(
            f"joint component '{component.name}' opacity_identifiers must be a mapping"
        )
    return values


@dataclass(frozen=True)
class JointGeometryComponent:
    """One named prediction and likelihood within a joint retrieval.

    ``required_parameters`` lists every local parameter needed by the forward
    model. ``likelihood_parameters`` lists local nuisance or other values that
    the likelihood needs. ``parameter_bindings`` must explicitly map the
    union of these local names to global names in the shared prior set. A
    global name can appear in more than one component to tie those parameters.
    """

    name: str
    forward_model: JointForwardEvaluator
    likelihood: object
    required_parameters: tuple[str, ...]
    parameter_bindings: Mapping[str, str]
    observation: object | None = None
    likelihood_parameters: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    _model_parameter_names: tuple[str, ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise RobertValidationError("joint component name must not be empty")
        if not callable(self.forward_model):
            raise RobertValidationError(
                f"joint component '{name}' forward_model must be callable"
            )
        if not callable(getattr(self.likelihood, "loglike", None)):
            raise RobertValidationError(
                f"joint component '{name}' likelihood must expose callable loglike"
            )

        required = _names(
            self.required_parameters,
            label=f"joint component '{name}' required_parameters",
            allow_empty=True,
        )
        likelihood_parameters = _names(
            self.likelihood_parameters,
            label=f"joint component '{name}' likelihood_parameters",
            allow_empty=True,
        )
        for attribute in (
            "offset_parameter",
            "jitter_parameter",
            "uncertainty_scale_parameter",
        ):
            local_parameter = getattr(self.likelihood, attribute, None)
            if local_parameter is not None and local_parameter not in likelihood_parameters:
                raise RobertValidationError(
                    f"joint component '{name}' likelihood parameter "
                    f"'{local_parameter}' must be listed in likelihood_parameters"
                )
        local_names = set(required) | set(likelihood_parameters)
        bindings = {
            str(local).strip(): str(global_name).strip()
            for local, global_name in self.parameter_bindings.items()
        }
        if any(not local for local in bindings):
            raise RobertValidationError(
                f"joint component '{name}' parameter binding names must not be empty"
            )
        if any(not global_name for global_name in bindings.values()):
            raise RobertValidationError(
                f"joint component '{name}' global parameter names must not be empty"
            )
        missing = sorted(local_names - set(bindings))
        extra = sorted(set(bindings) - local_names)
        if missing or extra:
            details = []
            if missing:
                details.append("missing bindings for " + ", ".join(missing))
            if extra:
                details.append("unused bindings for " + ", ".join(extra))
            raise RobertValidationError(
                f"joint component '{name}' parameter bindings are inconsistent: "
                + "; ".join(details)
            )

        model_parameters = getattr(self.forward_model, "required_parameters", None)
        if callable(model_parameters):
            model_parameters = model_parameters()
        if model_parameters is None:
            model_names = required
        else:
            model_names = _names(
                model_parameters,
                label=f"joint component '{name}' model required_parameters",
                allow_empty=True,
            )
            if set(model_names) != set(required):
                raise RobertValidationError(
                    f"joint component '{name}' required_parameters must match its "
                    "forward model's required_parameters"
                )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "required_parameters", required)
        object.__setattr__(self, "likelihood_parameters", likelihood_parameters)
        object.__setattr__(self, "parameter_bindings", immutable_mapping(bindings))
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))
        object.__setattr__(self, "_model_parameter_names", model_names)

    @property
    def global_parameter_names(self) -> tuple[str, ...]:
        """Global names used by this component, in binding order."""

        return tuple(dict.fromkeys(self.parameter_bindings.values()))

    def local_parameters(
        self,
        global_parameters: Mapping[str, float],
        *,
        names: tuple[str, ...] | None = None,
    ) -> dict[str, float]:
        """Map selected local names from the joint parameter mapping."""

        selected_names = (
            tuple(self.parameter_bindings)
            if names is None
            else names
        )
        unknown_local = tuple(
            local for local in selected_names if local not in self.parameter_bindings
        )
        if unknown_local:
            raise RobertValidationError(
                f"joint component '{self.name}' has no binding for local parameter(s): "
                + ", ".join(unknown_local)
            )
        missing = tuple(
            self.parameter_bindings[local]
            for local in selected_names
            if self.parameter_bindings[local] not in global_parameters
        )
        if missing:
            raise RobertValidationError(
                f"joint component '{self.name}' is missing global parameter(s): "
                + ", ".join(dict.fromkeys(missing))
            )
        return {
            local: float(global_parameters[self.parameter_bindings[local]])
            for local in selected_names
        }

    def predict(self, global_parameters: Mapping[str, float]) -> object:
        """Evaluate the component with only its declared model parameters."""

        local = self.local_parameters(
            global_parameters,
            names=self._model_parameter_names,
        )
        return self.forward_model(local)


@dataclass(frozen=True)
class JointGeometryForwardModel:
    """Evaluate a named set of component models from one global mapping."""

    components: tuple[JointGeometryComponent, ...]

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components:
            raise RobertValidationError(
                "joint forward model must contain at least one component"
            )
        if any(not isinstance(component, JointGeometryComponent) for component in components):
            raise RobertValidationError(
                "joint forward model components must be JointGeometryComponent instances"
            )
        names = tuple(component.name for component in components)
        if len(set(names)) != len(names):
            raise RobertValidationError(
                "joint component names must be unique"
            )
        object.__setattr__(self, "components", components)

    def __call__(
        self,
        global_parameters: Mapping[str, float],
    ) -> Mapping[str, object]:
        """Return predictions keyed by component name."""

        return immutable_mapping(
            {
                component.name: component.predict(global_parameters)
                for component in self.components
            }
        )


@dataclass(frozen=True)
class _BoundLikelihood:
    """Adapt one likelihood to the global-to-local bindings of its component."""

    component: JointGeometryComponent

    @property
    def name(self) -> str:
        return str(
            getattr(
                self.component.likelihood,
                "name",
                type(self.component.likelihood).__name__,
            )
        )

    @property
    def manifest_wrapped_likelihood(self) -> object:
        """Expose the likelihood through the manifest's explicit wrapper API."""

        return self.component.likelihood

    @property
    def manifest_parameter_bindings(self) -> Mapping[str, str]:
        """Return the local-to-global bindings used by the wrapped likelihood."""

        return self.component.parameter_bindings

    @property
    def manifest_likelihood_parameters(self) -> tuple[str, ...]:
        """Return local nuisance parameters passed to the wrapped likelihood."""

        return self.component.likelihood_parameters

    def loglike(self, prediction: object, *args: object) -> float:
        if len(args) == 2:
            observation, global_parameters = args
            if not isinstance(global_parameters, Mapping):
                raise RobertValidationError(
                    "joint likelihood parameters must be a mapping"
                )
            local_parameters = self.component.local_parameters(
                global_parameters,
                names=self.component.likelihood_parameters,
            )
            return float(
                cast(Any, self.component.likelihood).loglike(
                    prediction,
                    observation,
                    local_parameters,
                )
            )
        if len(args) == 1:
            (global_parameters,) = args
            if not isinstance(global_parameters, Mapping):
                raise RobertValidationError(
                    "joint likelihood parameters must be a mapping"
                )
            local_parameters = self.component.local_parameters(
                global_parameters,
                names=self.component.likelihood_parameters,
            )
            return float(
                cast(Any, self.component.likelihood).loglike(
                    prediction,
                    local_parameters,
                )
            )
        raise RobertValidationError(
            "joint likelihood must receive either parameters or observation and parameters"
        )


def build_joint_geometry_problem(
    *,
    name: str,
    components: tuple[JointGeometryComponent, ...],
    parameters: RetrievalParameterSet,
    metadata: Mapping[str, str] | None = None,
    opacity_identifiers: Mapping[str, str] | None = None,
    invalid_loglike: float = float("-inf"),
) -> HeterogeneousRetrievalProblem:
    """Build a heterogeneous retrieval problem with one shared prior set.

    The global ``parameters`` object is the sole source of priors. Every
    parameter in it must be used by at least one selected component, and every
    component binding must name a parameter in it. To construct a component
    subset, pass that subset and the matching subset prior set.

    Theory-aware scenarios should provide their supported parameter ranges in
    this prior set and record the theory and scenario in ``metadata``. Pass a
    separate broad-prior set for baseline sensitivity checks. The builder
    does not apply theory masks in the likelihood, so sampler evidence keeps
    the normalization defined by the supplied priors.
    """

    if not isinstance(parameters, RetrievalParameterSet):
        raise RobertValidationError(
            "joint geometry parameters must be a RetrievalParameterSet"
        )
    forward_model = JointGeometryForwardModel(tuple(components))
    parameter_names = set(parameters.names)
    used_names = {
        global_name
        for component in forward_model.components
        for global_name in component.global_parameter_names
    }
    unknown = sorted(used_names - parameter_names)
    unused = sorted(parameter_names - used_names)
    if unknown:
        raise RobertValidationError(
            "joint component bindings name parameters absent from the prior set: "
            + ", ".join(unknown)
        )
    if unused:
        raise RobertValidationError(
            "joint prior set contains parameters unused by the selected components: "
            + ", ".join(unused)
        )

    likelihood_components = tuple(
        HeterogeneousLikelihoodComponent(
            name=component.name,
            prediction_key=component.name,
            likelihood=_BoundLikelihood(component),
            observation=component.observation,
            metadata={
                **dict(component.metadata),
                "required_parameters": ",".join(component.required_parameters),
                "likelihood_parameters": ",".join(
                    component.likelihood_parameters
                ),
                "parameter_bindings": ";".join(
                    f"{local}={global_name}"
                    for local, global_name in component.parameter_bindings.items()
                ),
            },
        )
        for component in forward_model.components
    )
    discovered_opacity_identifiers = {
        f"{component.name}:{species}": str(identifier)
        for component in forward_model.components
        for species, identifier in _opacity_identifiers(component).items()
    }
    discovered_opacity_identifiers.update(
        {} if opacity_identifiers is None else opacity_identifiers
    )
    return HeterogeneousRetrievalProblem(
        name=name,
        parameters=parameters,
        forward_model=forward_model,
        likelihood=HeterogeneousLikelihood(likelihood_components),
        invalid_loglike=invalid_loglike,
        metadata={} if metadata is None else metadata,
        opacity_identifiers=discovered_opacity_identifiers,
    )


__all__ = [
    "JointGeometryComponent",
    "JointGeometryForwardModel",
    "build_joint_geometry_problem",
]
