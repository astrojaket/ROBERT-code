"""Shared physical-state orchestration for multiple datasets.

Retrievals should normally prepare correlated-k opacity with ``exo_k`` for
each instrument mode. ``MultiDatasetEmissionForwardModel`` evaluates
temperature and chemistry once, then retains each mode-specific opacity and RT
solve. The native-spectrum path remains useful for visualization and
diagnostics, where a continuous high-resolution curve is useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

from robert_exoplanets.atmosphere import AtmosphereBuilder
from robert_exoplanets.core import RobertValidationError, Spectrum
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.instruments import (
    ObservationCollection,
    PreparedTopHatObservationResponse,
    TopHatObservationResponse,
)

from .emission import ParameterizedClearSkyEmissionForwardModel

NativeSpectrumEvaluator = Callable[[Mapping[str, float]], Spectrum]


class PreparedSpectrumResponse(Protocol):
    """Prepared response mapping a model spectrum to an observation grid."""

    def observe(self, spectrum: Spectrum) -> Spectrum: ...


@dataclass(frozen=True)
class MultiDatasetEmissionForwardModel:
    """Evaluate one atmosphere through several prepared correlated-k models."""

    models: Mapping[str, ParameterizedClearSkyEmissionForwardModel]
    responses: Mapping[str, PreparedSpectrumResponse] = field(default_factory=dict)
    atmosphere_builder: AtmosphereBuilder = field(init=False, repr=False)
    required_parameters: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        models = {str(name): model for name, model in self.models.items()}
        if not models or any(not name for name in models):
            raise RobertValidationError(
                "shared-atmosphere multi-dataset models require non-empty names"
            )
        if any(
            not isinstance(model, ParameterizedClearSkyEmissionForwardModel)
            for model in models.values()
        ):
            raise RobertValidationError(
                "shared-atmosphere models must be parameterized clear-sky emission models"
            )
        reference = next(iter(models.values()))
        for model in models.values():
            if model.atmosphere_builder is not reference.atmosphere_builder:
                raise RobertValidationError(
                    "shared-atmosphere models must use the same AtmosphereBuilder instance"
                )
            if model.required_parameters != reference.required_parameters:
                raise RobertValidationError(
                    "shared-atmosphere models must require the same parameters"
                )
        responses = {str(name): response for name, response in self.responses.items()}
        if responses and set(responses) != set(models):
            raise RobertValidationError(
                "shared-atmosphere response names must match model names"
            )
        if any(not callable(getattr(response, "observe", None)) for response in responses.values()):
            raise RobertValidationError("prepared responses must implement observe")
        object.__setattr__(self, "models", immutable_mapping(models))
        object.__setattr__(self, "responses", immutable_mapping(responses))
        object.__setattr__(self, "atmosphere_builder", reference.atmosphere_builder)
        object.__setattr__(self, "required_parameters", reference.required_parameters)

    def __call__(self, parameters: Mapping[str, float]) -> Mapping[str, Spectrum]:
        reference = next(iter(self.models.values()))
        parameter_values = reference.validated_parameters(parameters)
        atmosphere = self.atmosphere_builder.build(parameter_values)
        spectra = {
            name: model.evaluate_atmosphere(atmosphere, parameter_values)
            for name, model in self.models.items()
        }
        if self.responses:
            spectra = {
                name: self.responses[name].observe(spectrum)
                for name, spectrum in spectra.items()
            }
        return immutable_mapping(spectra)


@dataclass(frozen=True)
class NativeSpectrumMultiDatasetPrediction:
    """Named observed-grid spectra from one shared physical model."""

    native_spectrum: Spectrum
    spectra: Mapping[str, Spectrum]

    def __post_init__(self) -> None:
        object.__setattr__(self, "spectra", immutable_mapping(self.spectra))


@dataclass(frozen=True)
class NativeSpectrumMultiDatasetForwardModel:
    """Map one native spectrum to datasets for visualization or diagnostics.

    This is not the default correlated-k retrieval path.  Inference should use
    mode-specific opacity tables prepared with ``exo_k`` before radiative
    transfer; ``exo_k`` cannot recompress an already evaluated flux spectrum.
    """

    native_model: NativeSpectrumEvaluator
    observations: ObservationCollection
    response: TopHatObservationResponse = field(
        default_factory=TopHatObservationResponse
    )
    _prepared: tuple[tuple[str, PreparedTopHatObservationResponse], ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_prepared",
            tuple(
                (dataset.name, self.response.prepare(dataset.observation))
                for dataset in self.observations.datasets
            ),
        )

    def __call__(
        self,
        parameters: Mapping[str, float],
    ) -> NativeSpectrumMultiDatasetPrediction:
        native = self.native_model(parameters)
        spectra = {name: response.observe(native) for name, response in self._prepared}
        return NativeSpectrumMultiDatasetPrediction(
            native_spectrum=native,
            spectra=spectra,
        )


__all__ = [
    "MultiDatasetEmissionForwardModel",
    "NativeSpectrumMultiDatasetForwardModel",
    "NativeSpectrumMultiDatasetPrediction",
]
