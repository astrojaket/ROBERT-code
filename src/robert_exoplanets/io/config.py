"""Typed configuration skeleton for ROBERT v0.2."""

from __future__ import annotations

from dataclasses import dataclass

from robert_exoplanets.bodies import Planet, Star
from robert_exoplanets.core.exceptions import RobertConfigError
from robert_exoplanets.instruments import Observation


@dataclass(frozen=True)
class RobertConfig:
    """Minimal typed run configuration.

    Full YAML/Pydantic parsing is a later milestone. This object establishes the
    v0.2 boundary where validated config should become domain objects.
    """

    run_name: str
    planet: Planet
    observations: tuple[Observation, ...]
    star: Star | None = None

    def __post_init__(self) -> None:
        if not self.run_name:
            raise RobertConfigError("run_name must not be empty")
        if not self.observations:
            raise RobertConfigError("at least one observation is required")
        object.__setattr__(self, "observations", tuple(self.observations))
