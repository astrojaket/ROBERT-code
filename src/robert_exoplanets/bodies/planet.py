"""Planet domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.core.exceptions import RobertValidationError


@dataclass(frozen=True)
class Planet:
    """Planetary metadata required by ROBERT workflows."""

    name: str
    radius_m: float | None = None
    mass_kg: float | None = None
    gravity_m_s2: float | None = None
    semi_major_axis_m: float | None = None
    system_distance_m: float | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("planet name must not be empty")
        for field_name in (
            "radius_m",
            "mass_kg",
            "gravity_m_s2",
            "semi_major_axis_m",
            "system_distance_m",
        ):
            value = getattr(self, field_name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise RobertValidationError(f"{field_name} must be finite and positive when provided")
        if self.gravity_m_s2 is None and not (self.mass_kg is not None and self.radius_m is not None):
            raise RobertValidationError("planet requires gravity_m_s2 or both mass_kg and radius_m")
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def has_direct_gravity(self) -> bool:
        """Whether gravity was supplied directly."""

        return self.gravity_m_s2 is not None
