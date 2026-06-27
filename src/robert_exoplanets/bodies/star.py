"""Star domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from robert_exoplanets.core.exceptions import RobertValidationError
from robert_exoplanets.core.spectrum import Spectrum


@dataclass(frozen=True)
class Star:
    """Stellar metadata used by ROBERT workflows."""

    name: str
    radius_m: float | None = None
    effective_temperature_k: float | None = None
    log_g_cgs: float | None = None
    metallicity_dex: float | None = None
    spectrum: Spectrum | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("star name must not be empty")
        if self.radius_m is not None and self.radius_m <= 0:
            raise RobertValidationError("radius_m must be positive when provided")
        if self.effective_temperature_k is not None and self.effective_temperature_k <= 0:
            raise RobertValidationError("effective_temperature_k must be positive when provided")
        object.__setattr__(self, "metadata", dict(self.metadata))
