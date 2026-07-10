"""Star domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from robert_exoplanets.core._immutability import immutable_mapping
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
        if self.radius_m is not None and (not np.isfinite(self.radius_m) or self.radius_m <= 0):
            raise RobertValidationError("radius_m must be finite and positive when provided")
        if self.effective_temperature_k is not None and (
            not np.isfinite(self.effective_temperature_k) or self.effective_temperature_k <= 0
        ):
            raise RobertValidationError("effective_temperature_k must be finite and positive when provided")
        if self.log_g_cgs is not None and not np.isfinite(self.log_g_cgs):
            raise RobertValidationError("log_g_cgs must be finite when provided")
        if self.metallicity_dex is not None and not np.isfinite(self.metallicity_dex):
            raise RobertValidationError("metallicity_dex must be finite when provided")
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))
