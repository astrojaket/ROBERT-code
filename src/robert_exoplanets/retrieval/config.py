"""Configuration objects for retrieval workflows."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalConfig:
    """High-level retrieval settings.

    These values describe the shape of a future retrieval run without selecting
    a real atmospheric model, likelihood, or sampler implementation yet.
    """

    target_name: str
    instrument: str
    wavelength_unit: str = "micron"
    flux_unit: str = "eclipse_depth"
    parameters: tuple[str, ...] = field(
        default_factory=lambda: (
            "log_h2o",
            "log_co",
            "temperature",
            "radius_ratio",
        )
    )

    def __post_init__(self) -> None:
        if not self.target_name:
            raise ValueError("target_name must not be empty")
        if not self.instrument:
            raise ValueError("instrument must not be empty")
        if not self.parameters:
            raise ValueError("parameters must contain at least one parameter name")

