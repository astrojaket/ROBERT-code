"""Conservative unified-memory guards for Apple laptop execution."""

from __future__ import annotations

from dataclasses import dataclass
import os

from robert_exoplanets.core import RobertValidationError


def physical_memory_bytes() -> int:
    """Return installed physical memory without adding a runtime dependency."""

    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, TypeError, ValueError):  # pragma: no cover - platform fallback
        return 8 * 1024**3


@dataclass(frozen=True)
class MetalResourcePolicy:
    """Reject graphs whose conservative peak estimate is unsafe.

    Apple Silicon uses unified memory: exhausting the GPU allocation also
    deprives the operating system and applications.  The default ceiling is
    the smaller of 1 GiB or 8% of installed RAM.  Estimates deliberately
    include sort/compiler scratch factors rather than only visible arrays.
    """

    total_memory_bytes: int
    maximum_fraction: float = 0.08
    absolute_maximum_bytes: int = 1024**3
    maximum_batch_size: int = 1

    @classmethod
    def laptop_safe(cls) -> "MetalResourcePolicy":
        return cls(total_memory_bytes=physical_memory_bytes())

    @property
    def working_set_limit_bytes(self) -> int:
        return min(
            self.absolute_maximum_bytes,
            int(self.total_memory_bytes * self.maximum_fraction),
        )

    def require_safe(self, label: str, estimated_peak_bytes: int) -> None:
        estimate = int(estimated_peak_bytes)
        if estimate < 0:
            raise RobertValidationError("estimated Metal memory must be non-negative")
        if estimate > self.working_set_limit_bytes:
            gib = estimate / 1024**3
            limit_gib = self.working_set_limit_bytes / 1024**3
            raise RobertValidationError(
                f"refusing unsafe Metal {label}: conservative peak {gib:.2f} GiB "
                f"exceeds laptop-safe limit {limit_gib:.2f} GiB; reduce batch, "
                "layers, wavelengths, g ordinates, or Mie capacity"
            )

    def require_safe_batch(self, label: str, batch_size: int) -> None:
        """Refuse proposal batching by default on unified-memory laptops."""

        batch = int(batch_size)
        if batch < 1:
            raise RobertValidationError("Metal batch size must be positive")
        if batch > self.maximum_batch_size:
            raise RobertValidationError(
                f"refusing unsafe Metal {label}: batch {batch} exceeds "
                f"laptop-safe maximum {self.maximum_batch_size}; use isolated "
                "single-proposal processes"
            )

    def estimate_rorr_bytes(
        self,
        *,
        species: int,
        layers: int,
        spectral: int,
        g_ordinates: int,
        batch: int = 1,
    ) -> int:
        """Estimate RORR values, sort indices, and XLA/Metal sort scratch."""

        dimensions = (species, layers, spectral, g_ordinates, batch)
        if any(int(value) < 1 for value in dimensions):
            raise RobertValidationError("Metal RORR dimensions must be positive")
        points = int(batch) * int(layers) * int(spectral)
        input_values = points * int(species) * int(g_ordinates) * 4
        pair_values = points * max(int(species) - 1, 1) * int(g_ordinates) ** 2 * 4
        return input_values * 3 + pair_values * 24

    def estimate_opacity_sampling_bytes(
        self,
        *,
        species: int,
        pressure_grid: int,
        temperature_grid: int,
        layers: int,
        spectral: int,
        disc_points: int,
        batch: int = 1,
    ) -> int:
        """Estimate resident sampled tables plus interpolation/RT scratch.

        This is evaluated before a table is copied to the accelerator.  The
        multiplier accounts for input conversion and XLA temporaries rather
        than pretending that the visible arrays are the peak allocation.
        """

        dimensions = (
            species,
            pressure_grid,
            temperature_grid,
            layers,
            spectral,
            disc_points,
            batch,
        )
        if any(int(value) < 1 for value in dimensions):
            raise RobertValidationError(
                "Metal opacity-sampling dimensions must be positive"
            )
        table = (
            int(species)
            * int(pressure_grid)
            * int(temperature_grid)
            * int(spectral)
            * 4
        )
        interpolated = int(species) * int(layers) * int(spectral) * 4
        radiative_transfer = (
            int(batch) * int(disc_points) * int(layers) * int(spectral) * 4
        )
        return table * 3 + interpolated * 8 + radiative_transfer * 8

    def estimate_sh4_bytes(
        self,
        *,
        layers: int,
        spectral: int,
        g_ordinates: int,
        batch: int = 1,
    ) -> int:
        """Estimate compact LU, sources, residuals, and compiler scratch."""

        dimensions = (layers, spectral, g_ordinates, batch)
        if any(int(value) < 1 for value in dimensions):
            raise RobertValidationError("Metal SH4 dimensions must be positive")
        systems = int(batch) * int(spectral) * int(g_ordinates)
        system_size = 4 * int(layers)
        compact_arrays = systems * system_size * (16 + 11 + 8) * 4
        layer_sources = systems * int(layers) * 32 * 4
        return (compact_arrays + layer_sources) * 8


__all__ = ["MetalResourcePolicy", "physical_memory_bytes"]
