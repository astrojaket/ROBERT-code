"""Reproducible flat-spectrum ensembles and CLR posterior diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.instruments import Observation, ObservationCollection
from robert_exoplanets.postprocessing import weighted_quantile


MOLECULAR_MASS_AMU = {
    "H2O": 18.01528,
    "CO2": 44.0095,
    "CO": 28.0101,
    "H2S": 34.0809,
    "SO2": 64.066,
}


@dataclass(frozen=True)
class ConstantResolvingPowerGrid:
    """Logarithmic wavelength bins spanning two exact requested edges."""

    edges_micron: NDArray[np.float64]
    centers_micron: NDArray[np.float64]
    target_resolving_power: float
    effective_resolving_power: float

    def __post_init__(self) -> None:
        edges = np.array(self.edges_micron, dtype=float, copy=True)
        centers = np.array(self.centers_micron, dtype=float, copy=True)
        if edges.ndim != 1 or centers.shape != (edges.size - 1,):
            raise RobertValidationError("grid centers must correspond to bin edges")
        if np.any(np.diff(edges) <= 0.0) or np.any(edges <= 0.0):
            raise RobertValidationError("wavelength edges must be positive and increasing")
        expected = np.sqrt(edges[:-1] * edges[1:])
        if not np.allclose(centers, expected, rtol=1.0e-13, atol=0.0):
            raise RobertValidationError("grid centers must be geometric bin centers")
        edges.setflags(write=False)
        centers.setflags(write=False)
        object.__setattr__(self, "edges_micron", edges)
        object.__setattr__(self, "centers_micron", centers)

    @property
    def n_bins(self) -> int:
        return int(self.centers_micron.size)


def constant_resolving_power_grid(
    wavelength_min_micron: float = 3.0,
    wavelength_max_micron: float = 5.0,
    resolving_power: float = 100.0,
) -> ConstantResolvingPowerGrid:
    """Construct equal-log-width bins with exact outer edges.

    The bin count is ``round(R * ln(lambda_max/lambda_min))``. Edges are
    geometrically spaced and centers are geometric means. This makes every bin
    have the same exact ``lambda_center / delta_lambda``; it is generally very
    slightly different from the requested nominal resolving power because the
    two outer edges are held exact.
    """

    lower = float(wavelength_min_micron)
    upper = float(wavelength_max_micron)
    target = float(resolving_power)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 or upper <= lower:
        raise RobertValidationError("wavelength limits must be finite, positive, and ordered")
    if not np.isfinite(target) or target <= 0.0:
        raise RobertValidationError("resolving power must be finite and positive")
    n_bins = max(1, int(np.rint(target * np.log(upper / lower))))
    edges = np.geomspace(lower, upper, n_bins + 1, dtype=float)
    centers = np.sqrt(edges[:-1] * edges[1:])
    effective = float(centers[0] / (edges[1] - edges[0]))
    return ConstantResolvingPowerGrid(edges, centers, target, effective)


def generate_flat_spectrum_ensemble(
    source: ObservationCollection,
    *,
    n_realizations: int = 100,
    seed: int = 20260719,
    grid: ConstantResolvingPowerGrid | None = None,
) -> tuple[tuple[Observation, ...], dict[str, object]]:
    """Generate independent Gaussian realizations around source medians."""

    if n_realizations < 1:
        raise RobertValidationError("n_realizations must be positive")
    if seed < 0:
        raise RobertValidationError("seed must be non-negative")
    selected_grid = constant_resolving_power_grid() if grid is None else grid
    depths = np.concatenate(
        [np.asarray(item.observation.flux, dtype=float) for item in source.datasets]
    )
    errors = np.concatenate(
        [np.asarray(item.observation.uncertainty, dtype=float) for item in source.datasets]
    )
    median_depth = float(np.median(depths))
    median_error = float(np.median(errors))
    root_seed = np.random.SeedSequence(seed)
    child_seeds = root_seed.spawn(n_realizations)
    observations = []
    for realization_id, child_seed in enumerate(child_seeds):
        noise = np.random.default_rng(child_seed).normal(
            0.0, median_error, selected_grid.n_bins
        )
        observations.append(
            Observation(
                wavelength=selected_grid.centers_micron,
                wavelength_bin_edges=selected_grid.edges_micron,
                flux=np.full(selected_grid.n_bins, median_depth) + noise,
                uncertainty=np.full(selected_grid.n_bins, median_error),
                wavelength_unit="micron",
                flux_unit="transit_depth",
                observable="transit_depth",
                instrument="synthetic-R100-3-5-micron",
                metadata={
                    "realization_id": f"{realization_id:03d}",
                    "rng_algorithm": "numpy.PCG64",
                    "root_seed": str(seed),
                    "seed_spawn_key": ",".join(map(str, child_seed.spawn_key)),
                    "source": source.name,
                    "flat_depth_dimensionless": repr(median_depth),
                    "uncertainty_dimensionless": repr(median_error),
                },
            )
        )
    metadata: dict[str, object] = {
        "schema_version": 1,
        "source_spectrum": source.name,
        "source_metadata": dict(source.metadata),
        "source_point_count": int(depths.size),
        "median_transit_depth_dimensionless": median_depth,
        "median_transit_depth_ppm": median_depth * 1.0e6,
        "median_uncertainty_dimensionless": median_error,
        "median_uncertainty_ppm": median_error * 1.0e6,
        "n_realizations": n_realizations,
        "root_seed": seed,
        "rng_algorithm": "numpy.PCG64",
        "child_seed_spawn_keys": [list(item.spawn_key) for item in child_seeds],
        "wavelength_unit": "micron",
        "flux_unit": "dimensionless_transit_depth",
        "grid": {
            "edge_convention": (
                f"{selected_grid.n_bins} equal-log-width bins with exact "
                f"{selected_grid.edges_micron[0]:g} and "
                f"{selected_grid.edges_micron[-1]:g} micron outer edges"
            ),
            "center_convention": "geometric mean of adjacent bin edges",
            "target_resolving_power": selected_grid.target_resolving_power,
            "effective_center_over_width_resolving_power": selected_grid.effective_resolving_power,
            "n_bins": selected_grid.n_bins,
            "wavelength_min_micron": float(selected_grid.edges_micron[0]),
            "wavelength_max_micron": float(selected_grid.edges_micron[-1]),
            "edges_micron": selected_grid.edges_micron.tolist(),
            "centers_micron": selected_grid.centers_micron.tolist(),
        },
    }
    return tuple(observations), metadata


def closed_composition(
    free_log10_vmr: Mapping[str, NDArray[np.float64]],
    *,
    closure_species: str,
) -> dict[str, NDArray[np.float64]]:
    """Recover the omitted physical CLR category from unit-sum closure."""

    if not free_log10_vmr:
        raise RobertValidationError("at least one free CLR category is required")
    vmr = {name: np.power(10.0, np.asarray(values, dtype=float)) for name, values in free_log10_vmr.items()}
    shapes = {values.shape for values in vmr.values()}
    if len(shapes) != 1:
        raise RobertValidationError("CLR posterior arrays must have matching shapes")
    remainder = 1.0 - sum(vmr.values())
    if np.any(remainder <= 0.0) or np.any(remainder > 1.0):
        raise RobertValidationError("CLR posterior samples violate unit-sum closure")
    vmr[closure_species] = remainder
    return vmr


def composition_mean_molecular_weight(
    vmr: Mapping[str, NDArray[np.float64]],
    molecular_masses_amu: Mapping[str, float] = MOLECULAR_MASS_AMU,
) -> NDArray[np.float64]:
    """Calculate composition-weighted mean molecular weight in amu."""

    missing = sorted(set(vmr) - set(molecular_masses_amu))
    if missing:
        raise RobertValidationError("missing molecular masses: " + ", ".join(missing))
    return np.asarray(
        sum(np.asarray(values) * molecular_masses_amu[name] for name, values in vmr.items()),
        dtype=float,
    )


def abundance_constraint(
    vmr: Sequence[float] | NDArray[np.float64],
    weights: Sequence[float] | NDArray[np.float64],
    *,
    threshold: float = 0.01,
    credibility: float = 0.95,
) -> dict[str, float | bool]:
    """Assess a preregistered one-sided abundance lower-bound criterion.

    A gas is called constrained when its one-sided credible lower bound exceeds
    ``threshold``. The study primary definition uses a 95% lower bound and a
    1% VMR threshold; 0.1%, 10%, and 50% thresholds are reported as sensitivity
    checks.
    """

    values = np.asarray(vmr, dtype=float)
    sample_weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or sample_weights.shape != values.shape:
        raise RobertValidationError("abundance values and weights must be matching vectors")
    if np.any(values <= 0.0) or np.any(values > 1.0):
        raise RobertValidationError("abundances must lie in (0, 1]")
    if threshold <= 0.0 or threshold >= 1.0 or credibility <= 0.0 or credibility >= 1.0:
        raise RobertValidationError("threshold and credibility must lie in (0, 1)")
    total = float(np.sum(sample_weights))
    if not np.isfinite(total) or total <= 0.0 or np.any(sample_weights < 0.0):
        raise RobertValidationError("weights must be finite, non-negative, and nonzero")
    normalized = sample_weights / total
    lower = float(weighted_quantile(values, normalized, (1.0 - credibility,))[0])
    probability = float(np.sum(normalized[values > threshold]))
    return {
        "threshold_vmr": float(threshold),
        "credibility": float(credibility),
        "lower_bound_vmr": lower,
        "posterior_probability_above_threshold": probability,
        "constrained": bool(lower > threshold),
    }


__all__ = [
    "MOLECULAR_MASS_AMU",
    "ConstantResolvingPowerGrid",
    "abundance_constraint",
    "closed_composition",
    "composition_mean_molecular_weight",
    "constant_resolving_power_grid",
    "generate_flat_spectrum_ensemble",
]
