"""Run a stubbed JWST exoplanet emission retrieval.

This script demonstrates the intended ROBERT workflow without implementing
physical atmospheric retrieval yet.
"""

from __future__ import annotations

import numpy as np

from robert_exoplanets import Observation, RetrievalConfig, run_stub_retrieval


def build_demo_observation() -> Observation:
    """Create a tiny deterministic spectrum for workflow testing."""

    wavelength = np.linspace(5.0, 12.0, 8)
    flux = np.array([980, 995, 1004, 1010, 1008, 1002, 994, 985], dtype=float) * 1e-6
    uncertainty = np.full_like(flux, 35e-6)
    return Observation.from_arrays(wavelength, flux, uncertainty)


def main() -> None:
    observation = build_demo_observation()
    config = RetrievalConfig(
        target_name="WASP-Stub-b",
        instrument="JWST/MIRI LRS",
    )

    result = run_stub_retrieval(observation, config)

    print(f"Target: {result.config.target_name}")
    print(f"Instrument: {result.config.instrument}")
    print(f"Model: {result.model_name}")
    print(f"Converged: {result.converged}")
    print(f"Message: {result.message}")
    print(f"Best-fit baseline: {result.best_fit_parameters['baseline']:.6e}")
    print(f"Stub log-likelihood: {result.log_likelihood:.3f}")


if __name__ == "__main__":
    main()
