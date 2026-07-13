"""Build a ROBERT atmosphere from a HAT-P-32b-style model config."""

from __future__ import annotations

import numpy as np

from robert_exoplanets import Planet, build_atmosphere_setup


def example_config() -> dict:
    """Return a compact HAT-P-32b-style atmosphere config."""

    return {
        "pressure_grid": {
            "n_layers": 100,
            "p_top_bar": 1.0e-6,
            "p_bot_bar": 100.0,
        },
        "temperature_profile": {
            "type": "guillot14",
            "values": {
                "kappa_IR": 0.02,
                "gamma1": 0.5,
                "gamma2": 1.5,
                "T_irr": 1500.0,
                "alpha": 0.5,
                "T_int": 200.0,
            },
        },
        "molecules": {
            "free": {
                "names": ["H2O", "CO", "CO2"],
                "inactive": {"names": ["H2", "He"]},
                "log": True,
                "values": {
                    "H2O": 1.0e-3,
                    "CO": 1.0e-4,
                    "CO2": 1.0e-5,
                },
            },
        },
    }


def main() -> None:
    """Build and summarize the atmosphere."""

    planet = Planet(name="HAT-P-32b", radius_m=1.4e8, gravity_m_s2=4.3)
    setup = build_atmosphere_setup(example_config(), planet=planet)
    atmosphere = setup.build_atmosphere_builder().build(setup.default_parameters)

    print("Run: config-driven-atmosphere")
    print(f"Layers: {atmosphere.n_layers}")
    print(f"Species: {', '.join(atmosphere.species)}")
    print(f"Default parameters: {', '.join(sorted(setup.default_parameters))}")
    print(f"Temperature range: {np.min(atmosphere.temperature):.1f}-{np.max(atmosphere.temperature):.1f} K")
    print(f"Mean molecular weight: {float(np.median(atmosphere.mean_molecular_weight)):.3f} amu")


if __name__ == "__main__":
    main()
