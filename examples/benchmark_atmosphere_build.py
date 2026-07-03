"""Benchmark the current ROBERT atmosphere-build path."""

from __future__ import annotations

import os

from robert_exoplanets import (
    AtmosphereBuilder,
    BackgroundGasMixture,
    CompositionMeanMolecularWeight,
    FreeChemistry,
    ParmentierGuillot2014TemperatureProfile,
    PressureGrid,
    time_callable,
)


def main() -> None:
    """Run a small atmosphere-build benchmark."""

    repeat = int(os.environ.get("ROBERT_BENCH_REPEAT", "200"))
    warmup = int(os.environ.get("ROBERT_BENCH_WARMUP", "20"))
    pressure_grid = PressureGrid.logspace(
        min_pressure=1.0e-6,
        max_pressure=100.0,
        n_layers=100,
        unit="bar",
        name="benchmark atmosphere grid",
    )
    builder = AtmosphereBuilder(
        pressure_grid=pressure_grid,
        temperature_profile=ParmentierGuillot2014TemperatureProfile(
            gravity=4.3,
            internal_temperature=200.0,
        ),
        chemistry_model=FreeChemistry(
            active_species=("H2O", "CO", "CO2"),
            fixed_mixing_ratios={"CO2": 1.0e-5},
            parameter_names={"H2O": "log_H2O", "CO": "log_CO"},
            parameter_mode="log10",
            background=BackgroundGasMixture.hydrogen_helium(),
        ),
        mean_molecular_weight_model=CompositionMeanMolecularWeight(),
    )
    parameters = {
        "kappa_IR": 0.02,
        "gamma1": 0.5,
        "gamma2": 1.5,
        "T_irr": 1500.0,
        "alpha": 0.5,
        "log_H2O": -3.0,
        "log_CO": -4.0,
    }

    result = time_callable(
        lambda: builder.build(parameters),
        name="atmosphere-build",
        repeat=repeat,
        warmup=warmup,
        metadata={"n_layers": str(pressure_grid.n_layers)},
    )

    print("Benchmark: atmosphere-build")
    print(f"Layers: {pressure_grid.n_layers}")
    print(f"Repeats: {result.repeats}")
    print(f"Warmups: {result.warmups}")
    print(f"Best: {result.min_s * 1.0e3:.3f} ms")
    print(f"Median: {result.median_s * 1.0e3:.3f} ms")
    print(f"Mean: {result.mean_s * 1.0e3:.3f} ms")
    print(f"Calls per second: {result.calls_per_second:.1f}")


if __name__ == "__main__":
    main()
