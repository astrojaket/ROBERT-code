# Temperature Profiles

ROBERT treats temperature profiles as modular components that evaluate
temperature on a `PressureGrid`. This keeps emission radiative transfer
independent from the way a profile is obtained.

Current profile types:

- `IsothermalTemperatureProfile`: returns one constant temperature for every
  atmospheric layer.
- `TabulatedTemperatureProfile`: interpolates an input pressure-temperature
  table onto ROBERT layer centers.

`TabulatedTemperatureProfile` interpolates linearly in `log10(pressure)`, which
matches the usual atmospheric convention for P-T profiles sampled across many
orders of magnitude in pressure. By default, it rejects pressure grids that
extend outside the table coverage. A `clip` extrapolation policy exists for
legacy comparison workflows, but it must be requested explicitly.

The HAT-P-32b NemesisPy emission example uses an externally generated P-T CSV.
ROBERT now supports this as an input profile without coupling the future
radiative-transfer backend to the CSV file format.
