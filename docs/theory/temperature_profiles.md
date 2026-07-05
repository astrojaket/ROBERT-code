# Temperature Profiles

ROBERT treats temperature profiles as modular components that evaluate
temperature on a `PressureGrid`. This keeps emission radiative transfer
independent from the way a profile is obtained.

Current profile types:

- `IsothermalTemperatureProfile`: returns one constant temperature for every
  atmospheric layer.
- `TabulatedTemperatureProfile`: interpolates an input pressure-temperature
  table onto ROBERT layer centers.
- `SplineTemperatureProfile`: evaluates a natural-cubic spline in
  `log10(pressure)`. The knot temperatures can be fixed or supplied as
  retrieval parameters.
- `MadhusudhanSeager2009TemperatureProfile`: evaluates the piecewise parametric
  Madhusudhan & Seager (2009) profile used in many emission retrievals.
- `ParmentierGuillot2014TemperatureProfile`: evaluates the dual-visible-channel
  Parmentier & Guillot style irradiated profile used by the local HAT-P-32b
  benchmark configuration.

`TabulatedTemperatureProfile` interpolates linearly in `log10(pressure)`, which
matches the usual atmospheric convention for P-T profiles sampled across many
orders of magnitude in pressure. By default, it rejects pressure grids that
extend outside the table coverage. A `clip` extrapolation policy exists for
legacy comparison workflows, but it must be requested explicitly.

The HAT-P-32b emission benchmark uses an externally generated P-T CSV. ROBERT
now supports this as an input profile without coupling the radiative-transfer
backend to the CSV file format.

## Retrieval Profiles

All retrieval-facing profiles declare their required parameter names through
`required_parameters()`. The atmosphere builder passes a single parameter
mapping to the selected profile, then forwards the evaluated temperature array
to chemistry and radiative-transfer components.

`SplineTemperatureProfile` uses fixed pressure knots and either fixed knot
temperatures or one retrieval parameter per knot. Knot pressures may be given in
any order; ROBERT stores them in increasing pressure order and reorders any
explicit parameter names in the same way.

`MadhusudhanSeager2009TemperatureProfile` expects:

- `P1`, `P2`, `P3`: transition pressures as `log10(pressure_unit)`.
- `T0`: top-boundary temperature in K.
- `alpha1`, `alpha2`: positive profile-shape parameters.

By default, `pressure_unit="bar"` and the profile reference pressure `P0` is the
lowest pressure on the requested pressure grid. A fixed `reference_pressure`
can be supplied when a retrieval wants `P0` to remain independent of the grid.

`ParmentierGuillot2014TemperatureProfile` expects:

- `kappa_IR`: infrared opacity in m2 kg-1.
- `gamma1`, `gamma2`: visible-to-infrared opacity ratios for two visible
  channels.
- `T_irr`: irradiation temperature in K.
- `alpha`: fractional weight of the second visible channel, between 0 and 1.

The PG14-style profile also needs gravity in m s-2. Gravity can be fixed when
the profile is constructed or supplied as a retrieval parameter named
`gravity`. The internal temperature can likewise be fixed at construction time
or retrieved through `T_int`.
