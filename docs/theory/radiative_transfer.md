# Radiative Transfer

ROBERT now has the first RT-facing reference building blocks:

- gas optical-depth assembly from evaluated correlated-k coefficients,
- random-overlap gas optical-depth combination for multi-species correlated-k
  runs,
- layer optical-depth contributors for CIA and Rayleigh scattering extinction,
- a NumPy clear-sky thermal-emission solver with Planck source-function
  integration.

These are not the full mature NEMESIS RT path yet. They are the explicit bridge
between atmosphere, chemistry, opacity, and later cloud, aerosol, and
scattering-source-function implementations.

## Current Scope

`assemble_gas_optical_depth` consumes:

- an evaluated `AtmosphereState`,
- evaluated correlated-k opacity on the same pressure and spectral grids,
- scalar or layer-dependent gravity in m s^-2.

It returns `GasOpticalDepth`, with:

- species optical depth, shaped `(species, layer, wavelength, g)`,
- total gas optical depth, shaped `(layer, wavelength, g)`,
- hydrostatic layer and species column-density diagnostics,
- cumulative optical depth from the top of the atmosphere,
- g-weighted tau and transmission diagnostics,
- a layer transmission-weighting proxy for plotting.

By default, `assemble_gas_optical_depth` sums species on the same g ordinate.
For multi-species correlated-k calculations, pass
`gas_combination="random_overlap"` to combine species distributions by
random-overlap ranking before RT integration.

The hydrostatic column relation is:

```text
N_layer = delta_pressure / (mean_molecular_weight * atomic_mass * gravity)
```

where pressure is converted to Pa, mean molecular weight is in amu, and
composition is interpreted as volume mixing ratio.

`solve_clear_sky_emission` consumes a `GasOpticalDepth` object and returns
`ClearSkyEmissionResult`, with:

- emergent spectral radiance,
- optional blackbody-star eclipse depth,
- Planck layer source functions,
- layer thermal-emission contribution diagnostics,
- total optical depth used by the solver,
- optional disk-averaged emission quadrature.

The solver can also consume additional layer optical depths, such as H2-H2/H2-He
CIA and H2/He Rayleigh scattering extinction. Its metadata records whether
scattering is absent or treated as extinction-only, so later scattering-capable
solvers can be distinguished from this reference path.

## Tau and Weighting Plots

For visual diagnostics before full emission RT, `GasOpticalDepth` provides:

- `g_weighted_layer_tau()` for layer-by-wavelength tau heatmaps,
- `g_weighted_cumulative_tau_from_top()` for tau-equals-one diagnostics,
- `band_transmission_to_space()` for cumulative transmission maps,
- `layer_transmission_weighting()` for a layer escape-weighting proxy.

The weighting proxy is:

```text
exp(-tau_above_layer) * (1 - exp(-tau_layer))
```

This is useful for seeing which pressures are optically active at each
wavelength. It should not be presented as a final thermal-emission contribution
function yet, because true emission contribution plots also need the layer
source function, usually the Planck function evaluated on the P-T profile.

The runnable example:

```bash
python examples/plot_synthetic_tau_weighting.py
```

writes a tau heatmap and a wavelength-averaged weighting profile over a
synthetic P-T profile. The same diagnostic object can later be used with the
HAT-P-32b k-tables once the clear-sky emission backend is wired.

The local HAT-P-32b emission benchmark:

```bash
python examples/benchmark_hat_p_32b_emission_rt.py
```

loads the external HAT-P-32b P-T CSV, R1000 `.kta` files, the local NEMESIS-style
CIA table, and the external emission CSV. It compares ROBERT's current
clear-sky result to the benchmark and writes both a plot and a JSON metric
report. This is a diagnostic benchmark, not a strict pass/fail validation,
because the current ROBERT path still omits FastChem abundance profiles, exact
NEMESIS layering/path parity, clouds/aerosols, and scattering source functions.

## Scattering Boundary

Scattering must enter ROBERT in two separate ways:

- as extinction optical depth, such as Rayleigh, cloud, or aerosol opacity;
- as a source-function treatment when scattering redirects radiation into the
  line of sight.

The current clear-sky solver handles absorption, thermal emission, and optional
extinction-only Rayleigh terms. Clouds/aerosols and multiple-scattering source
terms should be added as independent RT components rather than hidden inside
gas opacity.

## Design Direction

The RT package should keep a readable NumPy reference implementation first.
NEMESIS and NemesisPy show that the mature correlated-k path needs:

- validated gas optical-depth assembly,
- reference-tested random-overlap handling for multiple opacity sources,
- CIA opacity for H2-H2 and H2-He continua with broader HITRAN pair support,
- Planck/source-function emission integration,
- Rayleigh and cloud/aerosol scattering source functions,
- contribution-function diagnostics,
- performance backends only after the reference equations are tested.

Future opacity sampling and line-by-line modes should enter through opacity
providers and produce the same RT-facing concepts: layer optical depth,
transmission, and contribution diagnostics.
