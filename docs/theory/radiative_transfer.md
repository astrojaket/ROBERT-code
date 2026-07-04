# Radiative Transfer

ROBERT now has the first RT-facing reference building block: gas optical-depth
assembly from evaluated correlated-k coefficients. This is not the full
clear-sky emission solver yet. It is the explicit bridge between atmosphere,
chemistry, opacity, and the later Planck/source-function integration.

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

The hydrostatic column relation is:

```text
N_layer = delta_pressure / (mean_molecular_weight * atomic_mass * gravity)
```

where pressure is converted to Pa, mean molecular weight is in amu, and
composition is interpreted as volume mixing ratio.

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

## Design Direction

The RT package should keep a readable NumPy reference implementation first.
NEMESIS and NemesisPy show that the mature correlated-k path needs:

- validated gas optical-depth assembly,
- random-overlap handling for multiple opacity sources,
- CIA opacity for H2-H2 and H2-He continua,
- Planck/source-function emission integration,
- contribution-function diagnostics,
- performance backends only after the reference equations are tested.

Future opacity sampling and line-by-line modes should enter through opacity
providers and produce the same RT-facing concepts: layer optical depth,
transmission, and contribution diagnostics.
