# Radiative Transfer

ROBERT now has the first RT-facing reference building blocks:

- gas optical-depth assembly from evaluated correlated-k coefficients,
- random-overlap gas optical-depth combination for multi-species correlated-k
  runs,
- layer optical-depth contributors for CIA and Rayleigh scattering extinction,
- a NumPy clear-sky thermal-emission solver with Planck source-function
  integration,
- explicit disc geometry objects for normal-emission, uniform thermal-disc, and
  Lobatto phase quadrature calculations,
- a first-order direct-beam single-scattering source treatment for Rayleigh-like
  or isotropic scattering phase functions.

These are not a full mature RT path yet. They are the explicit bridge between
atmosphere, chemistry, opacity, and later cloud, aerosol, and
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
- optional point-spectrum diagnostics and disc-averaged geometry.

The solver can also consume additional layer optical depths, such as H2-H2/H2-He
CIA and H2/He Rayleigh scattering extinction. Its metadata records whether
scattering is absent or treated as extinction-only, so later scattering-capable
solvers can be distinguished from this reference path.

## Geometry and Source Function

`DiscGeometry` is the RT-facing angular container. It stores a sequence of
`DiscPoint` samples with emission zenith cosine, normalized disc weight, and
optional projected-disc, latitude/longitude, stellar zenith, and stellar azimuth
metadata. The current clear-sky solver uses the emission cosine and weight to
calculate point spectra and the disc average. The stellar-angle metadata is
reserved for reflected-light and scattering-source-function kernels.

The available helpers are:

- `normal_emission_geometry()` for a single `mu=1` point,
- `gauss_legendre_disk_geometry(n_mu)` for the current uniform thermal-disc
  integral,
- `lobatto_phase_geometry(phase_angle_deg, n_mu)` for projected-disc phase
  quadrature.

The default source function is thermal Planck emission only. If a
`SingleScatteringSource` is supplied, the solver adds a first-order direct-beam
scattering source term for optical-depth contributors whose `kind` contains
`scattering`. This uses the `DiscGeometry` stellar zenith and azimuth metadata,
so phase-aware geometries are required. The returned result stores separate
scattering source-function and contribution diagnostics.

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

loads the external HAT-P-32b P-T CSV, R1000 `.kta` files, an optional local CIA
table, and the external emission CSV. It compares ROBERT's current clear-sky
result to the benchmark and writes both a plot and a JSON metric report. This is a
diagnostic benchmark, not a strict pass/fail validation, because the current
ROBERT path still omits exact benchmark layering/path parity, clouds/aerosols,
and multiple-scattering source functions.

Set `HAT_P_32B_CIA_FILE` to include a local CIA binary table in this benchmark.
If that variable is not set, the script leaves the CIA term off unless
`ROBERT_HAT_P_32B_INCLUDE_CIA=1` is requested explicitly.

To exercise the first single-scattering source term in that benchmark, run:

```bash
ROBERT_HAT_P_32B_INCLUDE_SCATTERING_SOURCE=1 python examples/benchmark_hat_p_32b_emission_rt.py
```

The default benchmark leaves this off to preserve the historical thermal
emission comparison.

The standalone synthetic scattering example:

```bash
python examples/plot_single_scattering_reference.py
```

writes a phase-dependent reflected-light sanity plot and a dayside scattering
contribution profile.

## Scattering Boundary

Scattering must enter ROBERT in two separate ways:

- as extinction optical depth, such as Rayleigh, cloud, or aerosol opacity;
- as a source-function treatment when scattering redirects radiation into the
  line of sight.

The current clear-sky solver handles absorption, thermal emission, optional
extinction-only Rayleigh terms, and optional first-order single scattering of a
direct stellar beam. Clouds/aerosols, surfaces, and multiple-scattering source
terms should be added as independent RT components rather than hidden inside
gas opacity.

## Design Direction

The RT package should keep a readable NumPy reference implementation first.
The mature correlated-k path needs:

- validated gas optical-depth assembly,
- reference-tested random-overlap handling for multiple opacity sources,
- CIA opacity for H2-H2 and H2-He continua with broader HITRAN pair support,
- Planck/source-function emission integration,
- tested single-scattering diagnostics for Rayleigh and cloud/aerosol optical
  properties,
- multiple-scattering source functions,
- contribution-function diagnostics,
- performance backends only after the reference equations are tested.

Future opacity sampling and line-by-line modes should enter through opacity
providers and produce the same RT-facing concepts: layer optical depth,
transmission, and contribution diagnostics.
