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
- hydrostatic radius/path geometry anchored at a configurable reference radius
  and pressure,
- a first-order direct-beam single-scattering source treatment for Rayleigh-like
  or isotropic scattering phase functions,
- cloud/aerosol optical-property containers carrying extinction optical depth,
  single-scattering albedo, and asymmetry factor,
- a conservative two-stream multiple-scattering reference backend behind the
  thermal-emission solver interface,
- a Numba-backed thermal integration backend for thermal-only RT benchmarks.

These are not a full mature RT path yet. They are the explicit bridge between
atmosphere, chemistry, opacity, and later cloud, aerosol, and
scattering-source-function implementations. The current two-stream closure is a
benchmark hook, not the final cloud-scattering science solver.

## Reusable Retrieval Forward Model

`ClearSkyEmissionForwardModel` is the package-level orchestration layer for the
validated clear-sky retrieval path. It consumes typed `Planet`, `Star`,
`PressureGrid`, `SpectralGrid`, `CorrelatedKOpacityProvider`, and
`ClearSkyEmissionModelConfig` objects. It owns:

- validation and caching of prepared opacity outside likelihood calls,
- constant-with-altitude log10 VMR parameter mapping for arbitrary trace gases,
- explicit H2/He background fill and mean molecular weight,
- optional uniform temperature offset and radius scale parameters,
- gas optical-depth assembly, Rayleigh selection, geometry, and emission RT,
- required-parameter declaration, opacity identifiers, and manifest metadata.

Direct construction remains available for tests and specialist workflows, but
normal user code should use the typed Python factory configuration. This keeps
the entire target definition in a familiar importable `.py` file while the
factory performs one-time opacity loading, pressure-grid construction,
temperature evaluation, `exo_k` binning, and model preparation:

```python
from pathlib import Path

from robert_exoplanets import (
    ClearSkyEmissionFactoryConfig,
    ClearSkyEmissionModelConfig,
    ExoKOpacitySource,
    ExoKTableBinning,
    TabulatedTemperatureProfile,
    build_clear_sky_emission_model,
)

config = ClearSkyEmissionFactoryConfig(
    planet=planet,
    star=star,
    temperature_profile=TabulatedTemperatureProfile.from_csv("target_pt.csv"),
    opacity_source=ExoKOpacitySource(
        species=("H2O", "CO2", "NH3"),
        directory=Path("opacity"),
        filename_pattern="*_R1000.kta",
    ),
    opacity_binning=ExoKTableBinning(num=300),
    model=ClearSkyEmissionModelConfig(
        opacity_species=("H2O", "CO2", "NH3"),
        log_vmr_parameters={
            "H2O": "log_h2o",
            "CO2": "log_co2",
            "NH3": "log_nh3",
        },
        gas_combination="random_overlap",
    ),
)

model = build_clear_sky_emission_model(
    config,
    spectral_grid=observation.spectral_grid,
)
```

`ExoKOpacitySource(paths={...})` accepts explicitly selected KTA or HDF5 tables
supported by `exo_k`; directory discovery is the convenient KTA path. Set an
explicit `pressure_grid` when it should differ from the opacity table centers,
and set `opacity_binning=None` only for a provider already prepared on the
observation grid.

The factory is intentionally Python-first. YAML or TOML can be added later as
an input adapter without changing this validated typed boundary. The model does
not discover files or resample opacity during evaluation. Manifest metadata
includes the Python configuration interface, opacity source and binning
choices, temperature parameterization, physical constants, parameter mappings,
prepared opacity identity, and hashes of the pressure, spectral, and base
temperature grids.

The validated complete target example is
`examples/hat_p_32b_config.py`; copy it as the starting point for another
planet rather than copying the retrieval implementation.

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

`hydrostatic_path_geometry` can additionally compute layer-edge radii,
layer-center radii, and spherical shell path-length factors from the same
atmosphere, gravity, reference radius, and reference pressure. The current
HAT-P-32b benchmark still matches best with the default plane-parallel secant
path, but the spherical path object is now available for controlled geometry
tests and future cloud, surface, transmission, or phase-curve work.

`solve_clear_sky_emission` consumes a `GasOpticalDepth` object and returns
`ClearSkyEmissionResult`, with:

- emergent spectral radiance,
- optional blackbody-star eclipse depth,
- Planck layer source functions,
- layer thermal-emission contribution diagnostics,
- total optical depth used by the solver,
- optional point-spectrum diagnostics and disc-averaged geometry.

By default, `solve_clear_sky_emission` uses a plane-parallel secant path through
each layer. Passing `path_geometry=hydrostatic_path_geometry(...)` switches to
spherical shell path-length factors and records the reference pressure/radius,
top radius, and bottom radius in solver metadata.

The solver can also consume additional layer optical depths, such as H2-H2/H2-He
CIA and H2/He Rayleigh scattering extinction. Its metadata records whether
scattering is absent or treated as extinction-only, so later scattering-capable
solvers can be distinguished from this reference path.

`load_nemesispy_cia_table()` loads the vendored NemesisPy v1.0.1
`exocia_hitran12_200-3800K.tab` reference asset. ROBERT records its upstream
commit, SHA-256 checksum, BSD-3-Clause license, hydrogen spin-state choice, and
interpolation policies in retrieval provenance. Parameterized emission models
can attach this table once at construction and evaluate CIA for each atmosphere.

Cloud/aerosol opacity can enter in two equivalent ways:

- pass a `CloudOpticalProperties` object containing layer-by-wavelength
  extinction optical depth, single-scattering albedo, and asymmetry factor;
- pass split `LayerOpticalDepth` objects from
  `CloudOpticalProperties.as_layer_optical_depths()` for explicit absorption
  and scattering diagnostics.

Passing `multiple_scattering_backend="two_stream"` activates the first
conservative multiple-scattering reference closure. The returned
`total_optical_depth` is then the effective optical depth used by the solver,
while `extinction_optical_depth` remains the physical extinction optical depth
for tau plots and code-to-code comparisons.

Passing `thermal_integration_backend="auto"` uses the Numba thermal-integration
kernel when available and falls back to the NumPy reference path otherwise.
Use `thermal_integration_backend="numpy"` to force the readable reference
backend. Direct-beam single-scattering source runs currently remain on the
NumPy path.

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

The cloud-scattering benchmark bridge:

```bash
python examples/benchmark_cloud_scattering_picaso_virga.py
```

loads `ROBERT_CLOUD_PROPERTY_FILE` when set, accepting either dense `.npz`
arrays, long-table `.csv` files with PICASO/Virga-style aliases such as
`tau_ext`, `omega0`, and `g`, or PICASO `.cld` cloud tables. Index-style
PICASO `.cld` files need physical coordinates from a paired pressure table and
wave grid; the example auto-discovers the bundled PICASO base-case companions
when the file lives in a normal PICASO checkout, or they can be set explicitly
with `ROBERT_PICASO_PRESSURE_FILE` and `ROBERT_PICASO_WAVE_GRID_FILE`.

If no file is provided, the script generates a synthetic cloud property
product, runs extinction-only and two-stream RT, writes plots and a JSON report,
and times the cloud loading plus NumPy and Numba thermal RT paths. With the
public PICASO `jupiterf3.cld` base-case table (60 layers, 196 wavelengths) on
this laptop, cloud loading is about 21 ms and the Numba two-stream smoke path is
about 2.7 ms.

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

The first cloud/aerosol object follows the same separation: it stores optical
properties, but the RT backend decides whether scattering is extinction-only,
single-scattering direct beam, two-stream reference, or a future fuller
multiple-scattering solver.

## Cloud-Scattering Benchmark Ladder

ROBERT should benchmark scattering clouds in staged layers:

1. Analytic limits: no cloud, pure absorption, pure scattering, optically thin
   cloud, optically thick grey cloud, `omega0=0/1`, and isotropic versus
   forward-scattering limits.
2. Optical-property parity: compare ROBERT inputs against PICASO/Virga-style
   layer `tau_ext`, single-scattering albedo, asymmetry factor, and later phase
   moments before running RT.
3. Code-to-code RT: run identical one-dimensional grey and haze cases through
   ROBERT and PICASO, then move to Virga-generated condensate clouds.
4. Published science cases: reproduce selected PICASO/Virga examples for
   reflected light, cloudy thermal phase curves, and condensation-cloud
   benchmark atmospheres.

Useful reference targets include PICASO reflected-light benchmarking
([Batalha et al. 2019](https://arxiv.org/abs/1904.09355)), cloudy thermal phase
curves ([Robbins-Blanch et al. 2022](https://arxiv.org/abs/2204.03545)),
PICASO/Virga reflected-light phase curves
([Hamill et al. 2024](https://arxiv.org/abs/2411.14225)), Virga condensation
cloud benchmarks ([Batalha et al. 2025](https://arxiv.org/abs/2508.15102)),
and PICASO 4.0 cloud/climate updates
([Mang et al. 2026](https://arxiv.org/abs/2602.22468)).

## Design Direction

The RT package should keep a readable NumPy reference implementation first.
The mature correlated-k path needs:

- validated gas optical-depth assembly,
- reference-tested random-overlap handling for multiple opacity sources,
- CIA opacity for H2-H2 and H2-He continua with broader HITRAN pair support,
- Planck/source-function emission integration,
- tested single-scattering diagnostics for Rayleigh and cloud/aerosol optical
  properties,
- benchmarked multiple-scattering source functions,
- contribution-function diagnostics,
- performance backends only after the reference equations are tested.

Future opacity sampling and line-by-line modes should enter through opacity
providers and produce the same RT-facing concepts: layer optical depth,
transmission, and contribution diagnostics.
