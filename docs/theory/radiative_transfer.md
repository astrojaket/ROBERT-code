# Radiative Transfer

ROBERT now has the first RT-facing reference building blocks:

- gas optical-depth assembly from evaluated correlated-k coefficients,
- random-overlap gas optical-depth combination for multi-species correlated-k
  runs,
- layer optical-depth contributors for CIA and Rayleigh scattering extinction,
- a NumPy cloud-free thermal-emission solver with Planck source-function
  integration,
- explicit disc geometry objects for normal-emission, uniform thermal-disc, and
  Lobatto phase quadrature calculations,
- hydrostatic radius/path geometry anchored at a configurable reference radius
  and pressure,
- a first-order direct-beam single-scattering source treatment for Rayleigh-like
  or isotropic scattering phase functions,
- cloud/aerosol optical-property containers carrying extinction optical depth,
  single-scattering albedo, asymmetry factor, and optional phase moments,
- a coupled hemispheric-mean thermal two-stream source-function backend behind
  the thermal-emission solver interface,
- a four-term spherical-harmonics (P3/SH4) thermal multiple-scattering backend
  with explicit physical or HG phase moments and optional delta-M scaling,
- a Numba-backed thermal integration backend for thermal-only RT benchmarks.

These are not a full mature RT path yet. They are the explicit bridge between
atmosphere, chemistry, opacity, and later cloud, aerosol, and
scattering-source-function implementations. The two-stream backend is the fast
approximation. SH4 is the higher-fidelity retrieval backend, but a high-stream
reference remains necessary to establish its science domain. See [Choosing a
Radiative-Transfer Backend](rt_backend_selection.md) for emission,
transmission, reflected-light, and cloud-complexity recommendations.

## Reusable Retrieval Forward Model

`EmissionForwardModel` is the package-level orchestration layer for the
validated cloud-free retrieval path. It consumes typed `Planet`, `Star`,
`PressureGrid`, `SpectralGrid`, `CorrelatedKOpacityProvider`, and
`EmissionModelConfig` objects. It owns:

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
    EmissionFactoryConfig,
    EmissionModelConfig,
    ExoKOpacitySource,
    ExoKTableBinning,
    TabulatedTemperatureProfile,
    build_emission_model,
)

config = EmissionFactoryConfig(
    planet=planet,
    star=star,
    temperature_profile=TabulatedTemperatureProfile.from_csv("target_pt.csv"),
    opacity_source=ExoKOpacitySource(
        species=("H2O", "CO2", "NH3"),
        directory=Path("opacity"),
        filename_pattern="*_R1000.kta",
    ),
    opacity_binning=ExoKTableBinning(num=300),
    model=EmissionModelConfig(
        opacity_species=("H2O", "CO2", "NH3"),
        log_vmr_parameters={
            "H2O": "log_h2o",
            "CO2": "log_co2",
            "NH3": "log_nh3",
        },
        gas_combination="random_overlap",
    ),
)

model = build_emission_model(
    config,
    spectral_grid=observation.spectral_grid,
)
```

`ExoKOpacitySource(paths={...})` accepts explicitly selected KTA or HDF5 tables
supported by `exo_k`; directory discovery is the convenient KTA path. Set an
explicit `pressure_grid` when it should differ from the opacity table centers,
and set `opacity_binning=None` only for a provider already prepared on the
observation grid.

The factory is the typed internal boundary used by the maintained YAML task
adapter. The model does not discover files or resample opacity during
evaluation. Manifest metadata
includes the Python configuration interface, opacity source and binning
choices, temperature parameterization, physical constants, parameter mappings,
prepared opacity identity, and hashes of the pressure, spectral, and base
temperature grids.

For a complete target configuration, copy
`configurations/wasp69b_cloud_free_R1000.yaml` and change the science inputs
rather than copying or editing the retrieval implementation.

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

`solve_emission` consumes a `GasOpticalDepth` object and returns
`EmissionResult`, with:

- emergent spectral radiance,
- optional blackbody-star eclipse depth,
- Planck layer source functions,
- layer thermal-emission contribution diagnostics,
- total optical depth used by the solver,
- optional point-spectrum diagnostics and disc-averaged geometry.

By default, `solve_emission` uses a plane-parallel secant path through
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
  extinction optical depth, single-scattering albedo, asymmetry factor, and
  optionally phase moments through degree four;
- pass split `LayerOpticalDepth` objects from
  `CloudOpticalProperties.as_layer_optical_depths()` for explicit absorption
  and scattering diagnostics.

Passing `multiple_scattering_backend="two_stream"` activates the coupled
hemispheric-mean Toon thermal source-function solver. It solves upward and
downward diffuse flux moments as a banded boundary-value problem, reconstructs
outgoing intensities from the scattering source function, and retains physical
extinction optical depth in both `total_optical_depth` and
`extinction_optical_depth`. No effective-extinction substitution is used.

Passing `multiple_scattering_backend="sh4"` activates the Rooney, Batalha &
Marley P3 solution. Four intensity moments are coupled across every layer with
two incoming half-range boundary constraints at the top and bottom. ROBERT uses
stable layer-anchored eigenmodes, a banded multilayer solve, linear Planck
sources, and source-function angular reconstruction. The high-level backend
scattering-optical-depth weights supplied cloud and Rayleigh phase moments.
Homogeneous-sphere Mie clouds supply exact scalar moments through degree four;
degree four defines the delta-M forward fraction and degrees zero through three
enter SH4. A Henyey–Greenstein closure remains the explicit fallback for cloud
properties that provide only `g`.

In the controlled 160-layer PICASO comparison, matched ROBERT and PICASO SH4
point radiances agree within `3.23e-6` relative error for both isotropic and
`omega=0.9, g=0.6` forward-scattering cases. On a 64-layer, 900-wavelength,
four-g-ordinate, six-angle benchmark, median wall times were 0.304 s for Toon
and 0.798 s for SH4 on the validation laptop: SH4 was 2.62 times slower. The
repeatable benchmark is `examples/benchmark_sh4_rt.py`.

The next controlled comparison uses real molecular correlated-k structure from
the bundled HAT-P-32b ExoMolOP/exo_k archives. ROBERT assembles H2O, CO, CO2,
CH4, NH3, and HCN by random overlap and passes the identical 20-g optical-depth
cube to PICASO. Across 117 spectral bins, the maximum disk-integrated residual
is 0.0507% and the RMS disk residual is 0.0151%. The realistic FastChem+CIA
HAT-P-32b case agrees to 0.00361% maximum disk difference. See
`examples/compare_molecular_emission_picaso.py` and
`docs/review/15_molecular_emission_picaso_comparison.md`.

When an atmosphere supplies temperatures at pressure edges, clear thermal
emission uses the exact formal integral for a Planck source linear in optical
depth within each layer. Both NumPy and Numba integrations implement the same
stable small-optical-depth limit. Atmospheres without edge temperatures retain
the explicitly recorded constant layer-centre source fallback.

`solve_absorption_transmission` provides the first transmission path. It treats
each hydrostatic layer as a constant-property spherical shell, computes exact
full-chord lengths, integrates correlated-k transmission over g, and then
integrates occulting area over impact parameter. The result includes effective
radius and annulus-area diagnostics. Its present scope is direct-beam
absorption/extinction: scattered light returned to the beam, refraction,
finite-star effects, and limb darkening remain future validation work. The
the stable petitRADTRANS benchmark is documented in
`docs/review/19_petitradtrans3_multispecies_transmission.md`, with the later
independent PICASO molecular/cloud comparison in
`docs/review/36_official_picaso_molecular_cloud_parity.md`.

`ParameterizedTransmissionForwardModel` promotes this kernel into the typed
forward-model layer. Like `ParameterizedEmissionForwardModel`, it prepares
opacity once and evaluates temperature, chemistry, opacity, and RT for every
parameter vector. The name uses *transmission* because it predicts a
transmission spectrum/transit depth; *transit model* would more naturally imply
a time-dependent light-curve calculation, which ROBERT does not perform here.

The planetary radius is an explicit reference radius at a configured reference
pressure. Reference pressures outside the atmospheric grid are rejected rather
than extrapolated. Two gravity modes are available through the Python model
configuration:

- `constant`: one supplied or mass-derived reference gravity is used in all
  layers;
- `inverse_square`: layer-centre gravity and hydrostatic radii are iterated to
  consistency with `g(r) = g_ref (R_ref / r)^2`.

Inverse-square gravity is therefore a specific variable-gravity model. Moving
upward to lower pressure normally increases radius and decreases gravity. The
same converged layer-gravity profile is used for vertical gas columns and
spherical path geometry. Transmission results retain the path geometry,
effective radius, annulus-area contributions, and top/bottom vertical optical-
depth diagnostics.

The sign convention and formal solution follow Rutten (2003): optical depth
increases downward, `mu dI/dtau = I - S`, and the top boundary has no incident
thermal radiation. Thermal profiles are evaluated explicitly at pressure
edges because the Planck source is represented as linear in optical depth
within each layer. The implementation is tested against the absorbing formal
solution and the semi-infinite isothermal scattering suppression derived in
Appendix A of Taylor et al. (2021).

This remains an angular two-stream approximation. Strongly anisotropic
angle-resolved intensities require validation against SH4 or a high-stream
discrete-ordinates/matrix-operator calculation before science use.

Primary theory and validation sources:

- [Rutten (2003), *Radiative Transfer in Stellar Atmospheres*](https://robrutten.nl/rrweb/rjr-edu/coursenotes/rutten_rtsa_notes_2003.pdf),
  for the transfer equation, optical-depth convention, angular moments,
  scattering source function, formal solution, and Eddington limits;
- [Taylor et al. (2021), *How does thermal scattering shape the infrared spectra of cloudy exoplanets?*](https://doi.org/10.1093/mnras/stab1854),
  especially Appendix A for the isothermal/linear-Planck scattering limits and
  the science requirements inherited from the NEMESIS matrix-operator study;
- [Toon et al. (1989)](https://doi.org/10.1029/JD094iD13p16287), for the fast
  inhomogeneous two-stream source-function method used for PICASO parity.
- [Rooney, Batalha & Marley (2024)](https://doi.org/10.3847/1538-4357/ad05c5),
  for the four-term spherical-harmonics thermal solution and its high-stream
  validation.

Passing `thermal_integration_backend="auto"` uses the Numba thermal-integration
kernel when available and falls back to the NumPy reference path otherwise.
Use `thermal_integration_backend="numpy"` to force the readable reference
backend. Direct-beam single-scattering source runs currently remain on the
NumPy path.

## Geometry and Source Function

`DiscGeometry` is the RT-facing angular container. It stores a sequence of
`DiscPoint` samples with emission zenith cosine, normalized disc weight, and
optional projected-disc, latitude/longitude, stellar zenith, and stellar azimuth
metadata. The current cloud-free solver uses the emission cosine and weight to
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
synthetic P-T profile.

Current forward-model validation uses the maintained PICASO and
petitRADTRANS comparisons listed in `examples/BENCHMARKS.md`. Superseded
pre-YAML checks and their generated artifacts are not part of the active
validation workflow or Git history.

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

The Taylor et al. (2021) Figures 1–2 thermal-scattering reproduction is run with:

```bash
python examples/benchmark_taylor2021_figures_1_2.py
```

It overlays ROBERT against the archived NEMESIS Figure 1 forward spectra and
recreates Figure 2 using the archived TP profiles. ROBERT uses the existing
petitRADTRANS HDF5 H2O/CO and CIA tables; see
`docs/review/24_taylor2021_cloud_figures_1_2.md` for provenance and scope.

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
synthetic 64-layer, 900-wavelength, four-g benchmark on this laptop, the batched
NumPy two-stream solve takes about 0.24 s. This is suitable for validation but
still requires a compiled kernel before long nested-sampling retrievals.

## Scattering Boundary

Scattering must enter ROBERT in two separate ways:

- as extinction optical depth, such as Rayleigh, cloud, or aerosol opacity;
- as a source-function treatment when scattering redirects radiation into the
  line of sight.

The current cloud-free solver handles absorption, thermal emission, optional
extinction-only Rayleigh terms, and optional first-order single scattering of a
direct stellar beam. Clouds/aerosols, surfaces, and multiple-scattering source
terms should be added as independent RT components rather than hidden inside
gas opacity.

The first cloud/aerosol object follows the same separation: it stores optical
properties, but the RT backend decides whether scattering is extinction-only,
single-scattering direct beam, fast two-stream approximation, or a fuller
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
