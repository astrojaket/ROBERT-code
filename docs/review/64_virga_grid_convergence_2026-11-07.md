# Virga pressure and particle-radius grid tests

These local forward tests follow the [solar-composition cloud
sweep](63_virga_jwst_cloud_sweep_2026-11-07.md). They keep its planet, star,
three temperature-pressure profiles, solar C/O and metallicity, 15 gas
opacities including VO, and 300 logarithmic wavelength bins from 0.6 to
12 microns. Each model uses one condensate. The tests use Virga 2.0.2 and
ROBERT commit `744bac22dcd3f6e58f14615807ce599262cae808` in the
`robert-exoplanets` environment. All forward solves use matched clear controls
on the same grid and the same gas-opacity policy as the cloudy calculation.

## Native pressure grids

The 64-layer baseline spans 1e-5 to 100 bar. The 60- and 56-layer grids
remove four and eight top layers, respectively, and keep the remaining
edges. The 128-layer grid splits each baseline layer at its midpoint in
log pressure. The tests recalculate FastChem, Virga and radiative transfer
on each pressure grid. The cloud-only mean molecular weight stays fixed
within each temperature-pressure profile.

The six baseline models reproduce the archived cloudy spectra, cloud
optical depths, condensate profiles and effective radii exactly. Of the
24 cloud states tested, 23 give both spectra. Virga rejects reference
Al2O3 at fsed = 0.1 and log10(Kzz) = 7 on the 128-layer grid. It cannot find
a physical fall-speed solution at that grid spacing. The result record
retains the rejection and the native exception.

The table gives peak absolute changes in cloudy minus clear depth from the
64-layer result. Each pair lists emission and transmission in ppm.
Kzz is in cm2/s and the size-distribution width is 2 in every row.

| Case | fsed | log10(Kzz) | 60 layers | 56 layers | 128 layers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference Mg2SiO4 | 0.1 | 7 | 1364 / 386 | 2045 / 601 | 523 / 1.17 |
| Reference Mg2SiO4 | 0.1 | 11 | 47.8 / 138 | 82.9 / 280 | 1.02 / 0.99 |
| Cool Mg2SiO4 | 10 | 11 | <0.001 / <0.001 | <0.001 / <0.001 | 0.012 / <0.001 |
| Reference Al2O3 | 0.1 | 7 | 567 / 360 | 878 / 706 | Virga rejection |
| Hot Al2O3 | 0.1 | 10 | 2.11 / 16.5 | 6.96 / 53.2 | 0.61 / 1.52 |
| Reference MgSiO3 | 1 | 9 | 0.43 / 0.25 | 1.66 / 1.92 | 2.04 / 1.42 |

Changes in the clear atmosphere also affect the cloudy depths. For example,
removing eight top layers changes the fiducial MgSiO3 cloudy transmission
spectrum by up to 88.7 ppm. Its matched clear spectrum changes by up to
88.8 ppm. The 128-layer comparison changes the same cloudy spectrum by
2.14 ppm. Keep the model top and the number of layers separate when
interpreting these tests.

## Pressure above the gas tables

The 15 native correlated-k tables span about 1e-5 to 100 bar. Separate local
PICASO tables reach 1e-6 bar for H2O, CO, CO2 and CH4. Their spectral and g-ordinate grids
differ from the 20-point NEMESIS inputs used here. A production test above
1e-5 bar requires a consistent low-pressure opacity set.

The extended-top tests add log-spaced layers at constant top temperature
above the unchanged native pressure edges. These are sensitivity tests.
The 72-layer grid reaches 1e-6 bar. The 80-layer grid reaches 1e-7 bar.
They use either the pressure-floor gas coefficients at local temperature
or zero molecular line opacity in the added layers. They keep the gas
state, continuum, geometry and cloud calculation in both cases. The two
policies sample the effect of the gas assumption and give no guaranteed
bounds on the physical solution.

The native layers retain the same temperature, gas abundances, molecular
optical depths and continuum optical depths. Four of the six cloud cases
produce spectra at both new tops. Virga rejects both reference cases at
fsed = 0.1 and log10(Kzz) = 7 (Mg2SiO4 and Al2O3). The same fall-speed
condition causes both rejections. Each valid cloud state supplies the two
gas-policy spectra in both geometries.

- Reference Mg2SiO4 at fsed = 0.1 and log10(Kzz) = 11 changes by about
  327 ppm in transit depth at 9.68 microns on the 1e-7 bar grid, under
  either gas policy. The same feature changes by about 247 ppm at 1e-6
  bar. The line-free extension retains the change, which shows that the
  cloud and atmospheric geometry are important in this case.
- For hot Al2O3 at fsed = 0.1 and log10(Kzz) = 10, the peak change at
  1e-7 bar is 148 ppm in transit depth with floor-row gas opacity and
  19.3 ppm with no added-layer molecular lines. These are separate
  wavelength maxima. The fiducial MgSiO3 transmission spectrum changes by
  45.6 and 0.067 ppm under the two policies, respectively.

## Particle-radius grids

The native Virga tables have 60 radius centres from 0.0001 to
542.391 microns. The 96-centre grids keep these limits. For Mg2SiO4, the
107- and 114-centre grids use the same logarithmic spacing as the
96-centre grid and extend to 3.266 and 10.240 mm. The tests keep the optical
constants and all 713 wavelength points of the original Mie calculation.
The new tables pass readback checks for the radius grid and the physical
bounds of extinction, scattering and asymmetry. Changing the radius grid
can change the Virga fall-speed fit and the condensate and particle profiles
as well as the optical integration.

All 11 radius variants produce both spectra. The table gives the peak
absolute change in cloudy depth relative to the native 60-centre calculation.
Each pair lists emission and transmission in ppm.

| Case | fsed | log10(Kzz) | 96 radii | 107 radii | 114 radii |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference Mg2SiO4 | 0.1 | 7 | 90.1 / 28.1 | 90.1 / 30.7 | 113.9 / 37.8 |
| Reference Mg2SiO4 | 0.1 | 11 | 10.7 / 1.45 | 11.9 / 1.80 | 13.3 / 2.60 |
| Cool Mg2SiO4 | 10 | 11 | 0.097 / <0.001 | 2.52 / 0.0050 | 2.56 / 0.0051 |
| Reference Al2O3 | 0.1 | 7 | 32.3 / 3.84 | — | — |
| Hot Al2O3 | 0.1 | 10 | 1.14 / 0.67 | — | — |

The extreme reference Mg2SiO4 case still changes by 23.8 ppm in eclipse
depth and 7.04 ppm in transit depth between the two widest grids. All
radius variants preserve the original lower radius centre. States that
require smaller radii remain outside the tested range.

For the cool, large-particle Mg2SiO4 case, the condensate-weighted mass
tail beyond the upper radius-bin edge is 85.4% on the native grid and
86.8% on the 96-centre grid. It falls to 7.06% on the 107-centre grid and
0.093% on the 114-centre grid. The two wide grids differ by only 0.044 ppm
in eclipse depth and 0.000082 ppm in transit depth in this case.

## Virga reuse

The paired Python API, `VirgaEddySedCloudModel.evaluate_with_profiles`,
returns the usual optical properties and the per-species cloud profiles
from one native Virga call. The posterior collector also reuses the raw
Virga result when the spectrum and the cloud profiles use the same
provider, atmospheric state, gravity and resolved cloud controls. The
collector starts a fresh scope for each draw. The individual optics-only
and profiles-only APIs remain available for direct use.

The benchmark uses code commit
`4f4f220f23b0a8b387a93763bce252403b94f0d8` and the saved local
R=100 emission and noiseless transmission configurations. Each timed
batch evaluates the same three parameter draws. The reference path uses
the same code with the per-draw scope disabled. Times are medians of three
warm batches on one numerical thread.

| Geometry | Virga calls before / after | Batch time before / after | Speedup |
| --- | ---: | ---: | ---: |
| Emission | 6 / 3 | 6.90 s / 3.66 s | 1.89x |
| Transmission | 6 / 3 | 6.64 s / 3.34 s | 1.99x |

The spectra, posterior quantiles and cloud profiles agree exactly. The
paired API also gives exactly the same optical properties and physical
profiles as the separate APIs, with one native call in place of two.

## Checks and next tests

Across the three grid tests, 42 of 47 cloud states produce spectra. Virga
rejects the remaining five under its fall-speed validity check. A saved
comparison script checks each state file hash and the 300 wavelength
centres and 301 edges before comparing spectra. The separate pressure
extension verifier passes 122 checks, including retained native-layer
parity, shared continuum, clear controls and cloud-minus-clear products.
The extension runner finished both grids before its comparison step
stopped on a protocol key. The saved-array comparison completed separately.

The local test suite on the reuse change passes 1,099 tests with 36
optional or external-data skips. The run enables the real Virga tests.
Ruff passes on all changed Python files.

1. Obtain a consistent molecular-opacity set that reaches at least 1e-7
   bar. Repeat the top-sensitive, valid cases with that set.
2. Check the radius support of each cloud layer before using extreme
   spectra. Prioritize condensate-weighted tails and fall-speed validity.
3. Compare pressure and radius refinement together for selected strong,
   supported cases. Choose a spectral tolerance for each application.

## Result files

The external artifact root contains the full run arrays, figures, source
snapshots, input hashes and execution logs. It is
`/Users/jaketaylor/.codex/visualizations/2026/11/06/01a1b0d4-f26e-7571-83c0-e1a6be3d8b9a/virga-grid-convergence-20261107`.
The comparison figures show the change in cloudy depth relative to the
native 64-layer, 60-radius baseline. The matching JSON files also report
changes in the clear spectra and the cloudy minus clear spectra.

- `figures/grid_comparisons.json` gives the metrics and figure captions.
- `runs/native_pressure_grid_comparison.json` records native pressure parity.
- `runs/extensions/extension_policy_comparison.json` compares the added layers
  and the two gas policies.
- `runs/native_64/r96_same_bounds/radius_grid_comparison.json` gives the
  particle-grid comparisons.
- `runs/native_64/r96_same_bounds/radius_tail_diagnostics.json` gives the
  weighted area and mass tails at the actual bin edges.
- `virga_reuse_benchmark.json` and `logs/pytest_virga_reuse.log` record the
  code checks.
