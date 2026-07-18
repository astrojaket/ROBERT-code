# Emission intercomparison Version 2: Stage 2

> **Active contract revision:** this stage is regenerated over `0.3--12 micron`
> on 369 R=100 bins using PICASO resort-rebin correlated-k only. Opacity
> sampling is retired. The Stage-2 numerical gates are unchanged and remained
> frozen before the revised matrix was inspected. Historical `0.8--12 micron`
> values remain recoverable in Git history.

## Frozen scope and gates (written before the complete matrix)

Stage 2 measures single-molecule closure for H2O, CO, CO2, and CH4 at the
exact reference VMRs serialized in `version_2/common_contract.json`.  The
isothermal control, PG14 non-inverted profile, and PG14 inverted profile use
the common-contract arrays on 40, 80, and 160 cells; 80 cells is primary.
H2 and He fill the remainder in the serialized contract ratio, and the mean
molecular weight is recomputed from the serialized molecular masses.

Track A supplies one identical pressure-cell-by-wavelength optical-depth
tensor to genuinely compatible ROBERT and stable-pRT pure-absorption paths.
It is a g-weighted single-molecule diagnostic derived from ROBERT's independent
load of the pRT-HDF table.  PICASO 4.0 does not expose a finite native
exact-`omega0=0` identical-tensor interface, so no PICASO Track-A gate is
invented.  The independently labelled Stage-1 absorbing formal reference is
not promoted to a native PICASO result.

The following numerical gates were frozen in code and this review before the
complete matrix was inspected:

| Matched Track-A diagnostic | Limit |
| --- | ---: |
| Maximum ROBERT/pRT symmetric relative difference | `5e-4` |
| Maximum ROBERT/pRT eclipse-depth difference | `0.1 ppm` |
| Maximum Track-A 80-to-160 eclipse-depth change | `0.1 ppm` |
| Maximum isothermal analytic-control eclipse difference | `0.1 ppm` |
| Maximum absolute single-scattering albedo | `0` |

The pilot limits remain `7200 s` projected wall time and peak RSS below 60 per
cent of available memory.  Native Track-B spectra have no cross-framework
acceptance gate because their databases, interpolation, and correlated-k
representations are not identical.

## Pre-matrix pilot

The representative 80-cell H2O/PG14-non-inverted three-framework pilot took
`25.047740 s`, projecting `1052.005071 s` for the complete workload. Its
largest process peak RSS was `3,636,068,352 bytes`, or `35.58%` of the
`10,218,323,968 bytes` available at the decision. The frozen resource gates
therefore authorize the complete matrix.

## Track-B representation contract

PICASO's mandatory primary path is the official PICASO-4 resort-rebin
correlated-k family and exact four asset checksums serialized in the common
contract. Opacity sampling is retired and not run. ROBERT and stable pRT construct native opacity independently from
the local pRT-HDF family.

PICASO native optical-depth tensors are retained.  Its pressure-resolved array
is explicitly an absorbing-formal diagnostic applied to the native optical
depth, not a native SH contribution definition.  Stable pRT's supported
high-level native-flux interface does not expose its layer optical-depth
tensor; its native spectrum and native emission-contribution array are
retained and the missing tensor is recorded as a capability boundary.

## Results

The post-pilot matrix took `167.724046 s`. The result is an
`out_of_tolerance_closure_regime`: the isothermal analytic control and exact
zero-scattering gates are met, while three frozen Track-A limits are exceeded.
This classification measures the pressure/source discretization difference;
it does not classify either framework as failed.

| Track-A resolution | Maximum symmetric relative difference | Maximum eclipse difference |
| ---: | ---: | ---: |
| 40 cells | `1.867147e-2` | `10.150094 ppm` |
| 80 cells | `4.722222e-3` | `2.558145 ppm` |
| 160 cells | `1.183580e-3` | `0.640615 ppm` |

The maximum 80-to-160 Track-A change is `1.369643 ppm`, above its `0.1 ppm`
gate.  The approximately fourfold reduction of the shared-path maximum at
each grid doubling localizes the result to a vertically converging
representation regime.  At 80 cells the largest case difference is the CO
PG14 non-inverted profile (`2.558145 ppm`). Complete per-species/profile
metrics remain in `stage_2_report.json`; complete optical depths, spectra,
and vertical arrays remain in the NPZ products.

Both isothermal analytic controls meet their gate: ROBERT's maximum eclipse
difference is `0.001043 ppm` and stable pRT's is `0.000510 ppm`. These are
separate from the Stage-1 eight-angle continuous-angle result and do not relax
its sub-`0.01 ppm` claim restriction.

Track B is attribution-only.  On the primary 80-cell grid, the largest native
ROBERT/pRT R=100 difference is `50.317721 ppm`. The primary PICASO
resort-rebin opacity with the separately labelled absorbing-formal reference
differs from stable pRT by as much as `182.554883 ppm`, measuring the combined
database/interpolation/correlated-k representation effect.  PICASO's native
exact-`omega0=0` high-level thermal probe is finite but pathological, so it is
retained as capability evidence rather than used as the scientific spectrum;
the native PICASO opacity tensor is unchanged.

The corrected PICASO resort-rebin worker restores the absolute line-gas VMR
discarded by the normalized random-overlap mixer. A controlled H2O abundance
probe gives integrated optical-depth ratios `0.50055`, `1.00000`, and
`1.99563` for the `0.5x`, `1x`, and `2x` cases after the expected mean-
molecular-weight response. The retained PICASO spectra now show molecular
features and no opacity-sampling comparison is made.

All four exact reference VMRs, three temperature profiles, 40/80/160 grids,
and the 0.5/1/2 H2O abundance check were exercised.  The report records raw
case timings, exact interpreters and package/data/source checksums, process
peak RSS, warnings, and unsupported diagnostic interfaces.  The integrity
manifest records every committed array's shape, dtype, finite policy, size,
and checksum. The 160-cell ROBERT outputs are sharded by species and profile;
all committed artifacts are below 100 MB without reducing numerical precision
or dropping native tensors.
