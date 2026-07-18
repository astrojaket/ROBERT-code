# Emission intercomparison Version 2: Stage 4

> **Pre-matrix freeze:** The physical state, diagnostic definitions, wavelength
> bands/windows, numerical gates, and resource limits in this document were
> committed to the task worktree before the complete matrix was inspected.
> Measured results are added below without changing those declarations.

## Frozen scientific scope

Stage 4 compares the exact common-contract `1755 K` isothermal control, PG14
non-inverted profile, and PG14 inverted profile. The already-evaluated
temperature arrays in `common_contract.json` are supplied identically to every
framework on 40, 80, and 160 pressure cells; 80 cells is primary. The active
spectral domain is exactly 0.3--12 micron on 369 flux-conserving R=100 bins,
with every true native coordinate and native-resolution product retained.

Every case uses the exact frozen H2O, CO, CO2, and CH4 VMRs, H2/He background
fill, and declared mean molecular weight. Both H2--H2 and H2--He CIA are enabled
in every case. This fixed CIA state is the Stage-3
`molecular_plus_h2_h2_and_h2_he_cia` closure required by the canonical roadmap
and preceding reviews. It was selected before the pilot and is not a post-hoc
factor chosen after viewing Stage-4 results.

PICASO opacity sampling remains retired. The only active PICASO molecular path
is PICASO 4.0 resort-rebin correlated-k with the Stage-3 absolute summed
line-gas VMR restoration unchanged.

## Tracks and contribution definitions

Track A supplies an identical molecular-plus-both-CIA mean layer-optical-depth
tensor only to the compatible ROBERT and stable-pRT pure-absorption paths. No
PICASO identical-tensor gate is invented.

Track B holds the high-level physical state fixed while retaining native
database, interpolation, correlated-k, and radiative-transfer representations:

- ROBERT retains native random-overlap molecular tensors, separate CIA
  components, signed spectra, and its disk-integrated source decomposition;
- PICASO retains native total `taugas` and the exact-`omega0=0` native thermal
  probe as separate capability evidence. Its scientific spectrum and vertical
  diagnostic use the independently labelled absorbing formal solution applied
  to native `taugas`; these are not native SH contribution functions; and
- stable pRT retains its supported native spectrum and normalized
  `emission_contribution`. Its high-level interface does not expose a stable
  native layer optical-depth tensor, so none is fabricated.

Every normalized vertical product retains the complete pressure-by-wavelength
array, log-pressure centroid, and peak pressure. Similar-looking Track-B
diagnostics are compared descriptively and are not assumed to share a formal
definition.

## Predeclared numerical gates

Only matched Track-A definitions and analytic/convergence controls are gated.
Track-B native cross-framework comparisons have no acceptance gates.

| Diagnostic | Frozen limit |
| --- | ---: |
| Maximum Track-A ROBERT/pRT symmetric relative difference | `5e-4` |
| Maximum Track-A ROBERT/pRT eclipse-depth difference | `0.1 ppm` |
| Maximum Track-A 80-to-160 eclipse-depth change | `0.1 ppm` |
| Maximum isothermal analytic-control eclipse difference | `0.1 ppm` |
| Maximum Track-A contribution-centroid p95 absolute difference | `0.01 dex` |
| Maximum Track-A contribution-profile p95 total variation | `0.01` |
| Maximum absolute single-scattering albedo | `0` |

The resource gates are a projected complete-matrix wall time no greater than
`7200 s` and largest process peak RSS below 60 per cent of available memory.
The complete matrix stops if either pilot gate is exceeded.

The predeclared pressure-behaviour diagnostics are: optical (`0.3--0.8`
micron), near-IR water band (`1.35--1.55`), near-IR window (`2.0--2.3`),
methane band (`3.1--3.6`), CO/CO2 band (`4.2--5.0`), mid-IR water band
(`5.5--7.5`), and mid-IR window (`8.0--10.0`). These ranges are diagnostics,
not cross-framework gates.

## Preserved earlier-stage framing

Stage 1 remains an out-of-tolerance closure regime with an exact eight-angle
maximum of `0.1968967046 ppm`, the retained `0.196897 ppm` statement, and the
sub-`0.01 ppm` continuous-angle restriction. Stage 2 retains Track-A maxima of
`10.150094`, `2.558145`, and `0.640615 ppm` at 40/80/160 cells and an
80-to-160 change of `1.369643 ppm`. Stage 3 retains maxima of `7.110565`,
`1.788100`, and `0.447547 ppm`, an 80-to-160 change of `0.769349 ppm`, and the
corrected primary-grid both-CIA effects of `43.387801 ppm` for ROBERT,
`45.501171 ppm` for PICASO correlated-k, and `43.656935 ppm` for stable pRT.
None of these measurements is reclassified as a framework failure.

## Pilot, results, and verification

The initial cold-cache representative
`pg14_non_inverted_molecular_plus_h2_h2_and_h2_he_cia_80_cells` pilot took
`24.917126 s`. Its frozen factor of 18 projected `448.508267 s`, below the
`7200 s` limit. The largest process peak RSS was `4,127,260,672 bytes` against
`10,660,577,280 bytes` available, or `38.7152%`, below the 60 per cent limit.
Both resource gates authorized the matrix. The final reporting run repeated
the pilot after caches were populated (`11.397582 s`, `36.2124%`) and recorded
`92.041606 s` for the post-pilot matrix.

Report assembly in the first authorized run stopped on a shape-only indexing
error in the predeclared band/window summary after the solvers completed. No
scientific metric or gate was inspected. The single-case array index was
corrected, covered by a regression test, and the same frozen matrix was rerun;
no physics, opacity, profile, wavelength handling, diagnostic, or threshold
changed.

## Scientific outcome

Stage 4 is an `out_of_tolerance_closure_regime`. The isothermal analytic,
matched contribution, and exact-zero scattering gates are met. The frozen
Track-A spectral and 80-to-160 limits are exceeded. This is a measured,
vertically converging representation regime, not a failure classification for
ROBERT or stable pRT.

| Matched Track-A diagnostic | Frozen limit | Observed | Result |
| --- | ---: | ---: | --- |
| Maximum symmetric relative difference | `5e-4` | `1.865313e-2` | exceeded |
| Maximum eclipse-depth difference | `0.1 ppm` | `7.110565 ppm` | exceeded |
| Maximum 80-to-160 eclipse-depth change | `0.1 ppm` | `1.206041 ppm` | exceeded |
| Maximum isothermal analytic-control eclipse difference | `0.1 ppm` | `0.001043 ppm` | met |
| Contribution-centroid p95 absolute difference | `0.01 dex` | `0` | met |
| Contribution-profile p95 total variation | `0.01` | `0` | met |
| Maximum absolute single-scattering albedo | `0` | `0` | met |

The complete matched pressure-resolved diagnostics are identical to stored
precision because the shared track evaluates the same absorbing-formal layer
source definition on the same mean-optical-depth tensor. This exact diagnostic
closure does not imply identical framework-native contribution functions.

## Track-A spectral closure and convergence

The maximum eclipse-depth difference at each resolution remains set by the
PG14 non-inverted case inherited from Stage 3:

| Cells | Isothermal | PG14 non-inverted | PG14 inverted |
| ---: | ---: | ---: | ---: |
| 40 | `0.000537 ppm` | `7.110565 ppm` | `6.619660 ppm` |
| 80 | `0.000537 ppm` | `1.788100 ppm` | `1.667134 ppm` |
| 160 | `0.000537 ppm` | `0.447547 ppm` | `0.417414 ppm` |

The inverted profile sets the largest symmetric relative difference at every
resolution: `1.865313e-2`, `4.717567e-3`, and `1.182411e-3`. Both PG14 cases
decrease by approximately fourfold per grid doubling. The largest Track-A
80-to-160 change is `1.206041 ppm` for stable pRT's inverted shared-tau case;
the corresponding PG14 non-inverted value is the preserved Stage-3
`0.769349 ppm`.

## Native thermal-structure effects

All three native frameworks show a large, resolved thermal-structure signal.
At 80 cells the maximum absolute R=100 eclipse-depth change between the PG14
inverted and non-inverted profiles is `791.714071 ppm` for ROBERT,
`661.595699 ppm` for PICASO correlated-k, and `792.655774 ppm` for stable pRT.
Relative to the exact isothermal control, the largest PG14 non-inverted changes
are `875.762781`, `861.081231`, and `875.496242 ppm`, respectively. These are
within-framework physical contrasts, not cross-framework gates.

The common-contract shape checks are satisfied. The non-inverted profile is
strictly warmer with increasing pressure over all 80 cells (`1346.887526` to
`2057.844914 K`). The inverted profile contains both gradient signs and spans
`1363.978244` to `1985.090807 K`. The isothermal array is exactly `1755 K`.

## Track-B spectra and contribution-pressure behavior

At the primary 80-cell resolution the largest native ROBERT/stable-pRT
difference is `9.873270 ppm` for the inverted profile. The largest comparison
involving PICASO is `125.715939 ppm` for PICASO versus stable pRT in the
inverted profile; ROBERT versus PICASO is `124.136966 ppm`. For the
non-inverted profile the corresponding maxima are `5.983625`, `73.620040`, and
`71.513076 ppm`. These combine database, interpolation, correlated-k, and
radiative-transfer representation effects and do not rank the frameworks.

The largest primary-grid native pressure-centroid RMS difference is
`0.129370 dex` for PICASO versus stable pRT in the inverted profile. The
largest p95 vertical-profile total variation is `0.126104` for the isothermal
PICASO/stable-pRT diagnostic comparison. ROBERT/stable-pRT centroid RMS
differences are `0.042846`, `0.042858`, and `0.041766 dex` for the isothermal,
non-inverted, and inverted profiles.

The optical (`0.3--0.8 micron`) diagnostic peaks in the deepest 80-cell
pressure coordinate (`90.416981 bar`) for every framework and profile. In the
near- and mid-IR, the PG14 non-inverted median centroids generally lie around
`10^-2.5` to `10^-1.25 bar`, while the inverted profile moves several water-
band diagnostics to lower pressures: for example, the `5.5--7.5 micron`
median centroid is `10^-3.061 bar` for ROBERT and `10^-3.103 bar` for stable
pRT, compared with `10^-2.460` and `10^-2.507 bar` in the non-inverted case.
PICASO's corresponding centroids are `10^-2.904` and `10^-2.320 bar`.
Complete wavelength-by-wavelength centroids and peaks are retained, so these
band medians do not replace the full vertical arrays.

## Native vertical convergence

From 80 to 160 cells, the largest native spectral changes are `0.876201 ppm`
for ROBERT (non-inverted), `1.057130 ppm` for PICASO (inverted), and
`0.577684 ppm` for stable pRT (inverted). Their largest contribution-centroid
RMS changes are `0.000280`, `0.002543`, and `0.019489 dex`, respectively. The
largest p95 total-variation changes are `0.000842`, `0.007486`, and `0.026018`.
These are native representation convergence measurements, not cross-framework
acceptance gates.

## Artifacts and capability boundaries

The report is accompanied by 24 numerical NPZ shards and
`stage_4_integrity.json`. ROBERT's native products are sharded by profile so
the complete 16-point-g molecular tensors and separate H2--H2/H2--He arrays
remain intact. The largest committed binary is `40,782,104 bytes`, well below
GitHub's `100,000,000-byte` object limit. No numerical precision was lowered
and no native tensor was dropped.

PICASO's native total `taugas`, exact-zero Rayleigh/cloud evidence, and
pathological exact-`omega0=0` thermal probe remain stored separately from the
absorbing-formal spectrum and vertical diagnostic. The metadata records the
unchanged absolute line-VMR restoration. The Stage-3 unit regression for the
restoration remains active, and Stage 4 adds an artifact regression for its
metadata and summed VMR. Stable pRT has no fabricated native optical-depth
array.

The harmless optional-Vega and exact-zero cloud/Rayleigh warnings were
recorded on every PICASO run. No data were downloaded, warnings suppressed, or
exact zeros replaced with epsilons. PICASO opacity sampling was not run,
regenerated, plotted, checksummed, or reintroduced.

## Verification

Final verification used the exact isolated interpreters and frozen data paths:

- focused Version-2 contract, artifact, integrity, and strict SHA-256 tests:
  `69 passed`;
- Ruff over the complete repository: passed;
- exact-environment smokes: ROBERT `0.3.0`, PICASO `4.0` resort-rebin
  correlated-k over 661 points, and stable petitRADTRANS `3.3.3` all passed;
- complete ROBERT suite: `512 passed, 5 skipped`; and
- committed-binary audit: all 24 Stage-4 NPZ shards are below
  `100,000,000 bytes`, with a maximum of `40,782,104 bytes`.

The first stable-pRT smoke invocation used its repository-relative default and
reported the expected absent local opacity tree. The required smoke was then
run successfully with the frozen absolute input-data path used by the matrix.
The PICASO smoke emitted the recorded harmless optional-Vega warning. The
scientific gate exceedances remain measured results and were not treated as
orchestration or test failures.
