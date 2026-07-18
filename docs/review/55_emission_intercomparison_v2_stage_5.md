# Emission intercomparison Version 2: Stage 5

> **Pre-matrix freeze:** The physical state, perturbation definitions,
> diagnostics, numerical gates, and resource limits in this document were
> written before the complete matrix was inspected. Measured results are added
> below without changing those declarations.

## Frozen scientific scope

Stage 5 uses the unchanged Version-2 Stage-4 state: the exact `1755 K`
isothermal, PG14 non-inverted, and PG14 inverted arrays on 40, 80, and 160
cells, with 80 cells primary; the exact fixed H2O, CO, CO2, CH4, H2, and He
mixture and declared mean molecular weight; and both H2--H2 and H2--He CIA.
Clouds, Rayleigh scattering, and all scattering remain exactly disabled. The
active domain is exactly 0.3--12 micron with 369 flux-conserving equal-log
R=100 bins and exact endpoints. True native coordinates and spectra remain
separate from the lower-edge closure used only during bin integration.

PICASO opacity sampling remains retired. PICASO 4.0 resort-rebin correlated-k
is its only molecular path, and the Stage-3 absolute summed line-gas VMR
restoration is unchanged.

## Frozen perturbation and finite-difference design

The six pressure centres are `1e-4`, `1e-3`, `1e-2`, `1e-1`, `1`, and
`10 bar`. At centre pressure `P0`, the continuous localization is

`L(P;P0) = exp[-0.5 (log10(P/P0) / 0.35 dex)^2]`.

The primary perturbations are `T +/- 10 K L` and the signed Jacobian is

`J(P0,lambda) = [F(T + 10 L) - F(T - 10 L)] / 20 K`.

The continuous kernel is evaluated directly at the frozen ROBERT/stable-pRT
cell centres and at the frozen ROBERT/PICASO cell edges. No nearest-cell snap,
trimming, or renormalization is applied. The edge/cell relationship therefore
matches the Stage-4 framework interfaces exactly. Baseline, plus, and minus
spectra are retained before differencing.

The primary 80-cell PG14 non-inverted case additionally uses predeclared
`+/-5 K` and `+/-20 K` perturbations at all six centres. Linearity compares
those centered derivatives with the `+/-10 K` derivative. Symmetry measures
the centered even residual
`F(+A) + F(-A) - 2 F(0)` relative to the primary signed response scale. These
are diagnostics of finite-difference behaviour, not cross-framework native
agreement gates.

At each wavelength, the normalized absolute response is `abs(J)` divided by
its sum across all six centres. If and only if that sum is exactly zero, the
complete normalized column is stored as exact zeros. No epsilon or numerical
noise floor is introduced. Isothermal diagnostics that are analytically zero
by construction likewise remain exact zero.

## Frozen tracks and capability boundaries

Track A applies the localized temperature perturbations while holding each
profile's completed Stage-4 ROBERT shared mean molecular-plus-both-CIA layer
optical-depth tensor fixed. Only the source and lower-boundary Planck terms
change. The identical tensor and temperature arrays are supplied to the
compatible ROBERT and stable-pRT absorbing paths. PICASO has no invented
identical-tensor gate.

Track B recomputes each framework's native molecular and CIA opacity for every
required baseline, plus, and minus state. Native cross-framework differences
are attribution-only and have no acceptance gate.

Stable pRT's supported high-level native flux interface does not expose a
stable native layer optical-depth tensor, so none is fabricated. PICASO native
total `taugas`, exact-zero cloud/Rayleigh evidence, and pathological exact-
`omega0=0` thermal probe remain separate from the independently labelled
absorbing-formal spectrum and vertical diagnostics. PICASO absorbing-formal
diagnostics are not called native SH contribution functions.

Stage-4 contribution functions are projected through the same localization
kernels for comparison with Stage-5 responses. They are not identical
quantities: source contribution decomposes emergent flux, whereas a signed
temperature derivative includes `dB/dT`, and Track B additionally includes
temperature-dependent opacity derivatives.

## Predeclared numerical and resource gates

Only matched Track-A definitions, analytic controls, finite-difference
behaviour, and convergence are gated. Track-B native comparisons remain
attribution-only. A threshold exceedance is reported as a measured regime and
never as a framework failure.

| Diagnostic | Frozen limit |
| --- | ---: |
| Primary Track-A Jacobian p95 absolute difference divided by pair peak | `0.05` |
| Primary Track-A eclipse-Jacobian RMS difference | `0.02 ppm/K` |
| Primary Track-A response-centroid RMS difference | `0.15 dex` |
| Primary Track-A response-profile p95 total variation | `0.08` |
| Track-A 80-to-160 Jacobian p95 change | `0.05` |
| Track-A 80-to-160 eclipse-Jacobian RMS change | `0.02 ppm/K` |
| Track-A 80-to-160 response-centroid RMS change | `0.15 dex` |
| Track-A 80-to-160 response-profile p95 total variation | `0.08` |
| Track-A isothermal baseline analytic-control eclipse difference | `0.1 ppm` |
| Primary finite-difference 5/10/20 K linearity p95 relative difference | `0.02` |
| Primary finite-difference symmetry p95 relative even residual | `0.02` |
| Exact-zero normalization/control maximum | `0` |

The representative pilot is the 80-cell PG14 non-inverted baseline and all
six primary `+/-10 K` perturbations in ROBERT, PICASO 4.0, and stable pRT.
The full matrix is authorized only if projected wall time is at most `7200 s`
and the largest process-tree peak RSS is below 60 per cent of available
memory. The workload projection multiplier is frozen at `12`.

## Preserved earlier-stage framing

Stage 1 retains the exact `0.1968967046 ppm` eight-angle maximum, the rounded
`0.196897 ppm` statement, and the sub-`0.01 ppm` continuous-angle restriction.
Stage 2 retains Track-A maxima `10.150094`, `2.558145`, and `0.640615 ppm` and
the `1.369643 ppm` 80-to-160 change. Stage 3 retains `7.110565`, `1.788100`,
and `0.447547 ppm`, its `0.769349 ppm` 80-to-160 change, and corrected
both-CIA effects of `43.387801`, `45.501171`, and `43.656935 ppm` for ROBERT,
PICASO, and stable pRT. Stage 4 retains its out-of-tolerance, vertically
converging closure framing and all reported measurements. None is rewritten
or reclassified.

## Pilot, scientific results, and verification

The final reporting run's representative pilot measured `45.499764 s` for 13
states in each native framework. Its frozen factor of 12 projected
`545.997169 s`, below `7200 s`. The largest process-tree member peak RSS was
`4,287,512,576 bytes` against `10,609,967,104 bytes` available, or `40.4102%`,
below the frozen 60 per cent limit. Both resource gates authorized the matrix.

All matched Track-A gates pass:

| Diagnostic | Limit | Observed | Result |
| --- | ---: | ---: | --- |
| Primary Jacobian p95 difference / pair peak | `0.05` | `0.00168817` | met |
| Primary eclipse-Jacobian RMS difference | `0.02 ppm/K` | `0.000786942 ppm/K` | met |
| Primary response-centroid RMS difference | `0.15 dex` | `0.00174522 dex` | met |
| Primary response-profile TV p95 | `0.08` | `0.00252234` | met |
| 80-to-160 Jacobian p95 change | `0.05` | `0.00151578` | met |
| 80-to-160 eclipse-Jacobian RMS change | `0.02 ppm/K` | `0.000648841 ppm/K` | met |
| 80-to-160 response-centroid RMS change | `0.15 dex` | `0.000829095 dex` | met |
| 80-to-160 response-profile TV p95 | `0.08` | `0.00207203` | met |
| Isothermal analytic-control eclipse difference | `0.1 ppm` | `0.001043488 ppm` | met |
| 5/10/20 K linearity p95 | `0.02` | `5.31613e-5` | met |
| Symmetry even-residual p95 | `0.02` | `0.00963057` | met |
| Exact-zero normalization maximum | `0` | `0` | met |

The primary Track-A Jacobian p95 value is `0.168817%`. This is tighter than
the gate but does not revise Stage 4's spectral closure measurements: Stage 5
asks a derivative/source-response question with the frozen Stage-4 opacity
field, not the baseline-flux identity tested previously.

## Native attribution and convergence

Track B is attribution-only. At 80 cells, the largest ROBERT/stable-pRT
Jacobian p95 difference is `0.515848%`, the largest eclipse-Jacobian RMS
difference is `0.00413243 ppm/K`, the largest response-centroid RMS difference
is `0.00817991 dex`, and the largest response TV p95 is `0.0117007`. Pairs
involving PICASO reach `4.05137%`, `0.0433885 ppm/K`, `0.0907680 dex`, and
`0.0949702`, respectively. These values combine native database,
temperature interpolation, correlated-k, and RT-representation effects; they
do not rank or fail frameworks.

Native 80-to-160 Jacobian p95 changes are `0.132459%` for ROBERT,
`0.158200%` for PICASO, and `0.197647%` for stable pRT. The corresponding
response TV p95 values are `0.00182100`, `0.00226342`, and `0.00330296`.
Complete 40-to-80 and 80-to-160 R=100 spectral-response convergence is stored
in the report.

The Stage-4 contribution projection and Stage-5 response remain related but
non-identical. On the 80-cell inverted case their largest p95 total-variation
distances are `0.126132` for ROBERT, `0.136313` for PICASO, and `0.119270` for
stable pRT. The complete wavelength-resolved response tensors, centroids,
peaks, and seven predeclared optical-to-mid-IR band/window summaries are
retained rather than replaced by these headline values.

## Artifacts and capability evidence

Stage 5 writes 154 report/numerical/integrity products totalling
`2,709,194,860 bytes`. ROBERT perturbed opacity is sharded one state per file;
PICASO total `taugas` is sharded by profile and resolution. The largest object
is `42,005,798 bytes`, below GitHub's `100,000,000-byte` limit. Precision was
not reduced and required native tensors were not dropped. ROBERT baseline
opacity tensors remain checksum-linked Stage-4 products rather than duplicated.

The PICASO artifacts retain the unchanged absolute-line-VMR correction
metadata, native total `taugas`, native exact-zero probe, and exact-zero cloud
and Rayleigh evidence. The absorbing-formal spectra and vertical diagnostics
remain independently labelled. Stable pRT retains no fabricated native layer
optical-depth tensor.

The optional-Vega and exact-zero cloud/Rayleigh warnings were recorded. No
data were downloaded, warnings suppressed, or exact zeros replaced by
epsilons. PICASO opacity sampling was not run, regenerated, plotted,
checksummed, or reintroduced.

Final verification used the mandated isolated environments. The focused
Version-2 contract/artifact/integrity run passed `79` tests, Ruff passed over
the repository, and the complete suite passed `522` tests with `5` expected
skips. Exact-environment smokes passed for ROBERT `0.3.0`, PICASO `4.0`
resort-rebin correlated-k (`661` native points), and stable pRT `3.3.3` (`49`
native points). Strict SHA-256 verification covered every declared Stage-5
product. The final Git index audit found no blob at or above `100,000,000`
bytes.
