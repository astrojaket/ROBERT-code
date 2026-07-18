# Emission intercomparison Version 2: Stage 6

> **Pre-matrix freeze:** The scientific state, perturbation design, diagnostics,
> numerical gates, resource limits, and capability boundaries below are frozen
> before the representative pilot or complete matrix is inspected. Measured
> results will be appended without changing these declarations.

## Frozen scientific scope

Stage 6 inherits the completed Version-2 Stage-5 physical state without
reinterpretation: the exact already-evaluated PG14 non-inverted and PG14
inverted temperature arrays on 40, 80, and 160 cells, with 80 cells primary;
the exact common-contract H2O, CO, CO2, CH4, H2, and He baseline mixture and
molecular masses; and both H2--H2 and H2--He CIA. Temperature is fixed within
every composition finite difference. Clouds, Rayleigh scattering, and all
scattering remain exactly disabled.

The physical matrix contains the two PG14 profiles. The 1755 K isothermal
blackbody atmosphere is retained separately as an analytic-zero composition-
response control. Its composition Jacobian and normalized response are stored
as exact zeros; floating-point cancellation is not normalized into a signal.

The active domain is exactly 0.3--12 micron with 369 flux-conserving equal-log
R=100 bins and exact endpoints. True native coordinates and spectra remain
separate from the constant-value lower-edge closure used only to integrate a
correlated-k table whose first stored centre lies above 0.3 micron. A boundary
closure value is never labelled native data.

PICASO opacity sampling remains retired. PICASO 4.0 resort-rebin correlated-k
is its only active molecular path. The Stage-3 restoration of the absolute
summed line-gas VMR after PICASO's normalized mixer is applied independently
to every composition state using that state's pressure-dependent summed
H2O+CO+CO2+CH4 VMR. It is not frozen to the baseline and no opacity asset or
correction factor is tuned.

## Frozen composition perturbation contract

The target species are H2O, CO, CO2, and CH4, one at a time. Non-target
molecular VMRs remain fixed. At the six pressure centres `1e-4`, `1e-3`,
`1e-2`, `1e-1`, `1`, and `10 bar`, the localization is

`L(P;P0) = exp[-0.5 (log10(P/P0) / 0.35 dex)^2]`.

For the primary half-step `Delta=0.10 dex`,

`q_target,+/-(P) = q_target,reference 10^[+/- Delta L(P;P0)]`.

After changing the target, the exact remaining-atmosphere fraction is split
layer by layer as `H2 = 0.8547 remainder` and `He = 0.1453 remainder`. This is
the common-contract Version-2 ratio, not Version 1's approximate 85:15 split.
All six gases are checked to sum to unity at every edge and cell centre. Mean
molecular weight is the VMR-weighted sum of the six declared molecular masses
and is recomputed at every point. H2--H2 and H2--He CIA are likewise
recomputed for every perturbed state.

The continuous kernel and composition are evaluated independently on the
frozen pressure edges supplied as PICASO levels and on the frozen geometric
cell centres used by ROBERT and stable pRT. There is no nearest-cell snap,
kernel renormalization, trimming, or profile interpolation. The exact frozen
PG14 cell arrays are reused; their edge arrays use the established Stage-4
canonical evaluation required by the framework interface.

Baseline, plus, and minus native and R=100 spectra and eclipse depths are
retained. The centered signed composition Jacobian is

`J_s(P0,lambda) = [F(q_s,+) - F(q_s,-)] / (2 Delta)`

in `W m^-2 m^-1 dex^-1`; its eclipse form is in `ppm/dex` under the exact
6550 K blackbody and WASP-17 projected-area ratio. The signed half-response is
`[F(+) - F(-)]/2` in spectral-density units.

For each target species and wavelength, the normalized absolute pressure
response is `abs(J)` divided by its sum over the six pressure centres. An
exact-zero sum produces an exact-zero column. Own/cross-species sensitivity
fractions integrate `abs(J)` over pressure and normalize over the four target
species at each wavelength; an exact-zero total again produces exact zeros.
Pressure centroids use the normalized response weighted by `log10(P/bar)`,
and peaks are the corresponding discrete perturbation centres. The zero-
signal floor for the analytic control is exact analytic assignment, while
non-control centered differences use the V1-precedent roundoff classifier
`32 epsilon max(abs(F_state))/Delta` before normalization. This classifier
depends only on machine precision, the state flux scale, and the declared
step, not on cross-framework results.

The 80-cell finite-difference ladder uses `Delta=0.05`, `0.10`, and `0.20 dex`
for both PG14 profiles, all four targets, and all six pressure centres.
Linearity compares centered derivatives at 0.05 and 0.20 dex with the primary
0.10 dex derivative. Symmetry uses the even residual
`F(+Delta)+F(-Delta)-2F(0)` divided by the primary signed-response scale.

## Frozen tracks and capability boundaries

Track A is restricted to compatible ROBERT and stable-pRT absorbing paths.
For every baseline, plus, and minus composition state ROBERT independently
recomputes the shared-source molecular random-overlap opacity, H2--H2 CIA,
H2--He CIA, and mean molecular weight, then collapses the total optical depth
with the native quadrature weights. The identical state-specific layer tensor
and fixed temperature are supplied to both RT paths. Baseline optical depth is
never reused for a perturbed composition. PICASO has no invented identical-
tensor Track-A path or gate.

Track B has ROBERT, PICASO 4.0, and stable pRT independently recompute native
molecular opacity/mixing, CIA, composition conversion, and mean molecular
weight for every required state. Native cross-framework comparisons are
attribution-only and have no acceptance gate.

Stable pRT's supported native high-level flux interface does not expose a
stable layer optical-depth tensor, so none is fabricated. PICASO native total
`taugas`, its pathological exact-`omega0=0` native thermal probe, and exact-
zero cloud/Rayleigh evidence remain separate from independently labelled
absorbing-formal spectra and vertical diagnostics. The latter are not called
native SH contribution functions.

Stage-4 source contribution functions, Stage-5 signed temperature responses,
and Stage-6 signed composition derivatives are retained and compared but are
three distinct quantities. A composition derivative may change sign and
includes abundance redistribution, overlap, CIA, mean-molecular-weight, and
background-gas coupling.

## Predeclared numerical and resource gates

The matched Track-A definitions, analytic controls, finite-difference
behaviour, and convergence alone are gated. The following V1 Stage-6 values
are carried forward because the derivative definitions, localization grid,
and compatible pure-absorption RT comparison are unchanged; Version 2 changes
the physical state and exact composition but not the numerical meaning of
these diagnostics. A threshold exceedance is a characterized regime, never a
framework failure.

| Diagnostic | Frozen limit |
| --- | ---: |
| Primary Track-A R=100 signed-Jacobian p95 difference / pair peak | `0.05` |
| Primary Track-A eclipse-Jacobian RMS difference | `0.50 ppm/dex` |
| Primary Track-A pressure-centroid RMS difference | `0.15 dex` |
| Primary Track-A full-response p95 total variation | `0.08` |
| Primary Track-A cross-species-fraction p95 total variation | `0.08` |
| Track-A 80-to-160 signed-Jacobian p95 change | `0.05` |
| Track-A 80-to-160 eclipse-Jacobian RMS change | `0.50 ppm/dex` |
| Track-A 80-to-160 pressure-centroid RMS change | `0.15 dex` |
| Track-A 80-to-160 full-response p95 total variation | `0.08` |
| Track-A 80-to-160 cross-species-fraction p95 total variation | `0.08` |
| 0.05/0.10/0.20 dex linearity p95 relative difference | `0.02` |
| Finite-difference symmetry p95 relative even residual | `0.02` |
| Analytic isothermal composition-Jacobian maximum | `0` |
| Exact-zero normalization/fraction maximum | `0` |

The representative pilot is one complete 80-cell PG14 non-inverted target
shard (baseline and six `+/-0.10 dex` localizations) in all three native
frameworks plus the compatible ROBERT/stable-pRT state-dependent Track-A
paths. The projected full workload includes two profiles, four targets, three
resolutions, and both additional linearity amplitudes. The full matrix is
authorized only if projected wall time is at most `7200 s` and the largest
process-tree peak RSS is below 60 per cent of memory available at the
decision. All committed shards must be below `100,000,000 bytes` without
reducing precision or dropping required tensors.

## Preserved earlier-stage framing

Stage 1 retains the exact `0.1968967046 ppm` eight-angle maximum and the
sub-`0.01 ppm` continuous-angle restriction. Stage 2 retains Track-A maxima
`10.150094`, `2.558145`, and `0.640615 ppm` and the `1.369643 ppm` 80-to-160
change. Stage 3 retains `7.110565`, `1.788100`, and `0.447547 ppm` and corrected
both-CIA effects of `43.387801`, `45.501171`, and `43.656935 ppm` for ROBERT,
PICASO, and stable pRT. Stage 4 remains an out-of-tolerance but vertically
converging closure regime. Stage 5 retains all matched Track-A passes, its
`0.168817%` primary Jacobian p95, `0.000786942 ppm/K` eclipse-Jacobian RMS,
`0.001043488 ppm` isothermal control, `0.515848%` native ROBERT/stable-pRT p95,
and attribution-only PICASO-pair values through `4.05137%`.

## Pilot, results, and verification

The repeated production pilot measured `50.896305 s`, projected the complete
matrix at `2443.022646 s`, and observed a largest process-tree member peak RSS
of `4,310,876,160 bytes`, or `41.790554%` of the memory available at the
decision. Both resource gates therefore authorized the matrix. The post-pilot
matrix and finite-difference audit took `7557.956396 s`; this measured value is
reported explicitly even though it exceeded two hours, because the frozen
stop decision was necessarily based on the passing pre-matrix projection.

The completed matrix is an **out-of-tolerance characterized regime** solely
because the matched Track-A finite-difference symmetry p95 is `0.0266245372`,
above the frozen `0.02` limit. No perturbation, opacity, interpolation,
quadrature, CIA, MMW, wavelength handling, or threshold was changed after the
freeze. The matched Track-A linearity p95 is `0.00217799083` and passes.

All ten matched-definition comparison and convergence gates pass:

| Diagnostic | Primary 80-cell | 80-to-160 | Limit |
| --- | ---: | ---: | ---: |
| Signed-Jacobian p95 / pair peak | `0.00326128735` | `0.00215697664` | `0.05` |
| Eclipse-Jacobian RMS | `0.132752669 ppm/dex` | `0.0747846691 ppm/dex` | `0.50 ppm/dex` |
| Pressure-centroid RMS | `0.00197857248 dex` | `0.00894567269 dex` | `0.15 dex` |
| Full-response p95 total variation | `0.00301640877` | `0.0167956416` | `0.08` |
| Cross-species-fraction p95 total variation | `0.000157076853` | `0.0000558248735` | `0.08` |

The isothermal composition Jacobian, its normalized pressure response, and
its own/cross-species fractions are exactly zero. The analytic-zero maximum
and zero-normalization maximum are both `0.0`; no cancellation noise is
promoted to a signal.

Track B remains attribution-only. At 80 cells, native signed-Jacobian p95
differences reach `0.815618%` for ROBERT/stable pRT and `4.071860%` for a pair
involving PICASO. Native pressure-response p95 total variation reaches
`0.225582`, pressure-centroid RMS reaches `0.183334 dex`, and cross-species-
fraction p95 total variation reaches `0.0948919`. These values describe the
combined database, interpolation, correlated-k/mixing, CIA, MMW, and native RT
attribution; they are not cross-framework acceptance tests.

Every perturbed PICASO state records application of the Stage-3 absolute
summed-line-VMR restoration to its actual pressure-dependent H2O+CO+CO2+CH4
sum. The complete six-species state, exact `0.8547:0.1453` H2/He remainder,
MMW, and both CIA pairs were independently checked for every state. PICASO
native `taugas` is retained without relabelling absorbing-formal diagnostics
as native SH contributions. Stable pRT's absent supported native layer-tau
interface remains absent rather than being fabricated.

The strict Stage-6 checksum index covers `767` files: `766` data/report shards
plus the integrity manifest. Their aggregate size is `15,278,571,444 bytes`
including the manifest, and the largest individual product is `50,898,616
bytes`, safely below the `100,000,000-byte` object limit. Full precision and
all required native tensors were retained. The optional-Vega and exact-zero
cloud/Rayleigh warnings were recorded without downloads, suppression, or
epsilon substitution.

The canonical machine-readable results are
`docs/data/emission_intercomparison/version_2/stage_6_report.json`, with
per-file units, axes, sizes, and SHA-256 digests in
`docs/data/emission_intercomparison/version_2/stage_6_integrity.json` and the
Version-2 `checksums.json`. Stage-4 contribution functions, Stage-5 signed
temperature responses, and these Stage-6 signed composition derivatives are
compared as distinct diagnostics, never as interchangeable quantities.

Final verification used the exact isolated interpreters. The focused Stage-6
suite passed `45` tests; the Stage-1 preservation plus Stage-6 strict checksum
rerun passed `50` tests after restoring the immutable Stage-1 lineage file
byte-for-byte; the broader Version-2 focused sweep's other `123` tests passed.
Repository-wide Ruff passed. The complete suite passed `567` tests with `5`
declared skips in `311.02 s`. The ROBERT `0.3.0`, PICASO `4.0` 661-bin resort-
rebin correlated-k, and stable-pRT `3.3.3` environment smokes all returned
finite positive thermal output. The PICASO smoke retained the harmless
optional-Vega warning and did not invoke opacity sampling.
