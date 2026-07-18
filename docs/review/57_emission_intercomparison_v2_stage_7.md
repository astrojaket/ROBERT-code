# Emission intercomparison Version 2: Stage 7

> **Pre-pilot and pre-matrix freeze:** The scientific state, absorbing-cloud
> definitions, placement/interpolation conventions, diagnostics, numerical
> gates, resource projection, and capability boundaries in the first section
> were fixed before the representative pilot or complete result matrix was
> inspected. Measured results are appended without changing them.

## Frozen scientific scope

Stage 7 isolates absorbing-cloud placement and extinction before any cloud
scattering comparison. Every cloud has exact single-scattering albedo
`omega0=0`; Rayleigh, cloud scattering, every other scattering term, delta-M,
and scattering source terms are disabled. No zero is replaced by an epsilon,
and numerical warnings are retained.

The gas state is the completed Version-2 Stage-4/5/6 baseline: exact frozen
H2O, CO, CO2, CH4, H2, and He VMRs, the exact `0.8547:0.1453` H2/He remainder,
the declared VMR-weighted molecular masses/MMW, and both H2--H2 and H2--He
CIA. Temperature is fixed within every clear/cloud comparison. The physical
matrix uses both exact evaluated PG14 profiles; the exact 1755 K isothermal
profile is an analytic cloud-effect control. All use 40/80/160 pressure cells,
80 primary, the frozen WASP-17 system and 6550 K blackbody, signed outward
flux/eclipse convention, and the exact 0.3--12 micron, 369-bin,
flux-conserving equal-log R=100 product.

PICASO opacity sampling remains retired and is neither run nor reintroduced.
PICASO 4.0 resort-rebin correlated-k is its sole molecular path, including the
Stage-3 restoration of the absolute summed line-gas VMR.

## Frozen cloud contract

The parametric matrix is the complete Cartesian product of column optical
depths `0.1`, `1`, `10`, and `100` at `5 micron`; continuous cloud tops at
`1`, `10`, and `100 mbar`; and slopes `-4`, `-2`, `0`, and `+2`. One clear
case and the archived PICASO/Virga physical extinction field give 50 cloud
definitions per profile. The tabulated field contributes extinction only;
its archived wavelength-dependent scattering properties are outside Stage 7
and do not override exact `omega0=0`.

The cloud bottom is the exact grid bottom, `100 bar`. Parametric extinction is
uniform in `d tau / d log(P)` between the continuous top and bottom. The cell
intersected by the top receives its exact fractional log-pressure overlap; no
top is snapped to a cell edge or centre. At wavelength `lambda`,

`tau(lambda) = tau(5 micron) [lambda/(5 micron)]^slope`.

Thus a negative slope is stronger at short wavelengths. The reference column
optical depth is recovered by summing all fractional cells at 5 micron.

The archived field is pressure-remapped by exact source/target overlap under
uniform within-source-cell `d tau / d log(P)`, conserving column optical
depth. Positive extinction is interpolated log-linearly in wavelength and
held constant outside its archived 1--12 micron range. Exact zeros are
retained without a numerical floor. Cells outside the archived pressure range
remain exactly zero.

ROBERT evaluates cloud extinction on its true native molecular wavelength
grid. PICASO receives dimensionless cell optical depth through its supported
cloud-table interface and performs native spectral mapping. Stable pRT
receives a `cm2/g` additional-absorption callback evaluated on its native
pressure nodes. This callback is a declared input parameterization, not a
fabricated stable-pRT native layer-tau tensor. R=100 spectra use the common
flux-conserving binning contract. A correlated-k lower-edge closure is used
only for bin integration and is never labelled a native wavelength or datum.

## Frozen tracks and capability boundaries

Track A is limited to genuinely compatible ROBERT and stable-pRT pure-
absorption paths. Both consume the same ROBERT-derived mean molecular plus
both-CIA layer optical depth and the identical pressure-by-wavelength cloud
extinction tensor. PICASO has no invented identical-tensor gate.

Track B supplies the same high-level physical definitions through ROBERT
layer extinction, PICASO native cloud tables, and stable-pRT native additional
absorption callbacks. Gas database, interpolation, correlated-k, native cloud
mapping, pressure-node/cell semantics, and RT effects are attribution only;
there is no native cross-framework gate.

PICASO native `taugas`, `taucld`, and its exact-`omega0=0` thermal output are
retained as separate capability evidence. Scientific PICASO spectra and
vertical diagnostics use an independently labelled absorbing formal solution
applied to those native tensors; they are not called native SH contribution
functions. Stable pRT retains its supported flux and native emission-
contribution output, but its supported high-level interface exposes no stable
native layer optical-depth tensor, so none is fabricated.

Source contribution, temperature response, composition response, cloud-
induced contribution redistribution, and cloud placement/extinction are
distinct quantities. Stage 7 does not relabel one as another.

## Frozen diagnostics and numerical gates

The signed cloud effect is cloudy minus clear within the same fixed
temperature profile. Isothermal blackbody cloud effects are assigned exact
zero before normalization, while raw cancellation is retained. The normalized
vertical cloud effect is the absolute cloudy/clear normalized-contribution
difference, renormalized over pressure only where its exact sum is non-zero.
Pressure centroids weight `log10(P/bar)`; peaks remain discrete pressure-cell
coordinates. Pressure increases downward in plots.

Spectral convergence compares flux-conserving R=100 products on the exact
endpoints. Vertical convergence sums nested fine-cell probability into the
matching coarse cells before centroid and total-variation metrics. The seven
Stage-4 optical-to-mid-IR band/windows remain the frozen band diagnostics.

Only matched Track-A quantities, analytic zero controls, and vertical
convergence are gated. Track B is attribution-only.

| Diagnostic | Primary limit | 80-to-160 limit |
| --- | ---: | ---: |
| R=100 absolute-spectrum p95 symmetric relative difference | `0.01` | `0.01` |
| Signed cloud-effect p95 absolute difference / pair peak | `0.02` | `0.02` |
| Cloud-effect eclipse-depth RMS difference | `0.50 ppm` | `0.50 ppm` |
| Contribution pressure-centroid RMS difference | `0.05 dex` | `0.05 dex` |
| Full contribution-profile p95 total variation | `0.05` | `0.05` |
| Full cloud-effect-profile p95 total variation | `0.08` | `0.08` |

The isothermal cloud-effect limit is `1e-10 ppm`; maximum absolute `omega0`
must equal exactly zero.

## Frozen pilot and data policy

The mandatory pilot is the 80-cell PG14 non-inverted clear/moderate pair with
`tau(5 micron)=1`, `10 mbar`, and slope 0 in Track B for all three frameworks
and Track A for ROBERT/stable pRT. Cold and warm wall time, process-tree peak
RSS, available memory, native wavelength counts, and retained tensor sizes are
recorded.

For every framework/track, the projection uses the cold-minus-warm setup cost
plus warm per-case time for three profiles (including the analytic control),
50 clouds, and three resolutions. Their sum is multiplied by a frozen `1.20`
plot/checksum/report-assembly overhead. The full matrix is authorized only if
the projected total is at most `7200 s` and peak process-tree RSS is no more
than 60 per cent of available memory.

Raw workers, full-precision spectra and tensors, the complete JSON report, and
the local SHA-256 integrity index default under the ignored
`examples/outputs/emission_intercomparison/version_2/stage_7/` tree. None is
committed or uploaded to GitHub. Reproducible code, tests, compact contracts,
this review, and a small human-readable summary are the only ordinary-Git
products; full numerical products remain local for a later Zenodo or
equivalent archival release.

## Pilot measurements and scientific results

The mandatory pilot ran the frozen 80-cell PG14 non-inverted clear/moderate
pair before any complete result matrix was inspected. Measured cold/warm wall
times were:

| Framework/track | Cold (s) | Warm (s) | Native wavelengths |
| --- | ---: | ---: | ---: |
| Track A ROBERT | `0.150033792` | `0.144567458` | `3696` |
| Track A stable pRT | `1.577177750` | `1.574056958` | `3696` |
| Track B ROBERT | `7.354452750` | `4.301428084` | `3696` |
| Track B PICASO | `19.837954417` | `19.256825333` | `661` |
| Track B stable pRT | `5.389816875` | `4.770195125` | `3697` |

The pilot itself took `68.146504000 s`. Peak process-tree RSS was
`4,272,816,128 bytes`; available memory at the decision was
`9,425,321,984 bytes`, so the measured fraction was `0.4533337042`
(`45.333370%`) and passed the frozen 60% limit. The projected worker time was
`6773.378503447 s`; applying the predeclared 1.20 assembly factor gave
`8128.054204136 s` (`2.258 h`). That exceeds the frozen `7200 s` limit.
Consequently `continue_full_matrix=false`. The user subsequently authorized
exceeding the wall-time projection. That authorization did not change any
scientific definition, numerical gate, precision, required case, or capability
boundary.

Retained tensor sizes include the Track-B ROBERT molecular tensor
`[1,80,3696,16]` (`18,923,520 bytes`), the two-case ROBERT cloud tensor
`[50,80,3696]` (`59,136,000 bytes`), PICASO gas
`[1,80,661,8]` (`1,692,160 bytes`), and PICASO two-case cloud
`[2,80,661,8]` (`3,384,320 bytes`). Stable pRT retains its supported vertical
contribution array but, as frozen, no fabricated native layer-tau tensor.

The representative cloud column sums to exactly `1.0` at 5 micron. Applicable
Track-A ROBERT/stable-pRT diagnostics all pass their frozen limits: absolute-
spectrum p95 symmetric relative difference is `0.00324979584`, signed cloud-
effect p95 over the pair peak is `0.00232598354`, eclipse-effect RMS difference
is `0.0438057716 ppm`, contribution-centroid RMS is exactly `0 dex`, and both
retained vertical total-variation p95 values are exactly zero. These are
representative-pilot results only, not complete-matrix acceptance.

Native Track B remains attribution-only. Its ROBERT/stable-pRT absolute-
spectrum p95 and eclipse-effect RMS differences are `0.00419148189` and
`0.223026176 ppm`; the ROBERT/PICASO values are `0.0684449041` and
`2.864554515 ppm`. The PICASO exact-`omega0=0` native thermal probe retained
`1322/1322` finite values, but remains capability evidence separate from the
independently labelled absorbing-formal scientific spectrum. Maximum PICASO
Rayleigh optical depth and maximum input single-scattering albedo are both
exactly zero. Optional-Vega, exact-zero, and stable-pRT scattering-advice
warnings were retained rather than hidden.

## Authorized complete execution and outcome

A repeated pilot inside the authorized launcher benefited from warmed local
caches but preserved the same contract. It took `42.664948500 s`, reached
`4,110,172,160 bytes` peak process-tree RSS with `9,376,956,416 bytes`
available, and projected `4906.158646882 s`; this second measurement passed
both frozen resource limits. Its cold/warm times were
`0.142425459/0.143698125 s` for Track-A ROBERT,
`1.575786625/1.587040167 s` for Track-A stable pRT,
`6.934487417/4.421375958 s` for Track-B ROBERT,
`7.344780625/7.386536750 s` for Track-B PICASO, and
`4.768925292/4.596499792 s` for Track-B stable pRT. The original failed pilot
is not relabelled or discarded.

The full launcher completed in `2371.303373625 s` (`0.658695 h`):
`2328.638425125 s` followed the repeated pilot for all matrix workers and
assembly. Complete-matrix peak process-tree RSS was `5,195,988,992 bytes`, or
`55.412319%` of the memory available at the repeated-pilot decision. The local
product tree contains 92 integrity-indexed artifacts plus its integrity index,
totalling `3,628,261,316 bytes`; the largest local NPZ is `688,826,304 bytes`.

The final status is **out-of-tolerance characterized regime**. The analytic
isothermal cloud effect and maximum input single-scattering albedo are exactly
zero. Primary contribution-centroid RMS, contribution-profile TV p95, and
cloud-response TV p95 are all exactly zero. The 80-to-160 contribution-centroid
RMS and contribution-profile TV p95 pass at `0.00456569396 dex` and
`0.0392107564`.

Seven frozen gates fail:

| Diagnostic | Observed | Limit |
| --- | ---: | ---: |
| Primary absolute-spectrum p95 | `0.327970825` | `0.01` |
| Primary cloud-effect p95 / pair peak | `0.0517204233` | `0.02` |
| Primary cloud-effect eclipse RMS | `6.21989875 ppm` | `0.50 ppm` |
| 80-to-160 absolute-spectrum p95 | `0.288476674` | `0.01` |
| 80-to-160 cloud-effect p95 / pair peak | `0.0540403749` | `0.02` |
| 80-to-160 cloud-effect eclipse RMS | `6.33828924 ppm` | `0.50 ppm` |
| 80-to-160 cloud-response TV p95 | `0.644274045` | `0.08` |

The largest primary absolute-spectrum p95 occurs for the PG14 inverted,
`tau=100`, 100-mbar-top, slope `-4` deck. The largest primary cloud-effect RMS
occurs for the PG14 inverted, `tau=100`, 1-mbar-top, slope `-4` deck. This
confirms that the moderate pilot is an accepted local regime but does not
validate the extreme high-tau/placement matrix. Track B remains attribution-
only and no framework is classified as failed.

All workers, full-precision arrays, the complete report, and the SHA-256
integrity index remain ignored locally at
`examples/outputs/emission_intercomparison/version_2/stage_7/` for later
Zenodo or equivalent archival release. The committed compact summary is
`docs/data/emission_intercomparison/version_2/stage_7_summary.json`.
