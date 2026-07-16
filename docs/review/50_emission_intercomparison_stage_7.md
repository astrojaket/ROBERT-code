# Emission intercomparison: Stage 7

## Predeclared cloud contract and acceptance gates

This section was written before the representative pilot and complete Stage-7
matrix were run.  It fixes the physical contract, diagnostic definitions, and
Track-A acceptance limits independently of the observed intercomparison.

Stage 7 is an absorbing-cloud placement/extinction comparison.  Every cloud
has exactly `omega0=0`; ROBERT receives no scattering component, PICASO has
Rayleigh and cloud scattering explicitly disabled, and petitRADTRANS uses only
an additional absorption-opacity callback with `scattering_in_emission=False`.
Delta-M scaling is off and solver-order comparisons are deferred to Stage 8.

The parametric matrix is the full Cartesian product of reference column
optical depths `0.1`, `1`, `10`, and `100`; cloud tops `1`, `10`, and
`100 mbar`; and extinction slopes `-4`, `-2`, `0`, and `+2`.  The reference
wavelength is frozen at `5 micron`.  At that wavelength, extinction is
distributed uniformly in `d tau / d log(P)` below the cloud top, using the
existing ROBERT fractional log-pressure overlap convention for the
intersected boundary layer.  The spectral law is

`tau(lambda)=tau(5 micron) [lambda/(5 micron)]^slope`.

The physical tabulated case is the archived PICASO/Virga extinction field in
`data/validation/end_to_end_cloud_parity/picaso_virga_independent_output.npz`,
paired with its versioned physical contract.  Pressure remapping conserves
layer optical depth assuming uniform `d tau / d log(P)` inside each archived
layer.  Positive extinction is interpolated log-linearly in wavelength, with
constant endpoint extension outside the archived `1--12 micron` interval.
Source and derived checksums are recorded.

The 40/80/160 contract remains unchanged: ROBERT uses pressure cells, PICASO
receives the matching pressure edges as levels, and pRT receives the ROBERT
geometric cell centres as nodes.  The 80-cell matrix is primary.  The four
Stage-4 thermal structures are retained.  Isothermal spectra with an
isothermal blackbody lower boundary have an analytically zero cloud signal;
their cloudy-minus-clear effects are set exactly to zero before normalization,
while raw cancellation maxima remain recorded.

Track A freezes ROBERT source-HDF molecular+CIA optical depth and the complete
pressure-by-wavelength cloud extinction array, then supplies both identically
to all three pure-absorption RT paths.  Track B supplies the high-level case
through native interfaces: ROBERT `CloudOpticalProperties`, PICASO's
dimensionless layer-`opd` cloud table and native interpolation, and pRT's
`cm2/g` additional-absorption callback on pressure nodes.  Track-B differences
in native gas opacity, placement, interpolation, wavelength sampling, opacity
units, and boundary treatment are attribution results and have no
cross-framework pass/fail gate.

Track A passes only if every fixed limit below is met.  Primary limits span
all profiles, cloud cases, and code pairs.  The 80-to-160 limits span all
profiles, cloud cases, and frameworks.

| Diagnostic | Primary limit | 80-to-160 limit |
| --- | ---: | ---: |
| R=100 absolute-spectrum p95 symmetric relative difference | `0.01` | `0.01` |
| Signed cloud-effect p95 absolute difference / pair peak | `0.02` | `0.02` |
| Cloud-effect eclipse-depth RMS difference | `0.50 ppm` | `0.50 ppm` |
| Contribution pressure-centroid RMS difference | `0.05 dex` | `0.05 dex` |
| Full contribution-profile p95 total variation | `0.05` | `0.05` |
| Full cloud-response-profile p95 total variation | `0.08` | `0.08` |

The exact-zero isothermal cloud effect has an additional `1e-10 ppm` maximum
gate after analytic handling, and the maximum absolute `omega0` gate is
exactly zero.

The complete artifact retains signed R=100 spectra and eclipse depths,
cloudy-minus-clear effects, cloud extinction, normalized contribution and
cloud-response profiles, pressure coordinates, and all case axes at every
resolution.  The report also retains band/window summaries, native-resolution
extrema, raw and summarized timings, process peak RSS where measurable,
interpreter paths and versions, warnings, contracts, method definitions, and
source/data/output checksums.  Stage-7 pressure diagnostics are compared with
Stage-4 contributions and Stage-5/6 temperature/composition responses on the
six earlier localization centres, explicitly as related diagnostics rather
than mathematical identities.

Before the full matrix, the launcher runs all frameworks and both tracks at
80 cells for every thermal profile and five representative cloud definitions.
It conservatively projects complete wall time by the full/pilot case ratio and
`(40+80+160)/80`.  The full run proceeds only if the projection is at most
`7200 s` and the largest measured process-tree peak RSS is no more than 60 per
cent of then-available memory.  Otherwise the launcher preserves the pilot and
stops.

## Results

The representative pilot authorized the full local run.  It measured a
largest process/orchestrator peak RSS of `5.04 GB` with `9.91 GB` available at
the decision point and projected `1435 s` for the complete solver workload,
below the `7200 s` ceiling.  The complete run plus its repeated safety pilot
took `3783.7 s` (`63.1 min`); the full-matrix portion was `3666.4 s`.  The
projection underestimated the complete tensor compaction and JSON assembly,
but the actual run remained laptop-safe and below the two-hour target.  Peak
orchestrator RSS in the complete run was `5.57 GB`.

Track A did **not** pass the predeclared full-domain gates.  The failures are
retained rather than relaxing limits after seeing the matrix:

| Diagnostic | Observed | Limit | Result |
| --- | ---: | ---: | --- |
| Primary absolute-spectrum p95 symmetric relative difference | `0.3085` | `0.01` | fail |
| Primary cloud-effect p95 difference / pair peak | `0.1575` | `0.02` | fail |
| Primary cloud-effect eclipse RMS | `3.47 ppm` | `0.50 ppm` | fail |
| Primary contribution centroid RMS | `0 dex` | `0.05 dex` | pass |
| Primary contribution-profile p95 TV | `0` | `0.05` | pass |
| Primary cloud-response p95 TV | `0` | `0.08` | pass |
| 80-to-160 absolute-spectrum p95 symmetric relative difference | `0.2703` | `0.01` | fail |
| 80-to-160 cloud-effect p95 difference / pair peak | `0.0883` | `0.02` | fail |
| 80-to-160 cloud-effect eclipse RMS | `3.60 ppm` | `0.50 ppm` | fail |
| 80-to-160 contribution centroid RMS | `0.0161 dex` | `0.05 dex` | pass |
| 80-to-160 contribution-profile p95 TV | `0.5000` | `0.05` | fail |
| 80-to-160 cloud-response p95 TV | `0.7080` | `0.08` | fail |
| Isothermal cloud-effect maximum | `0 ppm` | `1e-10 ppm` | pass |
| Maximum absolute `omega0` | `0` | `0` | pass |

The worst primary absolute difference is the retrieved-like,
`tau(5 micron)=100`, `1 mbar`, slope `-4` case.  The worst primary signed
cloud-effect result is the inverted, `tau=100`, `10 mbar`, slope `-4` case.
The same extreme short-wavelength opacity and high cloud placement dominate
absolute 80-to-160 convergence.  Optically thick source localization within
one coarse cell also produces large total-variation convergence even while
the contribution centroid converges to `0.0161 dex`; retaining both metrics
prevents a stable centroid from hiding unresolved profile shape.

The failure is not universal across the useful cloud domain.  For the
monotonic `tau=1`, `10 mbar`, grey case at 80 cells, shared ROBERT and pRT have
`0.203%` absolute-spectrum p95 difference, `0.155%` cloud-effect p95 difference
over pair peak, and only `0.00848 ppm` eclipse-effect RMS difference.  ROBERT
and the corrected isolated PICASO exact-absorption path agree to about
`3.2e-16` in that case.  For the retrieved-like archived Virga/Mie extinction
case, shared ROBERT/pRT reach `1.72%` absolute p95 but only `0.0391%` in the
cloud-effect metric and `0.0134 ppm` eclipse-effect RMS.  The full Cartesian
matrix therefore exposes a restricted extreme-domain discretization failure,
not a blanket failure of moderate absorbing clouds.

One analysis-path defect was found without changing any physical contract or
gate.  PICASO's low-level `get_thermal_1d` returned all NaNs when passed exact
`omega0=0`.  Its Stage-7 shared worker was replaced with the exact absorbing
formal path already validated in Stage 1 and used for PICASO contributions in
Stages 4--6.  All three corrected shared-PICASO artifacts are finite and keep
`omega0=0`, Rayleigh off, cloud scattering off, and delta-M off.  The complete
preserved worker matrix was then reanalysed in `88.3 s`; the original solver
wall time remains the reported full-matrix time.  The Track-A gates above were
unchanged and still fail.

Track B remains attribution-only.  For the moderate monotonic grey case,
native ROBERT/pRT have `1.99%` absolute spectral p95, `1.62%` cloud-effect p95,
and `0.473 ppm` effect RMS.  For the archived physical case they have `2.50%`,
`0.242%`, and `0.416 ppm`, respectively, while the pRT node-opacity placement
diagnostic differs from the ROBERT/PICASO conservative layer placement by
about `1.01 dex` in centroid RMS.  Pairs containing PICASO retain the
independent-opacity attribution from Stages 3--6: even the moderate grey case
reaches about `68%` absolute spectral p95 and `3.16 ppm` cloud-effect RMS for
ROBERT/PICASO.  These values are reported, not gated.

Band/window records preserve the signed effects.  In the monotonic moderate
grey Track-A case, the CO2 `4.1--4.5 micron` band has a mean effect of about
`-1.51 ppm`, an RMS of `3.45 ppm`, and a minimum near `-9.76 ppm`.  The archived
physical case gives about `-3.14 ppm` mean, `6.97 ppm` RMS, and `-19.1 ppm`
minimum in the same band.  Native Track B changes those values because each
framework constructs gas and cloud opacity independently; all native-grid
extrema remain in the JSON report.

The Stage-7 cloud-response profiles were projected onto the six Stage-5/6
pressure centres for explicit comparison with earlier diagnostics.  Median
p95 total-variation distances are about `0.745` versus Stage-4 projected
contribution, `0.777` versus Stage-5 temperature response, and `0.724` versus
the mean Stage-6 composition response over the selected grey and archived
cases.  The large distances are physically informative: cloud-induced source
redistribution is related to, but is not identical with, contribution,
temperature, or composition response.

The versioned outputs are
`docs/data/emission_intercomparison/stage_7_report.json` and
`docs/data/emission_intercomparison/stage_7_absorbing_cloud_arrays.npz`.
The latter preserves complete R=100 spectra, eclipse depths, signed effects,
cloud extinction, contributions, and cloud responses.  Normalized profiles
use documented recoverable packed-uint12 storage with maximum quantization
error `1.22e-4`; this keeps the `70 MB` artifact below GitHub's file limit.
Raw process-isolated native-grid artifacts remain ignored under
`examples/outputs/emission_intercomparison/stage_7/` and occupy about
`7.3 GB`.
