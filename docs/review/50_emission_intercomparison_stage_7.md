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

Pending the predeclared pilot and resource decision.
