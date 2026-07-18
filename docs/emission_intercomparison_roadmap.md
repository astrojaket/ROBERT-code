# Emission intercomparison roadmap

> **Version notice:** this file records the completed Version-1 results and the
> original Stage-8/9 outline.  New runs are governed by
> [Emission intercomparison Version 2](emission_intercomparison_version_2.md),
> which freezes the WASP-17 system, solar-like VMRs, PG14 profiles, blackbody
> star, rerun plan, and common task preamble.  Version-1 artifacts remain
> historical and must not be overwritten.

This document is the canonical stage numbering for the ROBERT, PICASO, and
petitRADTRANS thermal-emission intercomparison and its companion paper.  It
supersedes the earlier seven-stage draft: temperature and composition
Jacobians and absorbing clouds are now explicit completed stages, so cloud
scattering and retrieval work moves to Stages 8--9.

## Completed stages

1. **Grey/isothermal closure.** Analytic and shared-optical-depth checks of
   units, boundaries, angular integration, and pure-absorption RT.
2. **Single-molecule closure.** H2O, CO, CO2, and CH4 temperature/abundance
   matrices with shared optical depth and explicit vertical convergence.
3. **Multi-species and CIA.** Four-molecule H2/He atmospheres with H2--H2 and
   H2--He CIA, separating shared-RT closure from native-opacity attribution.
4. **Thermal structures and contribution functions.** Isothermal, monotonic,
   inverted, and retrieved-like profiles with complete pressure-resolved
   diagnostics.
5. **Temperature Jacobians.** Localized symmetric thermal perturbations,
   shared- and native-opacity tracks, and complete vertical responses.
6. **Composition Jacobians.** Localized H2O, CO, CO2, and CH4 log-VMR
   perturbations, composition normalization, CIA/MMW coupling, cross-species
   fractions, and finite-difference linearity tests.
7. **Absorbing clouds.** Grey/power-law decks and archived Virga/Mie
   extinction with `omega0=0`, shared- and native-cloud tracks, complete
   vertical diagnostics, and 40/80/160 convergence.  Moderate shared cases
   close, but the predeclared full-domain Track-A gates fail for extreme
   high-altitude, optically thick, steep-slope clouds and their vertical
   convergence; the failure remains a Stage-8 input constraint.

The versioned Stage 1--7 reports under
`docs/data/emission_intercomparison/` remain the source of numerical results.

## Version-2 Stage 1: WASP-17b common contract and grey closure

Version-2 Stage 1 is complete under
`docs/data/emission_intercomparison/version_2/`.  It freezes one typed common
contract for the WASP-17 system, 6550 K blackbody star, 40/80/160 grids, exact
six-species mixture, PG14 arrays, and flux-conserving 0.3--12 micron R=100
product.  The 80-cell pilot projected `44.60 s` and measured `0.674 GB` peak
RSS, authorizing the full laptop run; total solver wall time was `20.91 s`.

Eight of nine Stage-1 gates pass.  The continuous-angle analytic eclipse gate
remains failed (`0.196897 ppm` versus `0.01 ppm`) even though the separately
frozen angular relative gate passes.  The failure is a declared eight-angle
representation limit, not a reason to tune the quadrature or threshold after
inspection.  The compatible ROBERT/pRT shared-tensor subset and vertical
convergence pass.  Stage 2 may use the accepted common contract, but claims
requiring sub-`0.01 ppm` continuous-angle closure must not treat the eight-angle
Stage-1 product as validated.

## Version-2 Stage 2: frozen single-molecule closure

Version-2 Stage 2 is complete for H2O, CO, CO2, and CH4 at the exact common-
contract VMRs and H2/He fill rule.  The 80-cell pilot authorized the matrix at
`1052.01 s` projected wall time and `35.58%` of available memory; the post-pilot
matrix took `167.72 s`.

The matched ROBERT/pRT Track-A difference decreases by about fourfold per grid
doubling but exceeds the frozen full-matrix and 80-to-160 limits, defining an
out-of-tolerance vertical closure regime.  The 160-cell maximum is
`0.640615 ppm`; the isothermal controls remain below `0.001044 ppm`. Track B
records native database, interpolation, and representation effects without
classifying a framework as failed. PICASO resort-rebin correlated-k is the
only active PICASO molecular path; opacity sampling is retired.

## Version-2 Stage 3: frozen multi-species and CIA closure

Version-2 Stage 3 is complete for the exact frozen solar-derived fixed-abundance
H2O, CO, CO2, and CH4 mixture with the serialized H2/He background and mean
molecular weight.
The four line absorbers remain active throughout a `2 x 2` H2--H2/H2--He CIA
factorial crossed with the frozen isothermal and PG14 non-inverted profiles on
40/80/160 cells.  The representative 80-cell both-CIA pilot measured
`12.22 s`, projected `439.77 s`, and reached `34.18%` of available memory, so
the complete matrix was authorized; its post-pilot wall time was `166.94 s`.

The matched ROBERT/stable-pRT Track-A maximum decreases from `7.110565 ppm` at
40 cells to `1.788100 ppm` at 80 and `0.447547 ppm` at 160. It therefore
preserves an out-of-tolerance, vertically converging regime without classifying
either framework as failed. The isothermal controls remain below
`0.001044 ppm`. Track B attributes native line/CIA database, interpolation,
and representation effects without cross-framework gates: PICASO resort-rebin
correlated-k is the only active PICASO molecular path. PICASO's exact-zero native RT
and vertical-diagnostic boundaries and stable pRT's missing supported native
optical-depth tensor remain explicit.  Stage 1's `0.196897 ppm` eight-angle
result and sub-`0.01 ppm` restriction, and Stage 2's measured vertical regime,
are unchanged.

## Version-2 Stage 4: frozen thermal structures and contribution functions

Version-2 Stage 4 is complete for the exact `1755 K` isothermal, PG14
non-inverted, and PG14 inverted common-contract arrays on 40/80/160 cells.
Every case fixes H2O, CO, CO2, and CH4 at their exact reference VMRs with both
H2--H2 and H2--He CIA enabled. PICASO 4.0 resort-rebin correlated-k remains its
only active molecular path.

The compatible ROBERT/stable-pRT shared-mean-tau spectra preserve an
out-of-tolerance, vertically converging closure regime: the maximum eclipse
difference decreases from `7.110565 ppm` at 40 cells to `1.788100 ppm` at 80
and `0.447547 ppm` at 160. The largest shared 80-to-160 change is
`1.206041 ppm`, while the isothermal controls remain below `0.001044 ppm` and
the matched vertical diagnostic closes exactly to stored precision.

Native database/interpolation/correlated-k/RT differences remain attribution
only. At 80 cells the largest native ROBERT/stable-pRT difference is
`9.873270 ppm`, the largest pair involving PICASO is `125.715939 ppm`, and the
largest native pressure-centroid RMS difference is `0.129370 dex`. Complete
native/R=100 spectra, eclipse depths, pressure-resolved arrays, centroids,
peaks, band/window diagnostics, optical-depth tensors where supported, and
40/80/160 convergence are retained. Capability boundaries for PICASO and
stable pRT remain explicit rather than populated with invented products.

## Version-2 Stage 5: localized temperature responses and Jacobians

Version-2 Stage 5 is complete for the exact three Stage-4 profiles, fixed
four-molecule H2/He mixture, and both CIA pairs. Six `0.35 dex` Gaussian
localizations from `1e-4` to `10 bar` use symmetric `+/-10 K` perturbations;
the 80-cell PG14 non-inverted case also retains a predeclared `+/-5/10/20 K`
finite-difference audit.

The compatible ROBERT/stable-pRT frozen-opacity/source-response Track A passes
all predeclared gates. Its primary Jacobian p95 difference is `0.168817%`, and
the 80-to-160 value is `0.151578%`. Primary and converged eclipse-Jacobian RMS,
response-centroid, response-profile, isothermal analytic-control, linearity,
symmetry, and exact-zero normalization gates also pass.

Track B recomputes each framework's native line and CIA opacity for every
required temperature state and remains attribution-only. Primary ROBERT/
stable-pRT Jacobian p95 differences reach `0.515848%`; pairs involving PICASO
4.0 resort-rebin correlated-k reach `4.05137%`. Complete state spectra,
temperature responses/Jacobians, pressure diagnostics, band/window summaries,
Stage-4 contribution relations, supported optical-depth tensors, and
40/80/160 convergence are retained. PICASO and stable-pRT capability
boundaries remain explicit, and opacity sampling remains retired.

## Version-2 Stage 6: localized composition responses and Jacobians

Version-2 Stage 6 is complete for the two PG14 profiles and 40/80/160 grids.
Localized `+/-0.10 dex` H2O, CO, CO2, and CH4 perturbations use the frozen
six-centre, `0.35 dex` design; the exact Version-2 H2/He remainder, MMW, and
both CIA pairs are recomputed for every composition state. A complete
`0.05/0.10/0.20 dex` audit is retained at 80 cells.

All primary and 80-to-160 matched Track-A comparison/convergence gates pass.
The primary signed-Jacobian p95 difference is `0.326129%`, and the converged
value is `0.215698%`. The finite-difference linearity p95 is `0.00217799` and
passes, but symmetry is `0.0266245` against the frozen `0.02` limit. The stage
is therefore an out-of-tolerance characterized regime, without changing the
frozen design or classifying a framework as failed. Analytic isothermal and
zero-signal controls remain exactly zero.

Native Track B remains ungated attribution. It reaches `0.815618%` for the
primary ROBERT/stable-pRT Jacobian comparison and `4.071860%` for pairs
involving PICASO. Complete state spectra, composition Jacobians and responses,
pressure diagnostics, own/cross-species fractions, band/window summaries,
supported opacity/tau arrays, and convergence are checksum-indexed. PICASO
uses resort-rebin correlated-k only and verifies the state-dependent absolute
summed-line-VMR correction on every perturbation.

## Stage 7: absorbing clouds (completed; full-domain gates not accepted)

Stage 7 establishes cloud placement and extinction before scattering closures
are allowed to differ.  Every cloud has single-scattering albedo `omega0=0`.

The primary matrix contains:

- grey cloud decks with column optical depths `0.1`, `1`, `10`, and `100`;
- cloud-top pressures of `1`, `10`, and `100 mbar`;
- wavelength-dependent extinction slopes `-4`, `-2`, `0`, and `+2`;
- a shared tabulated extinction field from an archived physical cloud case;
- the established 40/80/160 vertical grids, with 80 cells primary.

Track A supplies identical pressure- and wavelength-dependent cloud extinction
to every RT path.  Track B uses each framework's native cloud parameterization
and treats implementation/provenance differences as attribution results.  The
stage reports spectra, eclipse depths, cloud contribution/response pressure
metrics, vertical and spectral convergence, and timings.  Explicit Track-A
gates are fixed before the full matrix runs.

The frozen reference wavelength is `5 micron`.  Parametric decks use the
established fractional-boundary, uniform-`d tau/d log(P)` placement below the
cloud top.  The tabulated case is the versioned PICASO/Virga extinction field
from the end-to-end cloud parity benchmark.  Stage 7 begins with a measured
80-cell cross-framework pilot and proceeds only when its conservative complete
matrix projection is at most two hours with comfortable memory margin.  The
predeclared methods and numerical gates are recorded in
`docs/review/50_emission_intercomparison_stage_7.md`.

The complete run remained laptop-safe, but the frozen Track-A limits were not
met across the full extreme matrix.  Stage 8 must therefore treat the accepted
moderate absorbing-cloud subset separately from the unresolved extreme
placement/discretization domain rather than assuming all Stage-7 cases are a
validated scattering baseline.

## Stage 8: cloud scattering and solver order

Stage 8 builds on the accepted Stage-7 extinction contracts in this order:

1. `omega0=0` regression to Stage 7;
2. isotropic scattering with `omega0=0.5`, `0.9`, and `0.99`;
3. anisotropic cases with asymmetry parameter `g=0.3`, `0.6`, and `0.9`;
4. wavelength-dependent `omega0(lambda)` and `g(lambda)`;
5. shared physical Mie phase moments and native microphysical clouds.

Each ladder spans cloud optical depths `0.1`, `1`, `10`, and `100`.  The
comparison includes absorption-only and single-scattering diagnostics,
Toon/two-stream, SH4/P3, and an independent high-order reference envelope such
as a 16--32+ stream, adding/doubling, matrix-operator, or Monte Carlo solution.
Delta-M scaling and phase-moment truncation are contract fields.

Both disk-integrated spectra and angle-resolved radiances are reported.  A
high-order envelope is mandatory before claiming science validity for
`omega0>=0.9`, `g>=0.6`, or scattering optical depth above one.

## Stage 9: directed JWST-like cross-retrievals

Stage 9 measures how the forward-model differences localized in Stages 1--8
propagate into posterior inference.  It has two equally explicit arms:

- **cloud-free retrievals**, using the validated molecular+CIA models; and
- **cloudy retrievals**, using cloud domains accepted in Stages 7 and 8 and
  retrieving the cloud parameters present in each injection.

Both arms use:

- R=100 spectra over `0.3--12 micron`, a frozen stellar spectrum, and a frozen
  instrument response; native JWST mode combinations are secondary tests;
- constant eclipse-depth uncertainties of `30`, `60`, and `100 ppm`, also
  reported as fractions of the median eclipse depth;
- exact noiseless spectral means for the primary directed comparisons;
- at least 20 deterministic noise seeds per uncertainty tier for coverage;
- all six directed cross-code injection/retrieval pairs;
- self-retrievals at `60 ppm` as sampler and parameterization controls;
- frozen priors, parameter transforms, likelihood, live-point/effective-sample
  targets, and termination criteria.

The cloud-free arm retrieves temperature-profile parameters, H2O, CO, CO2,
and CH4 abundances, and radius ratio or normalization.  The cloudy arm adds
only the cloud parameters present in its frozen injection contract.

Primary outputs are posterior median bias, bias divided by posterior standard
deviation, 68 and 95 per cent truth coverage, posterior width ratios,
pairwise distribution distances, best-fit residuals, reduced chi-square,
evidence differences when sampler uncertainty permits, likelihood evaluation
counts, wall time, peak resident memory, and effective sample size.  Failed,
incomplete, and multimodal runs remain visible.

## Stage-9 compute and deployment contract

Stage 9 is prepared locally but its complete sampler matrix is not a laptop
benchmark.

Local work includes implementation, unit/contract tests, frozen synthetic
inputs, retrieval configurations, manifests, plotting/summary tools, and small
end-to-end smoke retrievals.  These files are committed and pushed to the
normal GitHub branch.  Glamdring is the primary execution target: the cluster
checkout is created or updated from GitHub, and immutable commit hashes and
input checksums are recorded before submission.  DiRAC remains a fallback
rather than the default target.

Large chains, checkpoints, and raw per-run products remain on managed cluster
storage with checksums.  Small posterior summaries, convergence records,
manifests, and paper products are brought back to the development repository
and versioned.  No cluster-only manual edit may be required to reproduce a
run.

The full Stage-9 matrix is launched only after a local pilot records
steady-state likelihood time and peak memory, allowing the Glamdring process
count and memory request to be set from measurements rather than guesswork.

## Publication stopping criterion

The intercomparison is ready for submission when all nine stages have
reproducible manifests and primary figures; shared-input discrepancies satisfy
their predeclared gates or have a converged documented attribution; Stage 8
has a high-order scattering envelope for the claimed cloud domain; and both
cloud-free and cloudy Stage-9 directed retrievals report bias, coverage,
convergence failures, and resource use at every declared uncertainty tier.
