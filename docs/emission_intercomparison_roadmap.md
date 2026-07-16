# Emission intercomparison roadmap

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

- R=100 spectra over `0.8--12 micron`, a frozen stellar spectrum, and a frozen
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
