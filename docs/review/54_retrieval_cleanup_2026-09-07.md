# Retrieval cleanup and validation

Status: 2026-09-07 working tree. This extends the [repository audit](53_repository_audit_2026-09-07.md).

## Changes

- PyMultiNest is the supported nested sampler. UltraNest code, configuration,
  dependencies, and old cumulative-call example loops were removed.
- Optimal Estimation retains its explicit independent-Gaussian contract. It is
  the path to extend for detailed sounding models, with suitable validation.
- Single, multiple-dataset, heterogeneous, and device problems share a checked
  sampler protocol. CI runs scoped mypy checks on concrete implementations.
- Resume checks normalize defaults, reject changed scientific inputs and
  structural settings, and record allowed budget/stopping changes per attempt.
- Best-fit spectra and observations have portable JSON/NPZ products. Saved
  Gaussian fit statistics preserve offsets, jitter, scaling, and normalization.
  Unsupported objectives do not acquire fabricated Gaussian statistics.
- Factory opacity preparation is shared. Cache schema v2 identifies the source
  path, species, format, resolution, full binning settings, coordinates, units,
  and bin edges. Old caches require fresh preparation.

## R=1000 inputs and speed

The 16 tables from `Dropbox/NemesisPy-Docker/ktables_exomol/` were copied to
`opacity_data/ktables_exomol/`. All SHA-256 checks match. The collection is
3,892,412,160 bytes and stays outside Git. The local copy manifest records its
identity. R=1000 is the default resolution; target configurations and the
project template select this directory. R=100 is still explicit in quick checks.

Timing used six molecules, 80 layers, the same physical states, and separate
NIRCam F322W2, NIRCam F444W, and MIRI LRS grids. The 274 points cover
2.45–11.88 microns at resolving powers of about 20–506. A blackbody stellar
spectrum isolates this comparison from external stellar-file access.

The baseline is the wheel saved before this follow-up cleanup, not Git HEAD;
the initial working tree already contained substantial uncommitted work.
Each case has three process blocks and 72 timed evaluations per operation,
with one numerical thread. All saved spectra and likelihood values are
bit-for-bit equal before and after the changes.

| Case | Forward before / after | Likelihood before / after |
| --- | --- | --- |
| Clear | 166.05 / 166.97 ms | 166.15 / 167.57 ms |
| Mie cloudy | 552.21 / 539.10 ms | 545.11 / 547.72 ms |

The likelihood changes are below 1% and within observed timing variation.
This is not evidence of a material speed change. Final cache validation adds
about 38 ms to clear-model construction and 49 ms to cloudy construction.
It does not run inside the likelihood loop. Peak process RSS was about
0.76 GiB for the clear case and 2.11 GiB for the cloudy case, including startup
and compilation. The latter is not a sub-2-GiB result.

See [compact timing and input evidence](../data/r1000_cleanup_performance_20260907.json).

A later consolidation check compared the same clear and cloudy cases directly
with GitHub production commit `ef2e8074616f9b6740d64d6b8a6a4079d17be2f5`.
All spectra, likelihoods, physical states, and instrument grids were bit-for-bit
equal across the six input states. This closes the GitHub comparison gap for
these cases; it does not replace a full reproduction of the emission paper.

## Posterior plots

Automatic retrieval plotting selects 100 weighted posterior draws. The same
vectors define median, central 68.27%, and central 95.45% curves for each
instrument spectrum, T-P profile, VMR profile, and supported cloud profile.
Curves and bands use `mediumpurple`. Numerical products retain the vectors,
quantile probabilities, coordinates, and units. OE uses a labelled bounded
Gaussian approximation rather than a sampled nested posterior.

One hundred spectral evaluations cost about 17 seconds for the clear timing
case and 55 seconds for the cloudy case, before profile and figure overhead.
This is work after inference; it does not slow the sampler's likelihood loop.

## Scientific checks completed

| Check | Result |
| --- | --- |
| LBL K-band versus stored petitRADTRANS reference | Pass |
| LBL science grid, pressure resolution, stride, and wide band | Pass |
| H-minus continuum versus stored petitRADTRANS reference | Pass |
| HRS response/filter/covariance operator benchmark | Pass |
| R=1000 cloudy automatic/reference SH4 solver paths | Identical spectra on all three grids |
| Native PyMultiNest start and checkpoint resume | Pass |
| Two-rank R=100 emission injection recovery | Pass; truth inside 95% interval; reduced chi-square 1.223 |
| Two-rank R=100 transmission injection recovery | Pass; truth inside 95% interval; reduced chi-square 0.528 |
| Post-processing both completed R=100 retrievals | Pass; 100 saved draws, five spectral/T-P/VMR curves, and rendered figures |
| Cloud diagnostics with 100 prescribed R=1000 states | Pass on all three instrument grids; mass-fraction and particle-radius profiles |

The native runs used the installed, ABI-matched Open MPI 5.0.10 and MultiNest
3.10 libraries in `robert-exoplanets`. Fresh environments from the maintained
Conda file use its declared MPI stack. Do not mix those stacks within a run.

The final suite passed **837 tests**, with **27 optional checks skipped** and
**72% coverage** against the 70% floor. Native PyMultiNest start/resume checks
were enabled. A separate fresh-process JAX CPU run passed **24 tests**; five
real Metal hardware checks remain unrun. Ruff, scoped mypy, dependency checks,
and `git diff --check` passed. Wheel and source-distribution builds passed.

Keep optional JAX CPU checks in a separate process from the main suite. A
combined run accumulated JAX compilation memory and triggered three existing
process-RSS guards. The guards were kept unchanged, and the final separated
runs passed.

The cloud diagnostic check varies temperature, cloud mass fraction, and particle
radius. It checks the product path, not posterior calibration. Visual review
checked the median and two bands, pressure direction, units, and axis labels.

## Remaining boundaries

Keep the domain objects and numerical solvers; a rewrite would discard useful
physical validation. Continue to reduce workflow and diagnostic duplication
when a concrete extension needs it. The post-processing module still combines
saved-result handling and figure rendering; separate those responsibilities
when the next diagnostic family is added. Detailed sounding work should extend
OE with validated Jacobians and explicit observation-error models.
Custom Python models must declare their
physical and external-input identity; a callable cannot identify its own science.

Real Metal hardware, long shared-VMR HRS/LRS target runs, and broad posterior
calibration across seeds remain separate release checks. The local passes do
not certify those regimes or independent science-grade cloudy agreement.

## Repository consolidation

The `codex/production-consolidation-20260907` branch gathers the existing
uncommitted LBL/HRS and accelerator work with this retrieval cleanup. It also
contains both production-status and reproducible-CI commits from GitHub main.
The large total change against main includes that earlier feature work; it is
not a measure of code added by the folder cleanup.

The 36 maintained YAML files now live under `configurations/quickstart/`,
`configurations/examples/`, and `configurations/targets/`. Two redundant
MultiNest wrappers were removed in addition to the two obsolete UltraNest
wrappers. Maintained runners, tests, and documentation use the new locations.
Use `paths` for runtime directories; the stale `housekeeping` alias is rejected.
Scientific settings and instrument-specific grids were retained. Existing
prepared caches from the old schema require `--prepare-opacity` once.

A complete, verified Git bundle records the old refs under the ignored local
`external_data/consolidation/20260907/` directory. Cleanup removed 17 obsolete
local branch refs and two obsolete remote refs. Eight local branches remain;
unique paper, performance, and research histories were retained. No Git object
garbage collection was run. Heavy inputs, local research projects, and runtime
caches are excluded from the consolidation commit.

Five completed tasks were archived. Archiving the HAT-P-32b proposal test task
also deleted its managed worktree. The app snapshot restored its code and three
scripts, but not ignored simulation outputs. The user accepted rerunning those
closed test simulations if needed. The restored task remains open. This cleanup
does not mark unfinished research as complete.

The next research actions are Stage 9 posterior post-processing and paper
interpretation, CLR trial reconciliation and predictive checks, WASP-178b exact
refits/covariance assessment, and the deferred real-data HRS/LRS validation in
the development roadmap.

After configuration consolidation, the full suite passed **840 tests**, with
**27 optional checks skipped**, including native MultiNest start/resume.
All 36 maintained configurations loaded, the focused configuration suite passed
63 tests, and the bundled three-instrument forward example passed after cache
preparation. The five CI-policy tests also passed after merging GitHub main.
A separate JAX CPU run passed **24 tests**. Ruff, scoped mypy, dependency
checks, wheel/source builds, and archive-content checks passed. The package
retains the production classifier and excludes UltraNest and generated caches.
