# ROBERT code and upgrade audit — 6 November 2026

Follow-up on the same date: the Virga fixture failures and joint checkpoint
identity defect are corrected. Independent-element run creation now preserves
its required null fields. The full code suite passes 1090 tests, with 40
optional/external checks skipped and 73% coverage. Local 50/100-live-point
elemental checks converge without iteration caps. See the
[follow-up record](59_local_elemental_recovery_2026-11-06.md). The original
audit findings below retain their dated baseline for traceability; resolved
register items are marked explicitly.

## Result and scope

The published implementation supports the documented emission, transmission,
and retrieval routes. Several recent additions are experimental or still need
science checks. The published test suite is **not green** in this audit.

The audited GitHub `main` revision was
`17535f0eae9d3403bc8c4b836117965741e21fb0`. Its identity was checked over SSH.
The audit used a clean worktree of that revision and a separate read-only
review of the Developer checkout's unpublished changes. This documentation
update does not publish those source changes or change existing runs.

The checks below establish software status. They do not establish a new
scientific validation result. See [supported methods](../supported_methods.md)
for the scope of each existing comparison.

## Checks made for this audit

All Python commands used the `robert-exoplanets` Conda environment, Python
3.12.13. The clean suite selected JAX CPU, the Agg plotting backend, and one
thread per numerical library.

| Check | Tree | Result |
| --- | --- | --- |
| Full `python -m pytest -q` | Published revision | Failed: 11 failures, all in `tests/test_virga_config.py`. The suite collects 1,108 tests. |
| `python -m pytest -q tests/test_virga_config.py --tb=short --disable-warnings` | Published revision | 11 failed, 14 passed. The failures reproduce without running the wider suite. |
| `python -m ruff check .` | Published revision | Passed. |
| `python -m mypy` | Published revision | Passed for its two configured files. This is not a package-wide type check. |
| Chemistry, task config, configured H-minus, exact KTA bins, and opacity-memory tests | Developer tree | 133 passed. Two focused FastChem task checks and 11 covariance-observation tests also passed. |
| Retrieval supervision, joint geometry, posterior models, post-processing, and forward factory/transmission tests | Unpublished Developer tree | 65 passed, one skipped. The skip is Linux `/proc` MPI cleanup. |
| `git diff --check` | Developer tree | Passed before this documentation update. |

The unpublished subset command was:

```bash
conda run -n robert-exoplanets python -m pytest -q \
  tests/test_joint_geometry.py tests/test_callback_supervisor.py \
  tests/test_run_retrieval_supervision.py tests/test_posterior_models.py \
  tests/test_postprocessing.py tests/test_forward_model_factory.py \
  tests/test_transmission_forward_model.py
```

Native MPI failure tests, a new converged cloud recovery, and accelerator
hardware tests were not run. The authenticated Dirac and Glamdring shared SSH
sockets were absent. Cluster states below are dated task records, not live
queue checks.

## Findings that need action

### 1. Virga configuration tests stop before their intended assertions

`tests/test_virga_config.py` inserts `radiative_transfer.gravity_model` into
every fixture, including emission fixtures. The RT schema correctly rejects
this transmission-only option for emission. Eleven tests therefore fail at
input validation, before they can check cloud construction or cloud priors.

Evidence: the fixture's `_write_config` helper and
`RadiativeTransferConfig.validate_model_options` in
[`task_config.py`](../../src/robert_exoplanets/io/task_config.py).

Action: set this field only for transmission fixtures. Retain tests that reject
invalid emission options. Run the Virga configuration tests, then the portable
suite. Do not relax the public schema to accept an irrelevant emission field.
This is a test-fixture defect; it does not by itself show a Virga physics defect.

### 2. MPI failure handling has no completed native release check

Serial supervised failure handling is published in `54ee578`. The Developer
checkout extends it to MPI. The local subset passes, but its process cleanup
test is Linux-only and does not run on this Mac. A simulated launcher is also
not proof that native Open MPI/PyMultiNest stops cleanly when a worker fails.

Action: review the unpublished process lifecycle and failure classification.
After local checks, commit and push the intended changes. Pull that commit on
Glamdring and run a queued non-root-rank failure test. Require a nonzero launcher
exit, a retained cause, no orphan worker, and no successful retrieval result.

### 3. General posterior plots still need the requested figure contract

The unpublished spectrum change uses the saved pointwise posterior median for
the curve and bands. Its residual panel and fit statistic still use the saved
best-fit prediction and label that source. The general plot function still
sets a title and grid and uses larger legend text. This does not meet the
WASP-15b figure style.

Evidence: `_plot_fit` in
[`postprocessing.py`](../../src/robert_exoplanets/postprocessing.py).

Action: apply the shared style, use the label "Posterior median model", and
state the residual/statistic source. Verify that saved draws and probability
intervals remain unchanged. Check the final PNG and vector output for overlap.

### 4. Quality gates cover fewer paths than the package can run

CI has Linux Python 3.10–3.14 jobs, lint, a package build, a 70% coverage floor,
opacity integration, and HRS helper checks. It has no documentation link/build
gate, macOS smoke job, or accelerator hardware job. Mypy checks only
`retrieval/protocols.py` and `tests/test_retrieval_contract.py`.

Action: first restore the portable suite. Then add documentation links and
public-command checks, a macOS smoke job, and incremental domain/interface type
checks. Report optional native and hardware checks separately.

### 5. Joint-geometry checkpoint identity omits wrapped likelihood settings

The local `_BoundLikelihood` exposes a name and a call adapter. The manifest
writer sees this wrapper, rather than the component's underlying likelihood
options. A focused check changed a Gaussian component's
`include_normalization` setting: the configuration hash stayed equal and
`resume_compatibility_changes` returned no changes. The likelihood definition
had changed. This can permit an incompatible checkpoint resume.

Evidence: `_BoundLikelihood` and `build_joint_geometry_problem` in the local
`retrieval/joint_geometry.py`; `_likelihood_record`,
`_single_likelihood_record`, and `resume_compatibility_changes` in
[`manifest.py`](../../src/robert_exoplanets/retrieval/manifest.py).
The prototype file is not yet published.

Action: record the underlying likelihood type and all scientific settings,
plus parameter bindings. Add a regression that rejects a changed component
likelihood on resume. Complete component-aware posterior collection for
offsets and atmospheric profiles. Keep this prototype out of production
checkpoint workflows until these gates pass.

## Upgrade register

"Published" means present at the audited `main` revision. "Local" means present
only in the reviewed Developer tree. Neither label means science validated.

| Upgrade | Status on 6 November | Evidence and next completion check |
| --- | --- | --- |
| Independent FastChem elemental abundances | Published; focused fixed-temperature recovery complete with 50 and 100 live points. | [Element tutorial](../tutorials/05_fastchem_elemental_abundances.md), [follow-up](59_local_elemental_recovery_2026-11-06.md). Both runs include the truth, but C/H and Si/H remain broad. General target-science validation remains open. |
| Configured H-minus and exact spectral observations | Published, implemented and tested (`19ed2d4`). | [H-minus tutorial](../tutorials/06_configured_hminus.md). Check the intended hot-atmosphere case and H-, H, electron profiles; retain physical samples and declared bin edges. |
| Native-opacity preparation memory bound | Published (`17535f0`). | One native KTA provider is released before the next species; its unused full-size log-opacity cache is disabled. Full-table conversion can still allocate large arrays. The 4 November Dirac task reports successful preparation and deployment of this commit. Measure peak memory and numerical equality on the intended R=15000 inputs. |
| Pressure quench and optional CH4 gradient | Published; analytic convention tested. CH4 gradient is experimental (`5767ec3`). | [Quench contract](../validation/pressure_quench.md). Independent profile parity and a declared disequilibrium science case remain open. A separate solar noise ensemble reported 93/100 and 96/100 C/O interval coverage; that does not validate all quench regimes. |
| Virga EddySed clouds and fitted fsed/Kzz | Published, experimental (`2ed6666`, `8af6f2f`). Vapor mass-ratio correction is published (`1493f56`). | [Virga review](56_virga_cloud_experiment.md). Direct Virga parity and small forward checks exist. FastChem condensates are diagnostics, not a coupled cloud-abundance model. Complete a converged fitted P–T/cloud recovery before a new GCM cloud retrieval. |
| High-stream scattering comparison | Bounded synthetic check complete (`7fc0925`). | Four five-layer SH4/PythonicDISORT cases were compared through 64 streams. This does not complete real Mie-cloud or posterior validation. Retain phase-function, layer, and angular convergence gates. |
| Cloud injection recovery | Pilot incomplete; replacement prepared. | Last record, 4 November: pilot hit a 512-iteration cap and had effective sample size one. Replacement uses 12 Redwood CPUs, 128 live points, no iteration cap; it was not submitted. Finish MPI checks, then run it through `addqueue` and assess recovery. |
| Retrieval failure handling | Serial published (`54ee578`); MPI extension local and unfinished. | Native non-root failure, normal completion, and checkpoint-resume gates are required. See finding 2. |
| Posterior spectrum and cloud products | Cloud profiles published (`ea361c9`); median-curve refinements local. | Use the same 100 weighted draws for all products. Finish saved-array, interval, curve-label, statistic-source, and figure-style checks. |
| Joint emission/transmission composition | Published experimental Python API; checkpoint identity defect resolved. | Likelihood settings, supported covariance state, prepared observations, and parameter bindings are retained in manifests. Standard YAML still chooses one geometry. Physical joint recovery and posterior integration remain open. |
| LBL/HRS and WASP-77Ab joint HRS/LRS | Method/operator gates exist; full real-data atmospheric runs deferred. | [Method review](51_line_by_line_high_resolution_retrieval_methods.md), [target review](52_wasp77ab_real_joint_retrieval.md). Native stride 1 passed the recorded gate; stride 2 did not. Fixed-template recovery is not the full atmospheric retrieval. |
| JAX/Metal and opacity-sampling acceleration | Implemented experimental paths. | [Metal review](48_jax_metal_backend.md), [opacity-sampling review](50_jax_opacity_sampling_accelerator.md). Benefit depends on grid, hardware, and full likelihood cost. Posterior equivalence and long-run memory checks remain open. See the [new BeAR audit](58_bear_2026_performance_audit.md). |
| Reflected light and HWO | Planned. | A direct-beam source diagnostic exists inside thermal emission. A reflecting surface, reflected-light multiple scattering, disc/phase integration, and an HWO observation/noise model are still needed. |
| Benchmark paper and release | Analysis and draft revisions reported complete; author review and release remain. | Latest paper task reports a compiled draft synced on 26 October, including stronger-cloud analysis and controlled CO checks. Reconcile the reviewed manuscript, code revisions, and compact result manifests before release. This audit did not rebuild the paper. |
| Photochemistry emulator integration | External development; not integrated into ROBERT. | Last record, 4 November: checkpoint recovery job 1040773 was running after 100 preflight tests; the pilot was on hold. Verify restart equivalence, then emulator accuracy, bounds, conservation, and the ROBERT interface. No live job state was available here. |

The separate 4 November Dirac record reports WASP-178b retrieval job `671108`
submitted at `17535f0` after preparation passed. Its present state and posterior
are unverified. Do not treat submission as completion. Keep existing runs and
their recorded code versions unchanged.

## Ordered to-do list

| Priority | Action | Completion check |
| --- | --- | --- |
| Done | Correct the Virga emission test fixtures. | All 25 Virga config tests and the full suite pass without weakening schema checks. |
| P1 | Finish and publish MPI failure handling as a separate code change. | Local checks pass; the pulled commit passes a queued non-root-rank failure test and leaves no orphan workers. |
| Done | Correct joint-geometry likelihood identity. | Changed wrapped settings, covariance, type, velocity roles, or prepared data block resume. Old wrapper-only joint manifests are rejected. |
| P1 | Complete the prepared P–T/cloud injection recovery. | Use 12 Redwood CPUs and no iteration cap. Save convergence, posterior recovery, prior checks, and timings. Do this before the new GCM cloud application. |
| P1 | Check the existing Dirac WASP-178b run after SSH is restored. | Record job state, imported Git revision, opacity identity, completion status, and posterior diagnostics. |
| P2 | Finish posterior median products and figure style. | Saved curves match saved quantiles/draws; labels identify median and diagnostic sources; visual checks pass. |
| P2 | Complete chemistry/H-minus science checks beyond the local elemental case. | The fixed-1800 K elemental check passes. A nonzero H-minus case and target-specific recovery remain required. |
| P2 | Complete physical and output validation for the published joint prototype. | One shared-parameter emission/transmission recovery and complete posterior/result products pass. The binding and checkpoint tests are already complete. |
| P2 | Measure whole-likelihood CPU and GPU costs before further speed work. | Use fixed inputs and proposal vectors; record setup, compilation, warm times, memory, transfers, likelihood error, and full sampler wall time. Follow the [BeAR experiment plan](58_bear_2026_performance_audit.md). |
| P2 | Complete the deferred shared-atmosphere HRS/LRS target test. | Match the declared physics, priors, opacity, filter, and data operators; compare separate and joint posteriors and calibration. |
| P2 | Finish the paper/release evidence. | Scientific author review, reproducible revision/input manifests, final figures, availability text, and a tagged release agree. |
| P3 | Expand CI and incremental type checks. | Documentation/public commands and the declared platforms have visible checks; native/hardware skips remain explicit. |
| P3 | Develop HWO and emulator work after current gates. | Each starts with a separate physical contract and an independent validation case. |

Historical branches and archived studies are recovery material. Branch
existence is not evidence that work is current or complete. Port only reviewed
changes that the present workflow needs.
