# Emission intercomparison V2 Stage-9 Glamdring runbook

This runbook launches the frozen 72-retrieval matrix in four atmospheric-
scenario batches. All filesystem paths live below `/mnt/users/jaketaylor/`.
The `redwood` name is a scheduler queue argument only; it is not part of a
filesystem path.

> **Stronger grey-cloud rerun:** On branch
> `codex/stage9-numba-stronger-cloud`, set `STAGE9_PROJECT_ROOT` to
> `/mnt/users/jaketaylor/ROBERT-stage9-stronger-cloud`. Do not refresh or write
> to the original `/mnt/users/jaketaylor/ROBERT-stage9` tree. Prepare new
> injections and queue only `grey_absorbing_non_inverted` and
> `grey_scattering_non_inverted`. Export
> `STAGE9_BRANCH=codex/stage9-numba-stronger-cloud` before the commands below.

## 1. Fixed paths and checkout

```bash
export STAGE9_USER_ROOT=/mnt/users/jaketaylor
export STAGE9_REPOSITORY="$STAGE9_USER_ROOT/ROBERT-code"
export STAGE9_ENVIRONMENT_PARENT="$STAGE9_USER_ROOT/stage9-environments"
export STAGE9_PROJECT_ROOT="${STAGE9_PROJECT_ROOT:-$STAGE9_USER_ROOT/ROBERT-stage9}"
export STAGE9_REFERENCE_SOURCE="$STAGE9_USER_ROOT/stage9-reference-source"
export STAGE9_BRANCH="${STAGE9_BRANCH:-codex/emission-intercomparison-v2-stage-9-setup}"

cd "$STAGE9_REPOSITORY"
git fetch origin
git switch "$STAGE9_BRANCH"
git pull --ff-only origin "$STAGE9_BRANCH"
git log -1 --oneline
```

For a deployment prepared before the Glamdring MPI-launch correction, retain
the project tree and use the execution-only refresh in the next section. It
requires all non-execution science-contract content to remain identical.

For a deployment prepared before the MultiNest 3.10 seed-range correction,
use `--refresh-multinest-seeds` after pulling the corrected commit. This
audited migration records that frozen requested seeds are reduced into
MultiNest's native `0..30080` safe range only at the sampler boundary. It does
not alter immutable `run.json` definitions or checkpoint identities. It refuses
to run if any production run directory has output beyond its immutable
`run.json`; staged references, injections, and pilot directories are preserved.

## 2. Prepare and stage shared data

The following source layout is canonical. Change only a leaf name if the
already-staged source tree uses a different name; keep it below
`$STAGE9_USER_ROOT`.

```bash
export SOURCE_PICASO_REFDATA="$STAGE9_REFERENCE_SOURCE/picaso-refdata"
export SOURCE_PICASO_CK="$STAGE9_REFERENCE_SOURCE/picaso-resortrebin"
export SOURCE_PRT_INPUT_DATA="$STAGE9_REFERENCE_SOURCE/petitradtrans-input-data"
export SOURCE_ROBERT_OPACITY="$SOURCE_PRT_INPUT_DATA"

for source in \
  "$SOURCE_PICASO_REFDATA" \
  "$SOURCE_PICASO_CK" \
  "$SOURCE_PRT_INPUT_DATA" \
  "$SOURCE_ROBERT_OPACITY"; do
  test -d "$source" || { echo "Missing reference directory: $source"; exit 1; }
  test -n "$(find "$source" -type f -print -quit)" || {
    echo "Reference directory contains no files: $source"
    exit 1
  }
done

if [[ -f "$STAGE9_PROJECT_ROOT/integrity/setup_manifest.json" ]]; then
  "$STAGE9_ENVIRONMENT_PARENT/robert-stage9/bin/python" \
    "$STAGE9_REPOSITORY/scripts/prepare_emission_intercomparison_v2_stage_9.py" \
    "$STAGE9_PROJECT_ROOT" --refresh-execution-contract
else
  "$STAGE9_ENVIRONMENT_PARENT/robert-stage9/bin/python" \
    "$STAGE9_REPOSITORY/scripts/prepare_emission_intercomparison_v2_stage_9.py" \
    "$STAGE9_PROJECT_ROOT"
fi

export STAGE9_CLUSTER=glamdring
if [[ ! -f "$STAGE9_PROJECT_ROOT/integrity/reference_data_manifest.json" ]]; then
  "$STAGE9_ENVIRONMENT_PARENT/robert-stage9/bin/python" \
    "$STAGE9_REPOSITORY/scripts/stage_emission_intercomparison_v2_stage_9_reference_data.py" \
    "$STAGE9_PROJECT_ROOT" \
    --picaso-refdata "$SOURCE_PICASO_REFDATA" \
    --picaso-ck "$SOURCE_PICASO_CK" \
    --prt-input-data "$SOURCE_PRT_INPUT_DATA" \
    --robert-opacity "$SOURCE_ROBERT_OPACITY" \
    --mode symlink
fi

export STAGE9_PICASO_REFDATA="$STAGE9_PROJECT_ROOT/reference/picaso/refdata"
export STAGE9_PICASO_CK_DIRECTORY="$STAGE9_PROJECT_ROOT/reference/picaso/resortrebin"
export STAGE9_PRT_INPUT_DATA="$STAGE9_PROJECT_ROOT/reference/petitradtrans/input_data"

"$STAGE9_ENVIRONMENT_PARENT/robert-stage9/bin/python" \
  "$STAGE9_REPOSITORY/scripts/prepare_emission_intercomparison_v2_stage_9.py" \
  "$STAGE9_PROJECT_ROOT" --verify-only
```

Deployments that already completed setup and injections under a commit older
than the MultiNest seed correction must replace the execution refresh above
with this one-time command before verification:

```bash
"$STAGE9_ENVIRONMENT_PARENT/robert-stage9/bin/python" \
  "$STAGE9_REPOSITORY/scripts/prepare_emission_intercomparison_v2_stage_9.py" \
  "$STAGE9_PROJECT_ROOT" --refresh-multinest-seeds
```

The staging script inventories and links or copies existing reference trees;
it does not download or populate them. An empty source must be populated with
the validated reference data before staging.

The setup manifest must report 72 runs, 12 shards, zero noise vectors, and 12
required injection means.

## 3. Queue preflights

Exported Stage-9 paths above must remain in the submission shell. Glamdring's
`-s -n 1x12` pattern starts one wrapper in an explicitly one-node, 12-core
allocation. The wrapper then starts one self-contained 12-rank Conda
MPICH/Hydra world. With this topology, Glamdring translates `addqueue -m N`
into a per-CPU Slurm request. The production helper divides each intended total
memory budget across the 12 reserved cores and rounds upward. Do not pass the
total budget directly to `-m`, and do not replace `1x12` with unconstrained
`12`. Submit one preflight for each framework:

```bash
module list
```

If an OpenMPI module is listed, run `module unload openmpi` before submission.
The launcher rejects a loaded OpenMPI module instead of risking an ABI mixture.

```bash
export STAGE9_TASK=preflight

export STAGE9_FRAMEWORK=picaso
addqueue -q redwood -s -c s9-preflight-picaso -n 1x12 -m 3 \
  -r "$STAGE9_REPOSITORY/scripts/submit_emission_intercomparison_v2_stage_9_task.sh"

export STAGE9_FRAMEWORK=petitradtrans
addqueue -q redwood -s -c s9-preflight-prt -n 1x12 -m 6 \
  -r "$STAGE9_REPOSITORY/scripts/submit_emission_intercomparison_v2_stage_9_task.sh"

export STAGE9_FRAMEWORK=robert
addqueue -q redwood -s -c s9-preflight-robert -n 1x12 -m 8 \
  -r "$STAGE9_REPOSITORY/scripts/submit_emission_intercomparison_v2_stage_9_task.sh"
```

Do not continue until `integrity/preflight-{picaso,petitradtrans,robert}.json`
exist and agree with the frozen versions, paths, hashes, and 12-rank ABI.
MultiNest verbose progress reporting is enabled for production monitoring. This
changes terminal output only; it does not change the sampler trajectory,
stopping condition, checkpoints, or posterior.

## 4. Generate native injections

Generate three one-rank injections per scenario. Set `STAGE9_SCENARIO` to one
of the four batch names, submit these three commands, and wait for all three
`native_mean.npz` files before changing scenario:

```bash
export STAGE9_TASK=injection
export STAGE9_SCENARIO=clear_non_inverted

export STAGE9_FRAMEWORK=picaso
addqueue -q redwood -s -c "s9-inj-picaso-$STAGE9_SCENARIO" -n 1 -m 32 \
  -r "$STAGE9_REPOSITORY/scripts/submit_emission_intercomparison_v2_stage_9_task.sh"

export STAGE9_FRAMEWORK=petitradtrans
addqueue -q redwood -s -c "s9-inj-prt-$STAGE9_SCENARIO" -n 1 -m 64 \
  -r "$STAGE9_REPOSITORY/scripts/submit_emission_intercomparison_v2_stage_9_task.sh"

export STAGE9_FRAMEWORK=robert
addqueue -q redwood -s -c "s9-inj-robert-$STAGE9_SCENARIO" -n 1 -m 96 \
  -r "$STAGE9_REPOSITORY/scripts/submit_emission_intercomparison_v2_stage_9_task.sh"
```

Use `-m 128` for the two cloudy ROBERT injections. Repeat for
`clear_inverted`, `grey_absorbing_non_inverted`, and
`grey_scattering_non_inverted`. Verify three products per scenario and twelve
globally:

```bash
find "$STAGE9_PROJECT_ROOT/injections" \
  -path "*/$STAGE9_SCENARIO/native_mean.npz" -type f | wc -l
find "$STAGE9_PROJECT_ROOT/injections" -name native_mean.npz -type f | wc -l
```

PICASO cloud tables and returned spectra remain on PICASO's native
`wno`/`delta_wno` bin support. The adapter does not interpolate PICASO bin
centres onto either the cloud grid or the common R=100 grid.

## 5. Pilot gate

Run the committed 12-rank forward pilots for all three frameworks in
`clear_non_inverted` and `grey_scattering_non_inverted`. Submit them one at a
time. This is the PICASO clear example:

```bash
export STAGE9_TASK=forward-pilot
export STAGE9_FRAMEWORK=picaso
export STAGE9_SCENARIO=clear_non_inverted
export STAGE9_PILOT_OUTPUT="$STAGE9_PROJECT_ROOT/diagnostics/resource/forward-pilot-picaso-clear_non_inverted.json"

addqueue -q redwood -s -c s9-fwdpilot-picaso-clear -n 1x12 -m 3 \
  -r "$STAGE9_REPOSITORY/scripts/submit_emission_intercomparison_v2_stage_9_task.sh"
```

Change the framework, scenario, unique output filename, and memory request for
the remaining five jobs. The intended total budgets are 32 GB for PICASO,
64 GB for pRT, 96 GB for clear ROBERT, and 128 GB for cloudy ROBERT; with
`1x12`, use corresponding `-m` values of 3, 6, 8, and 11 GB per CPU.

Next run one cross-retrieval pilot per retriever, again one at a time. The
three frozen configurations are:

```text
runs/picaso/clear_non_inverted/clear_non_inverted__inj-robert__ret-picaso__060ppm__mean/run.json
runs/petitradtrans/clear_non_inverted/clear_non_inverted__inj-robert__ret-petitradtrans__060ppm__mean/run.json
runs/robert/clear_non_inverted/clear_non_inverted__inj-picaso__ret-robert__060ppm__mean/run.json
```

This is the PICASO submission:

```bash
export STAGE9_TASK=retrieval-pilot
export STAGE9_FRAMEWORK=picaso
export STAGE9_RUN_CONFIG="$STAGE9_PROJECT_ROOT/runs/picaso/clear_non_inverted/clear_non_inverted__inj-robert__ret-picaso__060ppm__mean/run.json"
export STAGE9_PILOT_OUTPUT="$STAGE9_PROJECT_ROOT/pilots/picaso/clear_non_inverted"
export STAGE9_PILOT_LIVE_POINTS=50
export STAGE9_PILOT_MAX_ITER=200

addqueue -q redwood -s -c s9-retpilot-picaso -n 1x12 -m 3 \
  -r "$STAGE9_REPOSITORY/scripts/submit_emission_intercomparison_v2_stage_9_task.sh"
```

Repeat with the pRT and ROBERT configuration/output paths and `-m 6`/`-m 8`
per-CPU requests. Resubmitting the same pilot output exercises MultiNest resume; for a
stronger checkpoint test, interrupt only through Glamdring's normal job-control
interface after a checkpoint exists, then resubmit the identical command.

Production remains disabled until the six forward pilots and three retrieval
pilots pass the finite-spectrum, native-bin, MPI, resume, RAM, wall-time, and
storage gates.

## 6. Four production batches

Only after the pilot gate passes:

```bash
export STAGE9_CONFIRM_PRODUCTION_SUBMISSION=YES
```

The four batches, in order, are:

1. `clear_non_inverted`
2. `clear_inverted`
3. `grey_absorbing_non_inverted`
4. `grey_scattering_non_inverted`

Each batch contains 18 retrievals. Submit one six-run retriever shard at a
time, wait for its six `posterior_summary.json` files, then submit the next:

```bash
export STAGE9_SCENARIO=clear_non_inverted

"$STAGE9_REPOSITORY/scripts/queue_emission_intercomparison_v2_stage_9_shard.sh" \
  "$STAGE9_PROJECT_ROOT/shards/picaso__$STAGE9_SCENARIO.json"

# Wait for six complete PICASO products before continuing.
find "$STAGE9_PROJECT_ROOT/runs/picaso/$STAGE9_SCENARIO" \
  -name posterior_summary.json -type f | wc -l

"$STAGE9_REPOSITORY/scripts/queue_emission_intercomparison_v2_stage_9_shard.sh" \
  "$STAGE9_PROJECT_ROOT/shards/petitradtrans__$STAGE9_SCENARIO.json"

# Wait for six complete pRT products before continuing.
find "$STAGE9_PROJECT_ROOT/runs/petitradtrans/$STAGE9_SCENARIO" \
  -name posterior_summary.json -type f | wc -l

"$STAGE9_REPOSITORY/scripts/queue_emission_intercomparison_v2_stage_9_shard.sh" \
  "$STAGE9_PROJECT_ROOT/shards/robert__$STAGE9_SCENARIO.json"
```

Require 18 completed summaries before moving to the next scenario:

```bash
find "$STAGE9_PROJECT_ROOT/runs" \
  -path "*/$STAGE9_SCENARIO/*/posterior_summary.json" -type f | wc -l
```

Expected cumulative counts are 18, 36, 54, and 72. A successful run also has
`result.json`, `result_arrays.npz`, `diagnostic_spectra.npz`,
`diagnostic_tp.npz`, `diagnostic_chemistry.npz`,
`posterior_envelope_metadata.json`, and its native MultiNest `chains/`
directory. The three diagnostic NPZ products are mandatory default outputs.
They contain the best-fitting solution and weighted posterior
q2.5/q16/q50/q84/q97.5 envelopes; they do not claim that a real observation
has known input values. Preserve incomplete chains and resubmit only that
run's generated `addqueue-launch.sh`; do not resubmit an entire shard.

## 7. Diagnostics and archival

After all 72 runs complete:

```bash
export MPLBACKEND=Agg

"$STAGE9_ENVIRONMENT_PARENT/robert-stage9/bin/python" \
  "$STAGE9_REPOSITORY/examples/plot_emission_intercomparison_v2_stage_9_spectra.py" \
  "$STAGE9_PROJECT_ROOT"

"$STAGE9_ENVIRONMENT_PARENT/robert-stage9/bin/python" \
  "$STAGE9_REPOSITORY/examples/plot_emission_intercomparison_v2_stage_9_posteriors.py" \
  "$STAGE9_PROJECT_ROOT"
```

Inspect the native-injection comparison, retrieval spectral comparison,
posterior comparison, and truth-recovery products before using the verified
per-run archive tool to remove any raw chain directory.

For one completed production retrieval, generate its saved-spectrum fit,
analytic PG14 temperature-pressure posterior, and weighted posterior corner
plot without loading opacities or evaluating a forward spectrum:

```bash
export STAGE9_RUN_CONFIG="$STAGE9_PROJECT_ROOT/runs/picaso/clear_non_inverted/clear_non_inverted__inj-robert__ret-picaso__060ppm__mean/run.json"

export MPLBACKEND=Agg
"$STAGE9_ENVIRONMENT_PARENT/robert-stage9/bin/python" \
  "$STAGE9_REPOSITORY/examples/plot_emission_intercomparison_v2_stage_9_run.py" \
  "$STAGE9_RUN_CONFIG"
```

The three PNG files are written below that run's `plots/` directory. Spectral
data are shown as the noiseless injection values with the run's 30, 60, or
100 ppm likelihood uncertainty as error bars. Framework colours follow the
paper palette: ROBERT is medium purple, petitRADTRANS is plum, and PICASO is
charcoal. The spectral panel labels the injector and retriever explicitly,
plots the best-fitting spectrum, and shades the wavelength-wise central 68%
posterior spectrum. That envelope is calculated from native forward-model
evaluations of every saved weighted posterior sample; it is not the observational
30, 60, or 100 ppm error envelope. The TP panel compares the best-fitting TP
profile with the exact Stage-9 truth reconstructed from the frozen simulation
contract and shades the saved central 68% TP posterior. The retrieval NPZ
itself contains no input/truth profile, so the same output schema remains
appropriate for real JWST observations.

Posterior dimensions and plotting metadata are read from each run's
`result.json` and `run.json`; the plotting code does not hard-code the fitted
parameter set or order. Labels, units, bounds, and optional reference values
come from the config. If a real-observation config omits reference/truth
values, no reference markers or input TP profile are plotted.

Newly completed retrievals calculate and save the exact spectral, TP, and
chemistry posterior quantiles automatically on the same 12-rank job. For a
retrieval completed before this capability was added, backfill all three
products as a scheduled Glamdring science post-processing job:

```bash
export STAGE9_TASK=posterior-envelopes
export STAGE9_FRAMEWORK=picaso
export STAGE9_RUN_CONFIG="$STAGE9_PROJECT_ROOT/runs/picaso/clear_non_inverted/clear_non_inverted__inj-robert__ret-picaso__060ppm__mean/run.json"

addqueue -q redwood -s -c s9-envelope-picaso-example -n 1x12 -m 3 \
  -r "$STAGE9_REPOSITORY/scripts/submit_emission_intercomparison_v2_stage_9_task.sh"
```

Use the run's retriever as `STAGE9_FRAMEWORK`. For a `1x12` job, pass the
per-CPU request to `-m`: 3 GB for PICASO, 6 GB for petitRADTRANS, 8 GB for
clear ROBERT, or 11 GB for cloudy ROBERT. Those reserve total node budgets of
36, 72, 96, and 132 GB respectively. The backfill is idempotent and stores only
q2.5/q16/q50/q84/q97.5, not the complete posterior spectrum, TP, or chemistry
matrices. To submit one six-retrieval framework/scenario batch, use:

```bash
export STAGE9_CONFIRM_ENVELOPE_SUBMISSION=YES

"$STAGE9_REPOSITORY/scripts/queue_emission_intercomparison_v2_stage_9_envelopes.sh" \
  picaso clear_non_inverted
```

Generate the large clear, non-inverted comparison product with one page per
uncertainty tier and injection framework:

```bash
"$STAGE9_ENVIRONMENT_PARENT/robert-stage9/bin/python" \
  "$STAGE9_REPOSITORY/examples/plot_emission_intercomparison_v2_stage_9_big_comparison.py" \
  "$STAGE9_PROJECT_ROOT" --scenario clear_non_inverted
```

Each page contains the injected spectrum with point-wise error bars, the two
available directed best-fitting spectra, the two best-fitting TP profiles and
their central 68% envelopes against the common input TP, and exactly four
molecular posterior panels: H2O, CO, CO2, and CH4. The multipage PDF and page
PNGs are written beneath `diagnostics/big_comparison/`.

Generate the publication atlas for all four scenarios after all 72 retrievals
and their compact posterior-envelope products are complete:

```bash
export MPLBACKEND=Agg

"$STAGE9_ENVIRONMENT_PARENT/robert-stage9/bin/python" \
  "$STAGE9_REPOSITORY/examples/plot_emission_intercomparison_v2_stage_9_paper_atlas.py" \
  "$STAGE9_PROJECT_ROOT"
```

This is saved-product post-processing only: it does not load opacities,
evaluate a forward model, start MultiNest, or require `addqueue`. It writes a
multipage `stage9_paper_atlas.pdf`, individual PDF and PNG pages, and a machine-
readable manifest beneath `diagnostics/paper_atlas/`. Each scenario receives a
spectral fit plus normalized-residual atlas, TP-recovery atlas, four-molecule
posterior atlas, optional cloud-posterior atlas, and parameter bias/truth-
inclusion matrix. Plot labels use mathematical scientific notation and the
frozen paper palette.

The bias matrix includes molecular and retrieved cloud parameters only; the
PG14 coordinates are excluded because thermal recovery is assessed from the
full TP profiles. Cell text is the signed posterior-median bias divided by the
weighted posterior standard deviation. Cell colour records whether the
injected value lies inside the central 68% interval, only inside the central
95% interval, or outside the central 95% interval. The accompanying
`parameter_bias_coverage.csv` retains every per-run value, while
`parameter_bias_coverage.json` aggregates by scenario/noise tier, directed
pair, and parameter family. These are single-spectrum truth-inclusion
statistics, not frequentist coverage estimates.

To regenerate only one scenario while refining a figure, use for example:

```bash
"$STAGE9_ENVIRONMENT_PARENT/robert-stage9/bin/python" \
  "$STAGE9_REPOSITORY/examples/plot_emission_intercomparison_v2_stage_9_paper_atlas.py" \
  "$STAGE9_PROJECT_ROOT" --scenario grey_scattering_non_inverted
```
