# Emission intercomparison V2 Stage-9 Glamdring runbook

This runbook launches the frozen 72-retrieval matrix in four atmospheric-
scenario batches. All filesystem paths live below `/mnt/users/jaketaylor/`.
The `redwood` name is a scheduler queue argument only; it is not part of a
filesystem path.

## 1. Fixed paths and checkout

```bash
export STAGE9_USER_ROOT=/mnt/users/jaketaylor
export STAGE9_REPOSITORY="$STAGE9_USER_ROOT/ROBERT-code"
export STAGE9_ENVIRONMENT_PARENT="$STAGE9_USER_ROOT/stage9-environments"
export STAGE9_PROJECT_ROOT="$STAGE9_USER_ROOT/ROBERT-stage9"
export STAGE9_REFERENCE_SOURCE="$STAGE9_USER_ROOT/stage9-reference-source"

cd "$STAGE9_REPOSITORY"
git fetch origin
git switch codex/emission-intercomparison-v2-stage-9-setup
git pull --ff-only origin codex/emission-intercomparison-v2-stage-9-setup
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
`-s -n 12` pattern starts one wrapper in a one-node, 12-core allocation. The
wrapper then starts one self-contained 12-rank Conda MPICH/Hydra world. Submit
one preflight for each framework:

```bash
module list
```

If an OpenMPI module is listed, run `module unload openmpi` before submission.
The launcher rejects a loaded OpenMPI module instead of risking an ABI mixture.

```bash
export STAGE9_TASK=preflight

export STAGE9_FRAMEWORK=picaso
addqueue -q redwood -s -c s9-preflight-picaso -n 12 -m 32 \
  -r "$STAGE9_REPOSITORY/scripts/submit_emission_intercomparison_v2_stage_9_task.sh"

export STAGE9_FRAMEWORK=petitradtrans
addqueue -q redwood -s -c s9-preflight-prt -n 12 -m 64 \
  -r "$STAGE9_REPOSITORY/scripts/submit_emission_intercomparison_v2_stage_9_task.sh"

export STAGE9_FRAMEWORK=robert
addqueue -q redwood -s -c s9-preflight-robert -n 12 -m 96 \
  -r "$STAGE9_REPOSITORY/scripts/submit_emission_intercomparison_v2_stage_9_task.sh"
```

Do not continue until `integrity/preflight-{picaso,petitradtrans,robert}.json`
exist and agree with the frozen versions, paths, hashes, and 12-rank ABI.

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

addqueue -q redwood -s -c s9-fwdpilot-picaso-clear -n 12 -m 32 \
  -r "$STAGE9_REPOSITORY/scripts/submit_emission_intercomparison_v2_stage_9_task.sh"
```

Change the framework, scenario, unique output filename, and memory request for
the remaining five jobs. Use 32 GB for PICASO, 64 GB for pRT, 96 GB for clear
ROBERT, and 128 GB for cloudy ROBERT.

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

addqueue -q redwood -s -c s9-retpilot-picaso -n 12 -m 32 \
  -r "$STAGE9_REPOSITORY/scripts/submit_emission_intercomparison_v2_stage_9_task.sh"
```

Repeat with the pRT and ROBERT configuration/output paths and their 64/96 GB
requests. Resubmitting the same pilot output exercises MultiNest resume; for a
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
`result.json`, `result_arrays.npz`, `diagnostic_spectra.npz`, and its native
MultiNest `chains/` directory. Preserve incomplete chains and resubmit only
that run's generated `addqueue-launch.sh`; do not resubmit an entire shard.

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
