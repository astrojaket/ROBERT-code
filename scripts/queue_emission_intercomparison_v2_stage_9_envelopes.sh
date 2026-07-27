#!/usr/bin/env bash
# Submit one six-run Stage-9 posterior-envelope batch through addqueue.
set -euo pipefail

if [[ "${HOSTNAME:-}" != *"glamdring"* ]]; then
  echo "Refusing submission: this command is for Glamdring only." >&2
  exit 2
fi
if [[ $# -ne 2 ]]; then
  echo "usage: $0 FRAMEWORK SCENARIO" >&2
  exit 2
fi
for required in STAGE9_PROJECT_ROOT STAGE9_REPOSITORY STAGE9_ENVIRONMENT_PARENT STAGE9_PICASO_REFDATA STAGE9_PICASO_CK_DIRECTORY STAGE9_PRT_INPUT_DATA; do
  if [[ -z "${!required:-}" ]]; then
    echo "$required must be exported before submission" >&2
    exit 2
  fi
done
if [[ "${STAGE9_CONFIRM_ENVELOPE_SUBMISSION:-}" != "YES" ]]; then
  echo "Set STAGE9_CONFIRM_ENVELOPE_SUBMISSION=YES to submit envelope jobs." >&2
  exit 2
fi

framework="$1"
scenario="$2"
case "$framework" in
  picaso) memory_per_cpu_gb=3 ;;
  petitradtrans) memory_per_cpu_gb=6 ;;
  robert)
    case "$scenario" in
      clear_non_inverted|clear_inverted) memory_per_cpu_gb=8 ;;
      grey_absorbing_non_inverted|grey_scattering_non_inverted)
        memory_per_cpu_gb=11
        ;;
      *) echo "unsupported Stage-9 scenario: $scenario" >&2; exit 2 ;;
    esac
    ;;
  *) echo "unsupported Stage-9 framework: $framework" >&2; exit 2 ;;
esac
case "$scenario" in
  clear_non_inverted|clear_inverted|grey_absorbing_non_inverted|grey_scattering_non_inverted)
    ;;
  *) echo "unsupported Stage-9 scenario: $scenario" >&2; exit 2 ;;
esac

run_root="$STAGE9_PROJECT_ROOT/runs/$framework/$scenario"
if [[ ! -d "$run_root" ]]; then
  echo "run directory does not exist: $run_root" >&2
  exit 2
fi

submitted=0
skipped_complete=0
skipped_incomplete=0
while IFS= read -r -d '' run_config; do
  run_directory="$(dirname "$run_config")"
  run_id="$(basename "$run_directory")"
  if [[ ! -s "$run_directory/result.json" || ! -s "$run_directory/result_arrays.npz" ]]; then
    echo "Skipping retrieval without saved posterior: $run_id"
    skipped_incomplete=$((skipped_incomplete + 1))
    continue
  fi
  if [[ -s "$run_directory/diagnostic_spectra.npz" \
     && -s "$run_directory/diagnostic_tp.npz" \
     && -s "$run_directory/diagnostic_chemistry.npz" \
     && -s "$run_directory/posterior_envelope_metadata.json" ]]; then
    echo "Skipping completed envelope products: $run_id"
    skipped_complete=$((skipped_complete + 1))
    continue
  fi

  launcher="$run_directory/addqueue-envelope-launch.sh"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'export STAGE9_TASK=posterior-envelopes\n'
    printf 'export STAGE9_FRAMEWORK=%q\n' "$framework"
    printf 'export STAGE9_RUN_CONFIG=%q\n' "$run_config"
    printf 'export STAGE9_PROJECT_ROOT=%q\n' "$STAGE9_PROJECT_ROOT"
    printf 'export STAGE9_REPOSITORY=%q\n' "$STAGE9_REPOSITORY"
    printf 'export STAGE9_ENVIRONMENT_PARENT=%q\n' "$STAGE9_ENVIRONMENT_PARENT"
    printf 'export STAGE9_PICASO_REFDATA=%q\n' "$STAGE9_PICASO_REFDATA"
    printf 'export STAGE9_PICASO_CK_DIRECTORY=%q\n' "$STAGE9_PICASO_CK_DIRECTORY"
    printf 'export STAGE9_PRT_INPUT_DATA=%q\n' "$STAGE9_PRT_INPUT_DATA"
    printf 'exec %q\n' "$STAGE9_REPOSITORY/scripts/submit_emission_intercomparison_v2_stage_9_task.sh"
  } > "$launcher"
  chmod 750 "$launcher"

  submission_output="$(
    addqueue -q redwood -s -c "s9-envelope-${framework}-${scenario}-${run_id}" \
      -n 1x12 -m "$memory_per_cpu_gb" -r "$launcher" 2>&1
  )"
  printf '%s\n' "$submission_output"
  if [[ "$submission_output" == *"Batch job submission failed"* ]] \
    || [[ "$submission_output" != *"Sending program's output to file:"* ]]; then
    echo "Stopping after an unconfirmed envelope submission: $run_id" >&2
    exit 1
  fi
  submitted=$((submitted + 1))
done < <(find "$run_root" -mindepth 2 -maxdepth 2 -name run.json -type f -print0 | sort -z)

echo "Envelope jobs submitted: $submitted"
echo "Already complete: $skipped_complete"
echo "Retrieval posterior missing: $skipped_incomplete"
