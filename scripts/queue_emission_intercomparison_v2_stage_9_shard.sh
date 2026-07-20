#!/usr/bin/env bash
# Submit one frozen 6-run shard through Glamdring's addqueue wrapper.
set -euo pipefail

if [[ "${HOSTNAME:-}" != *"glamdring"* ]]; then
  echo "Refusing submission: this command is for Glamdring only." >&2
  exit 2
fi
if [[ $# -ne 1 ]]; then
  echo "usage: $0 SHARD_JSON" >&2
  exit 2
fi
for required in STAGE9_PROJECT_ROOT STAGE9_REPOSITORY STAGE9_ENVIRONMENT_PARENT STAGE9_PICASO_REFDATA STAGE9_PICASO_CK_DIRECTORY STAGE9_PRT_INPUT_DATA; do
  if [[ -z "${!required:-}" ]]; then
    echo "$required must be exported before submission" >&2
    exit 2
  fi
done

shard_json="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
retriever="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["retriever"])' "$shard_json")"
scenario="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["scenario"])' "$shard_json")"
memory_gb="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["preliminary_memory_gb"])' "$shard_json")"
mapfile -t run_configs < <(python -c 'import json,sys; [print(item) for item in json.load(open(sys.argv[1]))["execution_order"]]' "$shard_json")

if [[ "${STAGE9_CONFIRM_PRODUCTION_SUBMISSION:-}" != "YES" ]]; then
  echo "Set STAGE9_CONFIRM_PRODUCTION_SUBMISSION=YES after the approved pilots pass." >&2
  exit 2
fi

for relative_config in "${run_configs[@]}"; do
  export STAGE9_RUN_CONFIG="$STAGE9_PROJECT_ROOT/$relative_config"
  run_id="$(basename "$(dirname "$STAGE9_RUN_CONFIG")")"
  launcher="$(dirname "$STAGE9_RUN_CONFIG")/addqueue-launch.sh"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'export STAGE9_RUN_CONFIG=%q\n' "$STAGE9_RUN_CONFIG"
    printf 'export STAGE9_PROJECT_ROOT=%q\n' "$STAGE9_PROJECT_ROOT"
    printf 'export STAGE9_REPOSITORY=%q\n' "$STAGE9_REPOSITORY"
    printf 'export STAGE9_ENVIRONMENT_PARENT=%q\n' "$STAGE9_ENVIRONMENT_PARENT"
    printf 'export STAGE9_PICASO_REFDATA=%q\n' "$STAGE9_PICASO_REFDATA"
    printf 'export STAGE9_PICASO_CK_DIRECTORY=%q\n' "$STAGE9_PICASO_CK_DIRECTORY"
    printf 'export STAGE9_PRT_INPUT_DATA=%q\n' "$STAGE9_PRT_INPUT_DATA"
    printf 'exec %q\n' "$STAGE9_REPOSITORY/scripts/submit_emission_intercomparison_v2_stage_9.sh"
  } > "$launcher"
  chmod 750 "$launcher"
  addqueue -q redwood -s -c "s9-${retriever}-${scenario}-${run_id}" -n 1x12 -m "$memory_gb" -r "$launcher"
done
