#!/usr/bin/env bash
# Install the three Stage-9 runtime environments on Glamdring.
# Run this script only on Glamdring after reviewing the paths below.
set -euo pipefail

if [[ "${HOSTNAME:-}" != *"glamdring"* ]]; then
  echo "Refusing to install: this Stage-9 installer is for Glamdring only." >&2
  exit 2
fi
if [[ $# -ne 2 ]]; then
  echo "usage: $0 REPOSITORY_ROOT ENVIRONMENT_PARENT" >&2
  exit 2
fi

repository_root="$(cd "$1" && pwd)"
environment_parent="$(mkdir -p "$2" && cd "$2" && pwd)"
if command -v mamba >/dev/null 2>&1; then
  environment_solver=mamba
elif command -v conda >/dev/null 2>&1; then
  environment_solver=conda
else
  echo "Neither mamba nor conda is available; load the Glamdring Conda module first." >&2
  exit 2
fi

robert_prefix="$environment_parent/robert-stage9"
picaso_prefix="$environment_parent/robert-stage9-picaso-v4"
prt_prefix="$environment_parent/robert-stage9-petitradtrans"

cd "$repository_root"
"$environment_solver" env create --prefix "$robert_prefix" --file "$repository_root/environment.yml"
"$environment_solver" env create --prefix "$picaso_prefix" --file "$repository_root/environment-stage9-picaso-v4.yml"
"$environment_solver" env create --prefix "$prt_prefix" --file "$repository_root/environment-stage9-petitradtrans.yml"

for prefix in "$robert_prefix" "$picaso_prefix" "$prt_prefix"; do
  "$environment_solver" run --prefix "$prefix" python -m pip install --no-deps -e "$repository_root"
done

for prefix in "$robert_prefix" "$picaso_prefix" "$prt_prefix"; do
  "$environment_solver" run --prefix "$prefix" python - <<'PY'
import importlib.metadata
import json
packages = {}
for name in ("robert-exoplanets", "numpy", "mpi4py", "pymultinest", "picaso", "petitRADTRANS"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        pass
print(json.dumps(packages, sort_keys=True))
PY
done

echo "Stage-9 environments installed. This script does not stage opacity/reference data."
