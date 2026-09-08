#!/usr/bin/env python3
"""Create one self-contained ROBERT run directory from a YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil

import yaml

from robert_exoplanets.io.task_config import load_task_config


ROOT = Path(__file__).resolve().parents[1]
_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_SBATCH = """#!/bin/bash -l
#SBATCH --job-name={run_name}
{account_directive}
{partition_directive}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --time={walltime}
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
{mail_directives}
#SBATCH --chdir={run_directory}

set -euo pipefail

cd "{run_directory}"
source "${{ROBERT_CONDA_ROOT:-${{HOME}}/miniconda3}}/etc/profile.d/conda.sh"
conda activate "${{ROBERT_CONDA_ENV:-robert-exoplanets}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${{SLURM_TMPDIR:-{run_directory}/scratch}}/numba-${{SLURM_JOB_ID:-local}}"
export MPLCONFIGDIR="${{SLURM_TMPDIR:-{run_directory}/scratch}}/matplotlib-${{SLURM_JOB_ID:-local}}"
mkdir -p "${{NUMBA_CACHE_DIR}}" "${{MPLCONFIGDIR}}"
{stellar_setup}

mpirun -np "${{SLURM_NTASKS}}" python -u run_retrieval.py --config configuration.yaml
"""

_README = """# {run_name}

This directory is one isolated ROBERT run. It contains the exact source YAML,
the generated execution YAML, retrieval/forward runners, general post-processing
scripts, and the Slurm submission script.

`configuration.yaml` is the file to edit before preparation or submission. Its
writable paths are deliberately local to this directory:

- `outputs/` — MultiNest checkpoints and run products;
- `opacity_cache/` — K-tables prepared onto the selected observation bins; and
- `scratch/` — Numba and Matplotlib runtime files.

The input data, FastChem, and K-table paths remain the values selected in the
source configuration. `source_configuration.yaml` is the unmodified copy for
comparison.

`submit.sbatch` is a standard Slurm launcher requesting {ntasks} MPI rank(s)
across {nodes} node(s). `submit.sh` is the Oxford Glamdring `addqueue`
launcher. Submit one wrapper with `-s`; it creates one Conda MPICH/Hydra MPI
world across the cores reserved with `-n 1x{glamdring_ranks}`.

Before submitting, set `ROBERT_CONDA_ROOT` if Conda is not installed under
`$HOME/miniconda3`. Set `ROBERT_CONDA_ENV` to use a differently named
environment. PHOENIX stellar spectra also require `PYSYN_CDBS` to point to the
Synphot reference-data root above `grid/phoenix`.

```bash
python run_retrieval.py --config configuration.yaml --validate-only
python run_retrieval.py --config configuration.yaml --initialize
python run_retrieval.py --config configuration.yaml --prepare-opacity
python run_retrieval.py --config configuration.yaml --smoke-only
sbatch submit.sbatch
# Oxford Glamdring example:
export ROBERT_MPI_RANKS={glamdring_ranks}
addqueue -q redwood -s -c "48 hours" -n 1x{glamdring_ranks} -m 4 -r ./submit.sh
# Deferred completed-MultiNest-to-OE analysis uses run_oe_from_nested.py.
python postprocess_retrieval.py --config configuration.yaml
python postprocess_forward.py --config configuration.yaml
```

Do not reuse this directory for a different model, prior set, data selection,
or independent MPI launch. Create another run directory instead. On Glamdring,
`-m` is memory in GiB per reserved CPU, not total job memory. Check the text
printed by `addqueue`: a successful submission reports where program output
will be written; `Batch job submission failed` means no job was queued.
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        required=True,
        help="directory containing isolated run directories",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configurations"
            / "emission.yaml"
        ),
        help="source task YAML; its run.name becomes the folder name",
    )
    parser.add_argument("--slurm-account", help="optional Slurm account")
    parser.add_argument("--slurm-partition", help="optional Slurm partition")
    parser.add_argument("--slurm-time", default="48:00:00", help="Slurm walltime")
    parser.add_argument(
        "--slurm-tasks",
        type=int,
        help="MPI ranks; defaults to 1 for OE and 128 for nested sampling",
    )
    parser.add_argument(
        "--glamdring-ranks",
        type=int,
        help="single-node Glamdring ranks; defaults to 1 for OE and 12 otherwise",
    )
    parser.add_argument("--slurm-mail-user", help="optional notification email")
    return parser.parse_args()


def create_run_directory(
    *,
    project_dir: Path,
    source_config: Path,
    slurm_account: str | None = None,
    slurm_partition: str | None = None,
    slurm_time: str = "48:00:00",
    slurm_tasks: int | None = None,
    glamdring_ranks: int | None = None,
    slurm_mail_user: str | None = None,
) -> Path:
    """Copy a configuration and runners into an isolated named run directory."""

    source = source_config.expanduser().resolve()
    config = load_task_config(source)
    run_name = config.run.name
    if not _SAFE_RUN_NAME.fullmatch(run_name):
        raise ValueError(
            "run.name may contain only letters, numbers, '.', '_' and '-': "
            f"{run_name!r}"
        )
    run_directory = project_dir.expanduser().resolve() / run_name
    if run_directory.exists():
        raise FileExistsError(
            f"run directory already exists: {run_directory}; choose a new run.name"
        )
    run_directory.mkdir(parents=True)

    shutil.copy2(source, run_directory / "source_configuration.yaml")
    for filename in (
        "run_retrieval.py",
        "run_oe_from_nested.py",
        "run_forward.py",
        "postprocess_retrieval.py",
        "postprocess_forward.py",
        "submit.sh",
    ):
        shutil.copy2(ROOT / filename, run_directory / filename)

    generated = config.model_dump(mode="json", exclude_none=True)
    configured_paths = config.paths
    paths = (
        {}
        if configured_paths is None
        else configured_paths.model_dump(mode="json", exclude_none=True)
    )
    paths.update(
        {
            "project_directory": ".",
            "observations_directory": str(config.observations.path),
            "k_table_directory": str(config.opacity.path),
        }
    )
    if config.atmosphere.chemistry.model == "fastchem_equilibrium":
        paths["fastchem_directory"] = str(
            config.atmosphere.chemistry.fastchem_path
        )
        generated["atmosphere"]["chemistry"].pop("fastchem_path", None)
    if config.clouds.model == "mie_catalog":
        paths["optical_constants_directory"] = str(
            config.clouds.optical_constants_path
        )
        generated["clouds"].pop("optical_constants_path", None)
    for key in (
        "opacity_cache_directory",
        "output_directory",
        "scratch_directory",
    ):
        paths.pop(key, None)
    generated["observations"].pop("path", None)
    generated["opacity"].pop("path", None)
    generated["opacity"].pop("cache_directory", None)
    generated.pop("outputs", None)
    generated["runtime"].pop("scratch_directory", None)
    generated.pop("paths", None)
    generated = {
        "schema_version": generated.pop("schema_version"),
        "paths": paths,
        **generated,
    }
    execution_config = run_directory / "configuration.yaml"
    execution_config.write_text(
        yaml.safe_dump(generated, sort_keys=False), encoding="utf-8"
    )
    # Validate the generated file before declaring the directory ready.
    load_task_config(execution_config)

    if slurm_tasks is not None and slurm_tasks < 1:
        raise ValueError("slurm_tasks must be positive")
    if glamdring_ranks is not None and glamdring_ranks < 1:
        raise ValueError("glamdring_ranks must be positive")
    is_oe_only = config.sampler.engine == "optimal_estimation"
    nodes = 1
    ntasks = slurm_tasks or (1 if is_oe_only else 128)
    glamdring_processes = glamdring_ranks or (1 if is_oe_only else 12)
    ntasks_per_node = ntasks
    account_directive = (
        f"#SBATCH --account={slurm_account}" if slurm_account else ""
    )
    partition_directive = (
        f"#SBATCH --partition={slurm_partition}" if slurm_partition else ""
    )
    mail_directives = (
        f"#SBATCH --mail-user={slurm_mail_user}\n"
        "#SBATCH --mail-type=BEGIN,END,FAIL"
        if slurm_mail_user
        else ""
    )
    stellar_setup = (
        'export PYSYN_CDBS="${PYSYN_CDBS:?Set PYSYN_CDBS to the Synphot '
        'reference-data root}"'
        if config.bodies.star.spectrum_model == "phoenix"
        else ""
    )
    (run_directory / "submit.sbatch").write_text(
        _SBATCH.format(
            run_name=run_name,
            run_directory=run_directory,
            account_directive=account_directive,
            partition_directive=partition_directive,
            nodes=nodes,
            ntasks=ntasks,
            ntasks_per_node=ntasks_per_node,
            walltime=slurm_time,
            mail_directives=mail_directives,
            stellar_setup=stellar_setup,
        ),
        encoding="utf-8",
    )
    (run_directory / "README.md").write_text(
        _README.format(
            run_name=run_name,
            nodes=nodes,
            ntasks=ntasks,
            glamdring_ranks=glamdring_processes,
        ),
        encoding="utf-8",
    )
    return run_directory


def main() -> None:
    args = _parse_args()
    directory = create_run_directory(
        project_dir=args.project_dir,
        source_config=args.config,
        slurm_account=args.slurm_account,
        slurm_partition=args.slurm_partition,
        slurm_time=args.slurm_time,
        slurm_tasks=args.slurm_tasks,
        glamdring_ranks=args.glamdring_ranks,
        slurm_mail_user=args.slurm_mail_user,
    )
    print(directory)


if __name__ == "__main__":
    main()
