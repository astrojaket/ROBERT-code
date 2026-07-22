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
#SBATCH --account=dp448
#SBATCH --partition=slurm
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=jake.taylor@physics.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --chdir={run_directory}

set -euo pipefail

cd "{run_directory}"
source "${{ROBERT_CONDA_ROOT:-${{HOME}}/miniconda3}}/etc/profile.d/conda.sh"
conda activate "${{ROBERT_CONDA_ENV:-robert-exoplanets}}"
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYSYN_CDBS="${{PYSYN_CDBS:-/scratch/dp448/dc-tayl1/grp/redcat/trds}}"

mpirun -np "${{SLURM_NTASKS}}" python -u run_retrieval.py --config configuration.yaml
"""

_README = """# {run_name}

This directory is one isolated ROBERT run. It contains the exact source YAML,
the generated execution YAML, retrieval/forward runners, general post-processing
scripts, and the Slurm submission script.

`configuration.yaml` is the file to edit before preparation or submission. Its
writable paths are deliberately local to this directory:

- `outputs/` — UltraNest checkpoints and run products;
- `opacity_cache/` — K-tables prepared onto the selected observation bins; and
- `scratch/` — Numba and Matplotlib runtime files.

The input data, FastChem, and K-table paths remain the values selected in the
source configuration. `source_configuration.yaml` is the unmodified copy for
comparison.

The submission script sets `PYSYN_CDBS` to
`/scratch/dp448/dc-tayl1/grp/redcat/trds` by default, which is the directory
above the shared `grid/phoenix` atlas. An existing exported value takes
precedence.

`submit.sbatch` is the DIaL3 launcher: it requests {ntasks} MPI rank(s) across
{nodes} node(s) and sends BEGIN, END, and FAIL notifications to
`jake.taylor@physics.ox.ac.uk`. `submit.sh` is the Glamdring `addqueue`
launcher; `addqueue` starts it once per rank, so that script deliberately does
not start another `mpiexec` layer.

```bash
python run_retrieval.py --config configuration.yaml --validate-only
python run_retrieval.py --config configuration.yaml --initialize
python run_retrieval.py --config configuration.yaml --prepare-opacity
python run_retrieval.py --config configuration.yaml --smoke-only
sbatch submit.sbatch
# On Glamdring, pass ./submit.sh to addqueue instead.
# Deferred completed-MultiNest-to-OE analysis uses run_oe_from_nested.py.
python postprocess_retrieval.py --config configuration.yaml
python postprocess_forward.py --config configuration.yaml
```

Do not reuse this directory for a different model, prior set, data selection,
or failed MPI launch. Create another run directory instead.
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
        default=ROOT / "configurations" / "wasp69b_cloud_free_R1000.yaml",
        help="source task YAML; its run.name becomes the folder name",
    )
    return parser.parse_args()


def create_run_directory(*, project_dir: Path, source_config: Path) -> Path:
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
    configured_paths = config.paths or config.housekeeping
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
    generated.pop("housekeeping", None)
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

    is_oe_only = config.sampler.engine == "optimal_estimation"
    nodes = 1
    ntasks = 1 if is_oe_only else 128
    ntasks_per_node = 1 if is_oe_only else 128
    (run_directory / "submit.sbatch").write_text(
        _SBATCH.format(
            run_name=run_name,
            run_directory=run_directory,
            nodes=nodes,
            ntasks=ntasks,
            ntasks_per_node=ntasks_per_node,
        ),
        encoding="utf-8",
    )
    (run_directory / "README.md").write_text(
        _README.format(run_name=run_name, nodes=nodes, ntasks=ntasks), encoding="utf-8"
    )
    return run_directory


def main() -> None:
    args = _parse_args()
    directory = create_run_directory(
        project_dir=args.project_dir,
        source_config=args.config,
    )
    print(directory)


if __name__ == "__main__":
    main()
