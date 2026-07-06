# Retrieval Workflows

Date: 2026-07-06

ROBERT now has the first sampler-independent retrieval layer. The design is:

```text
Observation -> RetrievalProblem -> inference method -> result + plots
```

The same `RetrievalProblem` can be passed to optimal estimation, UltraNest, or
future samplers.

## Install Retrieval Dependencies

The optional retrieval dependencies are:

```bash
python -m pip install -e ".[dev,perf,retrieval]"
```

On this laptop, UltraNest and mpi4py were installed with:

```bash
python -m pip install ultranest mpi4py
```

The OpenMPI launcher exists at:

```bash
/opt/homebrew/bin/mpiexec
```

Use that explicit path if `mpiexec` is not on `PATH`.

## Read Observations

The HAT-P-32b observation benchmark is:

```text
/Users/jaketaylor/Dropbox/PostDoc4/Emission_Example/Retrieval_Results/HAT-P-32b/quench_study_emission_G395H_spectra_band.npz
```

ROBERT reads this with:

```python
from robert_exoplanets import load_emission_observation_npz

observation = load_emission_observation_npz(
    "/Users/jaketaylor/Dropbox/PostDoc4/Emission_Example/Retrieval_Results/HAT-P-32b/quench_study_emission_G395H_spectra_band.npz",
    instrument="JWST/NIRSpec G395H",
)
```

Default keys are:

- `wavelength`
- `data`
- `err`

Use `wavelength_key`, `flux_key`, and `uncertainty_key` if a different NPZ
uses different names.

## Fast Benchmark Retrieval

This example tests the retrieval machinery with a cheap polynomial surrogate.
It is useful for checking priors, likelihoods, UltraNest, MPI, output writing,
and plotting before spending time in RT.

Optimal estimation:

```bash
python examples/benchmark_hat_p_32b_retrieval.py --method optimal_estimation
```

UltraNest on one process:

```bash
python examples/benchmark_hat_p_32b_retrieval.py --method ultranest --live-points 80 --max-ncalls 2000
```

UltraNest with 3 local MPI ranks:

```bash
/opt/homebrew/bin/mpiexec -n 3 python examples/benchmark_hat_p_32b_retrieval.py \
  --method ultranest \
  --live-points 80 \
  --max-ncalls 2000
```

Outputs are written to:

```text
examples/outputs/hat_p_32b_retrieval/
```

The PNG overlays the observation, the previous benchmark MAP spectrum when
present in the NPZ, and the ROBERT best fit.

## Change Priors

For the fast benchmark, priors can be changed from the command line:

```bash
python examples/benchmark_hat_p_32b_retrieval.py \
  --baseline-prior 0.0015 0.0045 \
  --slope-prior -0.004 0.004 \
  --curvature-prior -0.003 0.003
```

Inside code, priors are defined with:

```python
from robert_exoplanets import RetrievalParameter, RetrievalParameterSet, UniformPrior

parameters = RetrievalParameterSet(
    (
        RetrievalParameter("baseline", UniformPrior(1.0e-3, 5.0e-3)),
        RetrievalParameter("slope", UniformPrior(-5.0e-3, 5.0e-3)),
    )
)
```

Use `LogUniformPrior(lower, upper)` for positive scale parameters that should be
sampled logarithmically.

## Select Parameters

Each retrieval parameter must:

- have a unique name;
- have a finite prior;
- be consumed by the forward model.

To add a parameter:

1. Add it to `RetrievalParameterSet`.
2. Read it inside the forward-model callable.
3. Decide whether it is physical, nuisance, or calibration metadata.
4. Add it to plots/reports if useful.

The fast benchmark uses:

- `baseline`
- `slope`
- `curvature`

The RT template uses:

- `log_h2o`
- `temperature_offset`
- `radius_scale`

## RT-Linked Retrieval Template

The RT template calls ROBERT's emission RT solver with HAT-P-32b k-tables. It is
intended for short tests first, then longer UltraNest runs.

Optimal estimation:

```bash
python examples/retrieve_hat_p_32b_rt.py --method optimal_estimation
```

UltraNest with 3 local MPI ranks:

```bash
/opt/homebrew/bin/mpiexec -n 3 python examples/retrieve_hat_p_32b_rt.py \
  --method ultranest \
  --live-points 60 \
  --max-ncalls 600
```

Change the RT priors:

```bash
python examples/retrieve_hat_p_32b_rt.py \
  --log-h2o-prior -7 -2 \
  --temperature-offset-prior -300 300 \
  --radius-scale-prior 0.9 1.1
```

Outputs are written to:

```text
examples/outputs/hat_p_32b_rt_retrieval/
```

The first RT template is deliberately small. It uses one active gas (`H2O`), a
temperature offset, radius scaling, optional Rayleigh extinction, correlated-k
opacity, random overlap machinery, and the clear-sky emission solver. It is a
retrieval plumbing test, not yet a final HAT-P-32b science model.

## Slurm Example

Save this as `run_robert_ultranest.slurm` and edit paths/module names for the
cluster:

```bash
#!/bin/bash
#SBATCH --job-name=robert-hatp32b
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/robert-%j.out
#SBATCH --error=logs/robert-%j.err

set -euo pipefail

module purge
module load openmpi

cd /path/to/ROBERT-code
source /path/to/conda.sh
conda activate robert

python -m pip install -e ".[dev,perf,retrieval]"

mpiexec -n "${SLURM_NTASKS}" python examples/benchmark_hat_p_32b_retrieval.py \
  --method ultranest \
  --live-points 200 \
  --max-ncalls 10000
```

For the RT-linked template, replace the final command with:

```bash
mpiexec -n "${SLURM_NTASKS}" python examples/retrieve_hat_p_32b_rt.py \
  --method ultranest \
  --live-points 200 \
  --max-ncalls 10000
```

## Plot Results

Both retrieval examples write PNG plots automatically. To inspect them:

```text
examples/outputs/hat_p_32b_retrieval/hat_p_32b_optimal_estimation_retrieval.png
examples/outputs/hat_p_32b_rt_retrieval/hat_p_32b_rt_optimal_estimation_retrieval.png
```

For UltraNest, diagnostic files are also written in the `ultranest/` output
subdirectory. Corner plots can be generated from those products once we settle
the long-run output schema.

## Next Development Steps

1. Add posterior/corner plotting from UltraNest weighted samples.
2. Add multi-observation likelihood support for multiple JWST modes.
3. Move the RT template's HAT-P-32b model setup into a reusable package helper.
4. Add checkpoint manifests with git commit, opacity files, priors, sampler
   settings, and MPI size.
5. Add proper covariance/offset/jitter calibration parameter blocks.
