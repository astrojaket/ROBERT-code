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
python -m pip install -e ".[dev,perf,opacity,retrieval]"
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

The HAT-P-32b observation benchmark is bundled under
`examples/data/hat_p_32b/reference/`.

ROBERT reads this with:

```python
from robert_exoplanets import load_emission_observation_npz

observation = load_emission_observation_npz(
    "examples/data/hat_p_32b/reference/quench_study_emission_G395H_spectra_band.npz",
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

The RT template calls ROBERT's emission RT solver with HAT-P-32b k-tables. The
observation bin edges are passed to `exo_k`, which bins and recompresses the
correlated-k distributions before inference. ROBERT does not interpolate
individual k coefficients in wavelength.

The planet, star, temperature profile, molecular opacity source, binning
settings, and RT choices live in the ordinary Python configuration file
`examples/hat_p_32b_config.py`. The retrieval script passes that typed config
to ROBERT's public `build_clear_sky_emission_model` factory. Copy the config
file—not the retrieval implementation—when adding another target.

Optimal estimation:

```bash
python examples/retrieve_hat_p_32b_rt.py --method optimal_estimation
```

UltraNest with 3 local MPI ranks:

```bash
/opt/homebrew/bin/mpiexec -n 3 python examples/retrieve_hat_p_32b_rt.py \
  --method ultranest \
  --live-points 40 \
  --max-ncalls 10000 \
  --dlogz 1.5
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

Every call to `run_retrieval` requires an output directory. ROBERT writes
`manifest.json` before inference starts, followed by a stable `result.json` and
`result_arrays.npz`. The manifest records priors, likelihood settings, opacity
checksums, the configuration hash, code/runtime provenance, and the random seed
for nested sampling.

A recorded single-process HAT-P-32b run with 40 live points, a 10,000-call
ceiling, `dlogz=1.5`, and seed `20260710` converged before the call limit. It
reported best-fit log likelihood `-89.3702`, log evidence `-99.8347 ± 0.6391`,
and configuration hash
`e3dbb7e05ff765e462255049d518fd2120c635f0cef7d6ceee157c1c2fbde1f4`.
These settings are a validated example, not a universal convergence guarantee;
every changed model and prior must be checked independently.

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

## Injection-Recovery Validation

ROBERT includes a multi-gas HAT-P-32b-like injection-recovery case:

```bash
python examples/validate_hat_p_32b_injection_recovery.py \
  --method optimal_estimation
```

The case uses H2O, CO2, and NH3 correlated-k tables, explicit synthetic
wavelength-bin edges, seeded Gaussian noise, and the same exo_k plus emission
RT path used by retrievals. The output report records injected and recovered
parameters, absolute tolerances, posterior-standard-deviation errors, reduced
chi-square bounds, opacity identifiers, the retrieval config hash, and whether
inference converged. Passing requires all parameter tolerances, the fit-quality
criterion, and inference convergence simultaneously.

For a sampler validation run:

```bash
python examples/validate_hat_p_32b_injection_recovery.py \
  --method ultranest \
  --live-points 200 \
  --max-ncalls 20000 \
  --dlogz 0.5
```

ROBERT sanitizes non-finite source coefficients using the explicitly configured
runtime floor, then delegates zero-coefficient replacement to
`exo_k.Ktable.remove_zeros` before correlated-k recompression. Replacement
counts and the resulting zero floor are recorded in table metadata.

## Python Retrieval-Run Configuration

`RetrievalRunConfig` binds an observation, prepared model, ordered priors,
likelihood, output directory and either `OptimalEstimationRunConfig` or
`UltraNestRunConfig`. `run_configured_retrieval` builds the problem, writes its
manifest, runs inference and writes stable results.

The FastChem/Madhusudhan-Seager comparison lives in
`examples/hat_p_32b_fastchem_config.py`. It retrieves `metallicity` (`[M/H]`,
dex), `CtoO`, and the Madhu parameters `P1`, `P2`, `P3`, `T0`, `alpha1`, and
`alpha2`.

Inspect the saved reference MAP without running inference:

```bash
python examples/compare_hat_p_32b_fastchem_retrieval.py
```

Run a short diagnostic retrieval:

```bash
python examples/compare_hat_p_32b_fastchem_retrieval.py \
  --method optimal_estimation --run-retrieval
```

Run the configured production sampler, preferably under MPI:

```bash
mpiexec -n 3 python examples/compare_hat_p_32b_fastchem_retrieval.py \
  --method ultranest --run-retrieval \
  --live-points 400 --max-ncalls 100000 --dlogz 0.5
```

Before matching CIA and layering, ROBERT obtained chi-square `145.15` and a
`25.0 ppm` RMS difference from the saved reference MAP spectrum. ROBERT now
vendors the NemesisPy v1.0.1 CIA table and uses the same 100 logarithmic
pressure points from `100` to `1e-6 bar`. It also exposes the
NemesisPy-compatible pressure/temperature boundary-clipping policy explicitly
while retaining strict coverage as the default. The reference-MAP comparison
therefore improves to chi-square `140.82` and `18.79 ppm` RMS; the external run
reports MAP chi-square `134.37`.

Pressure-grid settings are ordinary Python arguments, making layer-count
speed/accuracy studies possible without changing RT code:

```python
config = make_model_config(
    pressure_top_bar=1.0e-6,
    pressure_bottom_bar=100.0,
    n_layers=100,
)
```

Generate spectrum/residual, P–T, VMR, and parameter comparison plots with:

```bash
python examples/plot_hat_p_32b_fastchem_comparison.py \
  --robert-result-dir examples/outputs/hat_p_32b_fastchem_comparison/optimal_estimation
```

## Next Development Steps

1. Run a fully converged high-live-point UltraNest comparison on HPC.
2. Benchmark layer-count speed and spectral/retrieval accuracy.
3. Add multi-observation likelihood support for multiple JWST modes.
4. Add resumable checkpoint status to the existing run manifest contract.
5. Add proper covariance and multi-instrument calibration parameter blocks.
