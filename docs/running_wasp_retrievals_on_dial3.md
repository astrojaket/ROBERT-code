# Running WASP Retrievals on DIaL3

These scripts follow the DIaL3 Slurm and conda guidance. Submit them from the
ROBERT repository root. Replace `CHANGE_ME` in each script with the project
account that should be charged. The production partition is `slurm`; use the
`devel` partition only for short development jobs (maximum two hours).

All WASP-69b physics scenarios import the planet, star, and derived gravity
from `examples/wasp69b_target.py`. To start a new planet, copy that small target
module and change its `PLANET` and `STAR` definitions, then point the copied
scenario script at the new module. The new target's observations, opacity
coverage, and priors remain separate inputs that must be checked explicitly.

The scripts assume the conda environment is named `robert-exoplanets` and that
Miniconda is installed at `~/miniconda3`. If either differs, export
`ROBERT_CONDA_ENV` or `ROBERT_CONDA_ROOT` before calling `sbatch`.

## Terminal-first check

Follow the usual two-stage workflow: first activate the same environment in a
terminal and run the case's smoke check or a short resumable retrieval chunk.
Only submit the corresponding Slurm script after the likelihood is finite and
the sampler starts advancing without exceptions.

```bash
source ~/miniconda3/bin/activate robert-exoplanets

# Optimized native-mode clear case: finite/deterministic likelihood only.
python examples/retrieve_wasp69b_clear_native_modes.py --smoke-only

# Cloud cases: finite likelihood only.
python examples/retrieve_wasp69b_mie_cloud.py --cloud-mode catalog --smoke-only
python examples/retrieve_wasp69b_mie_cloud.py --cloud-mode direct-nk --smoke-only
python examples/retrieve_wasp80b_mie_cloud.py --cloud-mode catalog --smoke-only
python examples/retrieve_wasp80b_mie_cloud.py --cloud-mode direct-nk --smoke-only

# WASP-80b optimized native-mode clear case.
python examples/retrieve_wasp80b_clear_native_modes.py --smoke-only

# NIRCam clear case: starts or resumes one 10,000-call retrieval chunk.
mpiexec -n 2 python examples/retrieve_wasp69b_nircam_clear.py \
  --output retrieval_runs/wasp69b_nircam_clear
```

The native-mode clear case can also be allowed to start sampling in the
terminal by omitting `--smoke-only`. It advances its cumulative call limit in
10,000-call chunks and safely resumes the same UltraNest output.

## Submit the verified case

```bash
sbatch slurm/wasp69b_nircam_clear.sbatch
sbatch slurm/wasp69b_clear_native_modes.sbatch
sbatch slurm/wasp69b_mie_catalog.sbatch
sbatch slurm/wasp69b_mie_direct_nk.sbatch
sbatch slurm/wasp80b_nircam_clear.sbatch
sbatch slurm/wasp80b_clear_native_modes.sbatch
sbatch slurm/wasp80b_mie_catalog.sbatch
sbatch slurm/wasp80b_mie_direct_nk.sbatch
```

Monitor jobs with `squeue --me`; cancel a job with `scancel JOB_ID`. Standard
output and error are written to `<job-name>-<job-id>.out` and `.err` in the
submission directory.

The WASP-80b configuration points to the versioned fiducial Eureka! spectrum
under `data/wasp80b_wiser2025/`. Its provenance file records the Zenodo DOI,
archive hash, table hashes, bin-width convention, and symmetric treatment of
the published asymmetric uncertainties.
