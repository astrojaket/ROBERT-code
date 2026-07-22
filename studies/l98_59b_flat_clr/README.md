# L 98-59 b flat-spectrum CLR ensemble

## Scientific question

This exploratory study tests whether a nearly flat 3--5 micron spectrum and a
symmetric centered-log-ratio (CLR) composition prior favor the heaviest
available opacity-bearing molecule. The target is **L 98-59 b**. The original
request's “L98-89b” is a naming error: ROBERT's source data and the cited
Bello-Arufe et al. (2025) paper and Zenodo record all identify L 98-59 b.

Ensemble A contains H2O, CO2, CO, H2S, and SO2. Ensemble B uses the identical
100 noise realizations but removes SO2. H2S is the omitted CLR coordinate in
both configurations. It is a physical, opacity-bearing fitted category whose
abundance is recovered by unit-sum closure. The CLR transform is symmetric, so
this omission gives H2S no privileged prior role. It is not an assumed
background gas and no phantom molecule is present.

## Preregistered constraint definition

Before inspecting retrieval results, the primary target-gas criterion is:

> The one-sided 95% posterior lower bound on VMR exceeds 1%.

Thus the primary frequencies are the number of A runs satisfying this
criterion for SO2 and the number of B runs satisfying it for CO2. Sensitivity
checks use 0.1%, 10%, and 50% VMR thresholds. These are posterior lower-bound
diagnostics, not detection claims. Central 68%/95% intervals, abundance ranks,
prior-bound proximity, and posterior probabilities above each threshold are
also retained.

## Synthetic-data convention

The generator loads the 218-point Eureka! spectrum from Zenodo record
10.5281/zenodo.14676143 with checksum verification. It uses the median transit
depth (619.782 ppm) and median quoted uncertainty (37.166 ppm). The grid has 51
equal-log-width bins with exact outer edges at 3 and 5 microns; centers are
geometric means of adjacent edges. The resulting exact center/width resolving
power is 99.8379573 in every bin (nominal R=100).

One NumPy `SeedSequence` with root seed `20260719` spawns 100 independent PCG64
streams. Each stream generates independent per-bin Gaussian noise around the
flat median. Every bin retains the 37.166 ppm uncertainty. The generated NPZ
files are small run inputs, but they are reproducible artifacts under the
ignored `outputs/` directory rather than committed opaque data.

MultiNest uses separate deterministic seeds 19000--19099 for ensemble A and
20000--20099 for ensemble B. These values respect the legacy Fortran RNG's
valid seed range; they affect sampling only, not the paired synthetic spectra.

## Retrieval scope

Both ensembles fit the CLR composition, an isothermal temperature from
100--800 K, and the 1-bar reference-radius scale from 0.85--1.15. They use the
same L 98-59 b planet/star values, real ExoMol R1000 correlated-k tables,
MultiNest 3.10 through PyMultiNest 2.12, 50 live points, and three MPI ranks.
No clouds, conventional background gas, phantom gas, detector offset, or
sampler substitution is used. These low-live-point runs are a science look for
prior/likelihood behavior and cluster-run planning, not evidence for molecules.

## Reproduction

Run from the repository's `robert-exoplanets` conda environment:

```bash
export ROBERT_K_TABLE_DIRECTORY=/path/to/ktables_exomol
conda run -n robert-exoplanets python studies/l98_59b_flat_clr/run_study.py generate
conda run -n robert-exoplanets python studies/l98_59b_flat_clr/run_study.py prepare-opacity
conda run -n robert-exoplanets python studies/l98_59b_flat_clr/run_study.py smoke
conda run -n robert-exoplanets python studies/l98_59b_flat_clr/run_study.py run
conda run -n robert-exoplanets python studies/l98_59b_flat_clr/run_study.py analyze
```

`ROBERT_K_TABLE_DIRECTORY` must point to the local ExoMol R1000 k-table
directory on each machine. The study runner writes that location into ignored,
resolved run configs, keeping the committed templates portable across local,
Glamdring, and DiRAC environments.

Resolved per-run configs, seeds, manifests, logs, MultiNest products, summaries,
and plots are written beneath `studies/l98_59b_flat_clr/outputs/`.
