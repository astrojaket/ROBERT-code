# Review 51: line-by-line and high-resolution retrieval methods

Date: 2026-08-31

Status: method review and measured release record complete.
See the [LBL/HRS release manifest](../data/lbl_hrs_release_manifest_20260831.json).
The release status is `complete_with_rejected_stride_2_candidate`.

## Decision in one minute

ROBERT uses one atmosphere state and two spectral branches:

```text
shared atmosphere
  ├─ LRS: correlated-k or validated opacity sampling → RT → bins
  └─ HRS: physical LBL grid → RT → velocity → LSF → pixels → likelihood
```

The HRS branch keeps the wavelength coordinate. It does not sort line samples
into a correlated-k distribution. The shared atmosphere uses VMR only.

The main numerical oracle is petitRADTRANS 3.3.3. It is an independent
RT/forward-solver boundary, not a ROBERT runtime dependency. ROBERT and pRT
may read the same external opacity tables; the solver is independent, but the
opacity source is shared. The external oracle may receive mass fractions,
but only after a deterministic conversion from the same ROBERT VMR state.
Mass fractions are not a second ROBERT retrieval state.

The ROBERT model window is 2.2984--2.3042 micron. The narrow pRT comparison
uses 2.2988--2.30378 micron, with the narrow 2.3--2.3 micron H2O table and
the full 0.3--28 micron CO table. The science-grid and combined-retrieval
cases use full 0.3--28 micron H2O and CO tables. The combined synthetic
observed window is 2.29885--2.30355 micron. Reports must identify the exact
table variant and window.

Measured snapshot: the base K-band check passed with peak RSS 500,645,888
bytes. Its ROBERT versus pRT Gaussian R=100,000 RMS relative error was
0.00322002, with maximum absolute relative error 0.0216524. The science-grid
check passed with peak RSS 928,579,584 bytes: baseline RMS 0.00325896
(maximum 0.0210529), wider-band RMS 0.00554096 (maximum 0.0468009), and
80-versus-160 layer RMS 0.000118379. All recorded process ceilings are
strictly below 2 GiB and all numerical thread values are at most 3.

## What the survey found

| Code or method | Primary or official source | Useful method | ROBERT adoption |
| --- | --- | --- | --- |
| petitRADTRANS | [HRS spectra tutorial](https://petitradtrans.readthedocs.io/en/latest/content/notebooks/high_resolution_spectra.html); [retrieval spectral model](https://petitradtrans.readthedocs.io/en/latest/content/notebooks/retrieval_spectral_model.html); [spectral-model API](https://petitradtrans.readthedocs.io/en/latest/autoapi/petitRADTRANS/spectral_model/index.html); [repository](https://gitlab.com/mauricemolli/petitRADTRANS) | Native LBL spectra, Doppler shifts, convolution, rebinning, and HRS preparation. | Use the native-grid boundary, explicit response chain, and final-resolution convergence check. Use pRT 3.3.3 as the independent RT/forward solver with shared external opacity tables. |
| POSEIDON | [HRS retrieval documentation](https://poseidon-retrievals.readthedocs.io/en/latest/content/notebooks/transmission_high_res_retrieval.html); [CCF tutorial](https://poseidon-retrievals.readthedocs.io/en/stable/content/notebooks/transmission_high_res_cross_correlate.html); [gas opacity database](https://poseidon-retrievals.readthedocs.io/en/latest/content/opacity_database_gas.html); [HRS paper](https://arxiv.org/abs/2505.09933); [repository](https://github.com/MartianColonist/POSEIDON) | LBL templates, CCF retrievals, and instrument/data preparation. | Separate native model, response, fixed data filter, and likelihood. Keep direct calibrated flux as the first HRS likelihood. |
| HyDRA-H | [Gandhi et al. 2019](https://arxiv.org/abs/1910.14042) | Bayesian simultaneous LRS/HRS thermal-emission retrieval with shared atmospheric parameters. | Use the shared-atmosphere joint design. Validate each ROBERT likelihood separately. |
| Brogi & Line | [Brogi & Line 2019](https://arxiv.org/abs/1811.01681); [combined-resolution framework](https://arxiv.org/abs/1612.07008) | Quantitative CCF likelihoods and combined LRS/HRS constraints. | Use CCFs only with explicit normalization, amplitude, velocity grid, and nuisance terms. |
| Gibson et al. | [Gibson et al. 2022](https://arxiv.org/abs/2201.04025) | Apply the same stellar/telluric filter to data and every forward model. | Use one fixed linear data/model operator. Telluric-fitting physics is outside the observation-preparation layer. |
| ExoJAX2 | [ExoJAX2 paper](https://arxiv.org/abs/2410.06900) | Differentiable line-level emission, transmission, and reflection spectra. | Keep the typed NumPy path independent. Validate a future differentiable backend against the same oracle. |

The table records method sources. It does not state that the corresponding
external validations have passed.

### POSEIDON opacity detail

POSEIDON has two relevant high-resolution paths. Its [native high-resolution
template documentation](https://poseidon-retrievals.readthedocs.io/en/latest/content/notebooks/high_res.html)
uses a fixed spacing of delta-wavenumber = 0.01 cm^-1. This is R=1e6 at
1 micron, but only about R=4.35e5 at 2.3 micron. It interpolates the
pressure-temperature state one layer at a time to limit resident memory for a
single LBL template. Its [HRS retrieval tutorial](https://poseidon-retrievals.readthedocs.io/en/latest/content/notebooks/transmission_high_res_retrieval.html)
uses constant R=250,000, `opacity_sampling`, and selected cross sections
pre-interpolated on fine pressure-temperature grids. This is a retrieval
speed and accuracy trade-off. Native template spacing is not the general HRS
retrieval grid. This distinction supports the ROBERT posterior stride gate.

For the relevant molecules, POSEIDON uses H2O POKAZATEL and CO Li2015. CO may
use a terrestrial-isotope weighted mixture or separate isotopologues. Its
pressure broadening uses ExoMol H2.broad and He.broad data where available.
CIA, free-free absorption, and Rayleigh scattering are included by the
opacity database. The [POSEIDON source repository](https://github.com/MartianColonist/POSEIDON)
and [John (1988)](https://ui.adsabs.harvard.edu/abs/1988A%26A...193..189J/abstract)
provide the source trail for the John H-minus bound-free/free-free convention.

This is a method comparison, not a claim that ROBERT uses the POSEIDON tables.
ROBERT uses VMR only and reads the external pRT LBL tables for both its LBL
provider and the pRT oracle. pRT remains an independent RT solver, not an
independent opacity source. The pRT boundary uses the pRT/Gray H-minus
convention. A John-versus-Gray comparison is a diagnostic only. It is not a
pRT acceptance gate.

## ROBERT abundance and oracle boundary

Use VMR or `log10(VMR)` in all ROBERT configuration and retrieval fields. Do
not add a mass-fraction retrieval parameter. For a pRT comparison, convert
the already-built ROBERT VMR state at the external boundary. Record:

- the VMR vector and its hash;
- molecular masses and background-gas convention;
- the species order and normalized pRT mass fractions;
- pRT version, table version, and table checksums.

The pRT mass-fraction vector must be reproducible from the VMR vector. It must
not be tuned independently to improve agreement.

## Six-step extension

### 1. Full 4D, two-seed PyMultiNest

Run PyMultiNest with the MultiNest backend for:

```text
temperature_K, log10_h2o_vmr, log10_co_vmr, radial_velocity_km_s
```

Use the same data, priors, live-point count, evidence settings, one MPI
process, and two independent seeds. Use `max_iter=0` for natural MultiNest
stopping. Record recovery, posterior intervals, evidence and error, completed
likelihood calls, peak RSS, wall time, and all thread variables. A previous
one-parameter proof does not close this four-parameter gate.

Measured status: pass. Both PyMultiNest seeds converged and recovered the
four injected parameters. The reported evidence difference was 0.0128706,
below the 0.5 limit. Peak RSS was 883,752,960 bytes.

### 2. Retrieval bias from native stride

Fit the noisy injection with stride 1 and each candidate stride. Compare all
four posterior centres and intervals, evidence, likelihood calls, RSS, wall
time, and final-response spectral error. Set and record thresholds before
calling a stride a production default.

Measured status: stride 1 is approved as the production default. Stride 2 is
rejected as a completed negative result. The paired signed evidence
differences (stride 2 minus stride 1) were -5.3645 and -5.4086, versus the
0.5 limit. The science-grid forward stride-2 RMS after Gaussian R=100,000
was 0.00152953 (maximum 0.00800497), but this does not override the retrieval
rejection.

### 3. Physical retrievable H-minus continuum

The H-minus path uses explicit H-minus, neutral-H, and electron VMR profiles.
It evaluates bound-free and free-free terms using the documented Gray/pRT
convention. It has no Saha or other equilibrium closure. The same VMR state
must affect LRS and HRS.

The local helper gate is `tests/test_hminus_continuum.py`. Measured status:
the physical H-minus LRS/HRS component checks passed, and the two-seed
PyMultiNest `log10(H-)` VMR injection recovery passed. The absolute truth
errors were 0.0009459 and 0.0009518 dex; the reduced chi-square was 0.9885
for both runs. Process peak RSS was 562,872,320 bytes.

### 4. Real multi-order preparation and correlated noise

Each named order owns its observation, mask, velocity bookkeeping, pixel
edges, LSF, and optional fixed filter. The prepared response is:

```text
Doppler → rotation → Gaussian LSF → pixel integration → fixed F
```

Positive velocity is a redshift. Runtime orbital velocity is added to fixed
barycentric and systemic velocity before the Doppler stage. Different orders
may use different LSFs and pixel responses.

The order preparation object owns the order data and response/filter
definition. It does not own a covariance block. Attach a per-order covariance
to its likelihood, or use a joint likelihood for cross-order covariance.
Apply the same matrix F to data and model. Propagate a full covariance as
`C_out = F C_in F.T`. Apply order masks to matching covariance rows and
columns. Use `DenseCovariance` for positive-definite matrices. A
rank-deficient PCA or SysRem filter produces a positive-semidefinite matrix;
represent it with `PositiveSemidefiniteCovariance`. It uses a
Moore-Penrose solve on the supported eigenspace, a pseudo-log-determinant over
the retained eigenvalues, and support rejection for residuals outside that
subspace. The likelihood normalisation uses the retained effective residual
rank, exposed by `effective_residual_rank`; a rank-zero covariance is not a
valid likelihood. Do not form an explicit inverse in the likelihood loop.
Separate per-order likelihoods assume independent orders and do not include
cross-order covariance; use a joint covariance when cross-order noise is
part of the data model.

Local helper tests cover H-minus, named orders, combined masks, pixel edges,
runtime velocity, the same filter for data/model, covariance propagation, and
likelihood normalization. Measured status: the synthetic two-order contract
passed. It used distinct order responses, passed `C_out = F C_in F.T` and
correlated chi-square checks, and had peak RSS 119,734,272 bytes. It is not
real target-data validation and does not include cross-order covariance.

### 5. Expanded pRT oracle grid

Run pRT 3.3.3 as an independent RT solver for the narrow and wider K-band windows, several pressure-layer
counts, temperature profiles, gravities, abundances, and isolated CO/H2O
cases. Use the same VMR state and record the external mass-fraction
conversion. Use bounded hyperslab reads for the multi-gigabyte LBL tables.

Measured status: pass for the recorded narrow, wide, pressure, isolated
species, and native-stride pRT oracle cases. The pRT and ROBERT solvers share
the external H2O and CO tables, so this is an independent-solver comparison,
not an independent-opacity comparison.

### 6. Release, CI, and provenance

Normal CI must use compact generated fixtures and no external pRT tables. It
must run the new H-minus, observation, covariance, and retrieval-helper tests.
External oracle and PyMultiNest jobs are opt-in release jobs.

Measured status: documentation, CI scaffolding, compact report records, and
the release manifest are complete. The manifest records the report hashes,
resource limits, VMR contract, PyMultiNest settings, and plot checksums.

## Resource contract

Use no more than three numerical threads. Set the variables before importing
numerical libraries:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
NUMEXPR_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 NUMBA_NUM_THREADS=3 \
OMP_THREAD_LIMIT=3
```

Use the `robert-exoplanets` Conda environment for ROBERT and one MPI process
for a laptop PyMultiNest run. Before the run, verify that `pymultinest`,
`mpi4py`, and the MultiNest shared library are available in that environment.
Keep the process budget strictly below 2 GiB and the opacity-read budget at
1 GiB or less. Read one bounded LBL window at a time.

The maintained pRT/ROBERT comparison validates the Gaussian R=100,000
response. Its target grid has no detector pixel edges, so pixel integration is
not part of the pRT acceptance gate. Pixel integration is covered by the
combined synthetic response and observation-helper tests.

## Provenance and plot contract

Every release record must identify the ROBERT commit, dirty-tree state,
environment versions, CPU and operating system, command, seed, PyMultiNest
settings, MPI process count, thread settings, VMR state, pRT conversion,
table checksums, memory, wall time, likelihood calls, and output checksums.

Each plot must record its source script, command, input report or data hash,
software versions, and output checksum. Do not commit generated chains,
spectra, plots, or multi-gigabyte tables. Store released products in the
external archive named by `docs/data_policy.md`.

## Current result status

The compact JSON records in `docs/data/` now contain the measured results. The
full four-parameter native retrieval passed. Native stride 1 is approved as
the production default; stride 2 is rejected because its paired signed
evidence differences were -5.3645 and -5.4086, versus the 0.5 limit. The
physical H-minus LRS/HRS checks and the VMR injection recovery passed.

The synthetic multi-order observation and covariance contract passed, but it
does not validate real target data or cross-order covariance. Treat those as
the next science validation scope. The release manifest is the authoritative
summary of report hashes, resource limits, sampler records, and plot checksums.
