# RFC 0002: line-by-line high-resolution modelling

Status: measured release record complete for the six-step LBL/HRS extension.
The release manifest is
[`lbl_hrs_release_manifest_20260831.json`](../data/lbl_hrs_release_manifest_20260831.json).
Its status is `complete_with_rejected_stride_2_candidate`: native stride 1 is
approved as the production default, and the stride-2 candidate is a recorded
negative result.

## 1. Purpose

ROBERT must use one atmosphere state for low-resolution (LRS) and
high-resolution (HRS) data. The HRS branch must keep the physical wavelength
coordinate. It must not turn line-by-line samples into a correlated-k
distribution.

The first science case is clear thermal emission from CO and H2O in the
2.2984--2.3042 micron K-band window. The external line-by-line tables have
native resolving power (R=10^6). A response chain maps the model to the
observed pixels.

Keep the wavelength scopes explicit. The ROBERT model window is
2.2984--2.3042 micron. The narrow pRT comparison uses 2.2988--2.30378
micron, with the narrow 2.3--2.3 micron H2O table and the full 0.3--28 micron
CO table. The science-grid and combined-retrieval cases use the full
0.3--28 micron H2O and CO tables. The combined synthetic observed window is
2.29885--2.30355 micron. Every report must name the table variant and window
that it used.

This RFC defines the implementation and release contract. The method survey
is in [review 51](../review/51_line_by_line_high_resolution_retrieval_methods.md).

The measured compact reports and the manifest are the release evidence. The
base K-band check has peak RSS 500,645,888 bytes and a ROBERT versus pRT
Gaussian R=100,000 RMS relative error of 0.00322002 (maximum 0.0216524).
The science-grid check has peak RSS 928,579,584 bytes: its baseline RMS is
0.00325896 (maximum 0.0210529), its wider-band RMS is 0.00554096 (maximum
0.0468009), and its 80-versus-160 layer RMS is 0.000118379. All process
ceilings are strictly below 2 GiB and all numerical thread values are at most
3.

## 2. Scientific boundary

ROBERT uses volume mixing ratios (VMR) as its only atmospheric abundance
state. Retrieval parameters, chemistry profiles, opacity providers, metadata,
and reports must use VMR or `log10(VMR)`.

The pRT oracle can require mass fractions. This is an external-boundary
conversion only:

1. Build one ROBERT atmosphere in VMR.
2. Convert that same state to pRT mass fractions with the selected molecular
   masses and background-gas convention.
3. Run pRT as an independent RT/forward solver at the oracle boundary. The
   ROBERT and pRT calculations may use the same external opacity tables; the
   solver implementation is independent, but the opacity source is shared.
4. Record the conversion rule, species order, mean molecular mass, and input
   VMR hash.

Mass fractions must not become a second ROBERT retrieval state. A mismatch
between the VMR state and the pRT conversion is a failed validation setup.

## 3. Methods that informed the design

The design uses the following primary or official sources:

| Source | Relevant method | ROBERT decision |
| --- | --- | --- |
| [petitRADTRANS high-resolution spectra](https://petitradtrans.readthedocs.io/en/latest/content/notebooks/high_resolution_spectra.html), [retrieval spectral model](https://petitradtrans.readthedocs.io/en/latest/content/notebooks/retrieval_spectral_model.html), [spectral-model API](https://petitradtrans.readthedocs.io/en/latest/autoapi/petitRADTRANS/spectral_model/index.html), and [source repository](https://gitlab.com/mauricemolli/petitRADTRANS) | Native LBL spectra, velocity shifts, convolution, rebinning, and HRS preparation. | Use pRT 3.3.3 as an independent RT/forward solver. ROBERT and pRT use shared external opacity tables, so independence refers to the solver, not the opacity source. Keep the native-grid boundary and test reduced sampling at the final observed resolution. |
| [POSEIDON HRS retrieval documentation](https://poseidon-retrievals.readthedocs.io/en/latest/content/notebooks/transmission_high_res_retrieval.html), [cross-correlation tutorial](https://poseidon-retrievals.readthedocs.io/en/stable/content/notebooks/transmission_high_res_cross_correlate.html), [gas opacity database](https://poseidon-retrievals.readthedocs.io/en/latest/content/opacity_database_gas.html), [HRS paper](https://arxiv.org/abs/2505.09933), and [repository](https://github.com/MartianColonist/POSEIDON) | LBL templates, cross-correlation retrievals, and instrument/data preparation. | Separate the native model, response, data filter, and likelihood. Keep calibrated direct flux as the first HRS likelihood. |
| [HyDRA-H](https://arxiv.org/abs/1910.14042) | Bayesian joint LRS/HRS thermal-emission retrieval with a shared atmosphere. | Use one shared atmosphere and named data-set likelihoods. |
| [Brogi & Line 2019](https://arxiv.org/abs/1811.01681) and the [combined-resolution framework](https://arxiv.org/abs/1612.07008) | Quantitative HRS likelihoods based on cross-correlation statistics and joint LRS/HRS information. | Permit CCF likelihoods only when normalization and nuisance terms are explicit. |
| [Gibson et al. 2022](https://arxiv.org/abs/2201.04025) | Apply the same stellar/telluric filter to data and every model. | Use one fixed linear data/model operator. Telluric-fitting physics is outside this observation-preparation layer. |
| [ExoJAX2](https://arxiv.org/abs/2410.06900) | Differentiable line-by-line spectra for future gradient-based work. | Keep the current typed NumPy path independent. Validate any future backend against the same native-grid oracle. |

These sources describe external software or published methods. They do not
define ROBERT's sampler, abundance convention, or release status.

POSEIDON has two relevant high-resolution paths. Its [native high-resolution
template documentation](https://poseidon-retrievals.readthedocs.io/en/latest/content/notebooks/high_res.html)
uses a fixed spacing of delta-wavenumber = 0.01 cm^-1. This is R=1e6 at
1 micron, but only about R=4.35e5 at 2.3 micron. It interpolates the
pressure-temperature state one layer at a time to limit RAM during a single
LBL template calculation. Its [HRS retrieval
tutorial](https://poseidon-retrievals.readthedocs.io/en/latest/content/notebooks/transmission_high_res_retrieval.html)
uses a constant R=250,000, `opacity_sampling`, and selected cross sections
pre-interpolated on fine pressure-temperature grids. This is a retrieval
speed and accuracy trade-off. Do not describe the native template spacing as
the general POSEIDON retrieval grid. This distinction motivates the ROBERT
posterior stride gate.

The documented molecular choices include H2O POKAZATEL and CO Li2015, with
separate CO isotopologues available. ExoMol H2.broad and He.broad data
provide H2/He broadening where available; CIA, free-free absorption, and
Rayleigh scattering are also included. The [POSEIDON source](https://github.com/MartianColonist/POSEIDON)
and [John (1988)](https://ui.adsabs.harvard.edu/abs/1988A%26A...193..189J/abstract)
are the source trail for its H-minus bound-free/free-free convention.

ROBERT remains VMR-only. The pRT oracle is an independent RT solver, but it
may read the same external H2O and CO tables as the ROBERT LBL provider. It
uses the pRT/Gray H-minus convention after the external VMR-to-mass-fraction
conversion. A John-versus-Gray comparison is a diagnostic only. It is not a
pRT acceptance gate.

## 4. Six-step extension

The earlier seven-stage plan is replaced by these six release steps. The
measured results below come from the compact reports and the [release
manifest](../data/lbl_hrs_release_manifest_20260831.json).

### Step 1: full four-parameter, two-seed PyMultiNest retrieval

Use the parameter vector below for the shared LRS/HRS injection:

```text
(temperature_K, log10_h2o_vmr, log10_co_vmr, radial_velocity_km_s)
```

Run two independent seeds with the same configuration. Use PyMultiNest with
the MultiNest backend, one MPI process, a bounded live-point count, and
`max_iter=0` for natural evidence stopping. Report, for each seed:

- convergence state and completed likelihood calls;
- posterior intervals and best-fit values for all four parameters;
- log evidence and reported evidence error;
- peak RSS, wall time, thread settings, and exact environment;
- recovery error against the injected VMR state and velocity.

Measured result: pass. The full four-parameter native recovery passed. Both
seeds converged, the fit recovered all four injected parameters, the truth was
inside the posterior intervals, and the reported evidence difference was
0.0128706, below the 0.5 limit. The full report peak RSS was 883,752,960
bytes. The one-parameter velocity proof is recorded separately and does not
replace this four-parameter result.

### Step 2: retrieval bias from native sampling stride

Fit the same noisy data with stride 1 and with each proposed production
stride. Compare the posterior centre and interval for all four parameters,
the log evidence, likelihood-call count, peak RSS, and wall time. Also keep a
forward spectral comparison after the final observed response.

Do not approve a stride from a forward RMS value alone. Measured result: the
native stride-1 reference is approved as the production default. The stride-2
candidate is rejected. Its paired signed evidence differences (stride 2 minus
stride 1) are -5.3645 and -5.4086, versus the 0.5 limit. The manifest records
this as a completed negative result. The science-grid forward stride-2 RMS
after Gaussian R=100,000 is 0.00152953 (maximum 0.00800497); this forward
result does not override the retrieval rejection.

### Step 3: physical retrievable H-minus continuum

`HMinusContinuumConfig` adds the bound-free and free-free H-minus terms. It
reads explicit `H-`, `H`, and `e-` VMR profiles. It does not calculate an
equilibrium abundance or use a Saha closure. The Gray/pRT convention and its
temperature and wavelength limits must be recorded.

The same VMR state must affect the LRS and HRS branches. The local unit gate
is covered by `tests/test_hminus_continuum.py`. Measured result: the physical
H-minus LRS/HRS component checks passed, and the two-seed PyMultiNest
`log10(H-)` VMR injection recovery passed. The absolute truth errors were
0.0009459 and 0.0009518 dex; the reduced chi-square was 0.9885 for both runs.
The process peak RSS was 562,872,320 bytes. The pRT HRS native and Gaussian
R=100,000 gates also passed.

### Step 4: real multi-order observation preparation and covariance

Each named order must carry its observation, detector mask, barycentric and
systemic velocity terms, pixel edges, LSF settings, and optional fixed linear
data/model filter. The physical response order is:

```text
Doppler → rotation → Gaussian LSF → pixel integration → fixed data filter
```

Positive radial velocity means a redshift. A runtime orbital or fitted
velocity is added to the fixed barycentric plus systemic terms before the
Doppler stage. The response preparation must support different LSFs and pixel
responses for different orders.

The order preparation object does not own a covariance block. It owns the
order data and response/filter definition. A per-order covariance is supplied
to its likelihood; a joint likelihood must own a joint covariance when
cross-order correlations are present. The same fixed matrix (F) is applied to
data and model. A full covariance
must use:

```text
C_out = F C_in F.T
```

Masks select the matching covariance rows and columns. Use
`DenseCovariance` for a positive-definite matrix. A rank-deficient PCA or
SysRem filter can produce a positive-semidefinite `F C_in F.T`; represent it
with `PositiveSemidefiniteCovariance`. This operator uses an eigenspace
Moore-Penrose solve, a pseudo-log-determinant over its supported eigenvalues,
and a support check that rejects residuals outside the retained subspace.
`CorrelatedGaussianLikelihood` uses the retained effective residual rank for
normalisation and exposes that rank. A rank-zero covariance can be represented
as a PSD operator for preparation, but it is not a valid likelihood because it
contains no data. The likelihood must never form an explicit covariance
inverse. Separate per-order likelihoods assume independent orders and do not
include cross-order covariance; a joint covariance is required to model those
correlations. The local gates are the observation, correlated-likelihood, and
public-export tests. Measured result: the synthetic two-order contract
passed. It used two distinct order responses, passed `C_out = F C_in F.T`
and correlated chi-square checks, and had peak RSS 119,734,272 bytes. This is
not real target-data validation and does not include cross-order covariance.

### Step 5: expanded pRT oracle grid

Run the independent pRT 3.3.3 RT solver for the narrow and wider K-band windows,
multiple pressure-layer counts, temperature profiles, gravities, abundances,
and isolated CO/H2O cases. Use the same ROBERT VMR state and document the
external VMR-to-mass-fraction conversion. Record table checksums and bounded
hyperslab reads.

Measured result: pass for the recorded narrow, wide, pressure, isolated
species, and native-stride oracle cases. The pRT and ROBERT solvers share the
external H2O and CO tables, so this is an independent-solver comparison, not
an independent-opacity comparison. The base and science metrics and peak RSS
are recorded near the start of this RFC and in the linked JSON reports.

### Step 6: release, CI, and provenance gates

Normal CI must use compact fixtures and no external opacity tables. It must
run the H-minus, observation, covariance, and retrieval-helper tests, plus
lint and the normal unit suite. External pRT and PyMultiNest runs are opt-in
release jobs.

Measured result: the release manifest and plot records are complete. Every
release record includes:

- ROBERT commit and dirty-tree state;
- Python and package versions, operating system, and CPU description;
- exact command, environment name, seed, and sampler settings;
- all seven thread variables and MPI process count;
- VMR parameter vector and the pRT boundary-conversion rule, if used;
- external table paths, sizes, licences, and SHA-256 values;
- peak RSS, wall time, likelihood calls, and output checksums;
- plot source, plot command, and plot-data checksum.

The manifest records the Git commit, dirty state, report hashes, resource
limits, VMR contract, PyMultiNest settings, and plot checksums.

## 5. Forward and likelihood contract

The combined model follows this sequence:

```text
shared atmosphere
  ├─ LRS: correlated-k or validated opacity sampling → RT → bins
  └─ HRS: physical LBL opacity → RT → velocity → LSF → pixels
                                               → order mask/filter → likelihood
```

The maintained pRT/ROBERT K-band comparison currently validates the Gaussian
R=100,000 response. Its target grid has no detector pixel edges, so pixel
integration is not part of that pRT acceptance gate. Pixel integration is
covered by the combined synthetic response and the observation-helper tests.

LBL species optical depths are added at each physical wavelength. The final
RT sample axis has unit weight only as an interface adapter; it is not a
correlated-k ordinate. HDF5 readers must use bounded hyperslabs and reject a
read that exceeds the configured memory budget.

The direct HRS likelihood uses calibrated flux and independent uncertainties.
Projected-continuum and cross-correlation likelihoods are allowed for data
that require a documented fixed filter. The correlated-noise likelihood uses
the full covariance after filtering, with `DenseCovariance` for positive-
definite matrices and `PositiveSemidefiniteCovariance` for rank-deficient
filtered matrices. Its Moore-Penrose solve, pseudo-log-determinant, support
rejection, and effective residual rank must be retained in the likelihood
contract. A CCF peak without normalization, amplitude treatment, velocity
grid, and nuisance terms is a diagnostic only. Separate per-order likelihoods
assume independent orders and do not include cross-order covariance.

## 6. Laptop resource contract

All numerical commands must set the thread controls before importing NumPy,
SciPy, HDF5, pRT, or PyMultiNest. No value may exceed three:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
NUMEXPR_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 NUMBA_NUM_THREADS=3 \
OMP_THREAD_LIMIT=3
```

Use the `robert-exoplanets` Conda environment for ROBERT. Use the pinned pRT
environment only for the external oracle. Before a PyMultiNest run, verify
that the ROBERT environment provides `pymultinest`, `mpi4py`, and the
MultiNest shared library. Use one MPI process for the laptop retrieval. Keep
the process budget strictly below 2 GiB and the single opacity-read budget at
1 GiB unless a measured release record defines a stricter value. Read one
window at a time. Clear prepared caches between independent cases.

## 7. Reproduction commands

The following commands are release jobs. They require external tables and
must not run in normal CI. The measured output records and checksums are
linked in the release manifest.

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
NUMEXPR_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 NUMBA_NUM_THREADS=3 \
OMP_THREAD_LIMIT=3 conda run -n robert-exoplanets python \
  examples/combined_resolution_full_pymultinest.py \
  --run-pymultinest --run-native-stride-comparison \
  --seed 24680 --pymultinest-live-points 64 \
  --pymultinest-max-iter 0 --pymultinest-evidence-tolerance 0.5 \
  --pymultinest-sampling-efficiency 0.8
```

The pRT oracle command is maintained only in the
[`examples/BENCHMARKS.md`](../../examples/BENCHMARKS.md) external-oracle
section. Do not copy large tables, generated spectra, chains, or plot products
into Git.
