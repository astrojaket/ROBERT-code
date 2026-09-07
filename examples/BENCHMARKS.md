# Maintained forward-model benchmarks

This file lists the maintained checks. The measured LBL/HRS release records
are summarised in the [release manifest](../docs/data/lbl_hrs_release_manifest_20260831.json).
The release status is `complete_with_rejected_stride_2_candidate`.

## Existing benchmark families

### Stellar spectra

- `benchmark_g_star_stellar_spectrum.py`: PHOENIX and blackbody stellar
  spectra, with secondary-eclipse normalization checks.

### PICASO

- `compare_grey_cloud_rt_picaso.py`: controlled grey-cloud radiative transfer.
- `benchmark_sh4_rt.py`: Toon and SH4 scattering comparisons.
- `benchmark_end_to_end_cloud_parity.py`: cloud-property and RT assembly.
- `benchmark_official_picaso_molecular_cloud_parity.py`: official molecular
  opacity and cloudy-emission parity.

### petitRADTRANS

- `benchmark_petitradtrans3_stable.py`: stable emission and transmission.
- `benchmark_petitradtrans3_multispecies_emission.py`: multi-species emission.
- `benchmark_petitradtrans3_multispecies_transmission.py`: multi-species
  transmission.
- `benchmark_petitradtrans4.py`: pRT 4 comparison.
- `benchmark_native_hdf_emission_convergence.py`: native-grid convergence.

## LBL/HRS benchmark family

- `run_petitradtrans3_lbl_kband_reference.py`: pRT 3.3.3 K-band oracle at
  native (R=10^6). It needs external H2O and CO tables.
- `run_petitradtrans3_lbl_kband_science_grid.py`: sequential pRT oracle cases.
- `benchmark_lbl_kband_emission.py`: ROBERT tabulated LBL emission and the
  Gaussian observed response.
- `benchmark_lbl_kband_science_grid.py`: wider band, pressure, temperature,
  gravity, abundance, isolated-species, and native-stride checks.
- `benchmark_hminus_continuum.py` and `run_petitradtrans3_hminus_oracle.py`:
  physical H-minus component checks and the opt-in pRT boundary oracle.
- `combined_resolution_injection_recovery.py`: shared-atmosphere LRS/HRS
  injection recovery and the small PyMultiNest/MultiNest plumbing proof.
- `combined_resolution_full_pymultinest.py`: full four-parameter, two-seed
  PyMultiNest retrieval and stride retrieval-bias check.
- `finalize_lbl_hrs_release.py`: validates the compact release records and
  writes the final manifest atomically. It can also make the bounded stride
  summary PNG.
- `validate_wasp77ab_smith2024_hrs.py` and
  `validate_wasp77ab_smith2024_lrs.py`: deterministic checks on the real
  WASP-77Ab HRS and NIRSpec data.
- `benchmark_wasp77ab_lbl_sampling.py`: real-grid HRS/LRS LBL stride
  comparison with strict memory and thread gates.
- `run_wasp77ab_smith2024_reference_pymultinest.py`: fixed-template
  HRS-only, LRS-only, and joint reference workflow; default invocation is a
  preflight and does not start sampling.
- `run_wasp77ab_robert_joint_pymultinest.py`: shared-VMR ROBERT HRS/LRS
  workflow with H2O, CO, CIA, Rayleigh, and physical H-minus components.

The HRS response order is:

```text
Doppler → rotation → Gaussian LSF → pixel integration → fixed data filter
```

The same fixed filter is applied to data and model. For a full covariance,
use `C_out = F C_in F.T`. An order owns its mask, pixel edges, velocities,
LSF, and response/filter definition. Covariance is supplied to the likelihood;
it is not stored as an order field. Use a joint likelihood for cross-order
covariance. Use `DenseCovariance` for a positive-definite covariance. A
rank-deficient PCA or SysRem filter must use `PositiveSemidefiniteCovariance`:
it applies a Moore-Penrose solve, a pseudo-log-determinant, and support
rejection outside the retained subspace. Likelihood normalisation uses the
retained effective residual rank. A rank-zero covariance is not a valid
likelihood. Separate per-order likelihoods assume independent orders and do
not include cross-order covariance.

The narrow pRT comparison uses 2.2988--2.30378 micron. Its target has no
detector pixel edges, so the pRT acceptance response is Gaussian R=100,000
only. Pixel integration is exercised in the combined synthetic workflow and
the observation-helper tests. Do not read the pRT gate as validation of the
full pixel response.

The ROBERT model window is 2.2984--2.3042 micron. The narrow validation uses
the narrow 2.3--2.3 micron H2O table and the full 0.3--28 micron CO table. The
science-grid and combined-retrieval cases use the full 0.3--28 micron H2O and
CO tables. The combined synthetic observed window is 2.29885--2.30355
micron. Record the exact table variant and window in every benchmark record.

## Six release steps

| Step | Required benchmark | Status |
| --- | --- | --- |
| 1 | Full 4D PyMultiNest/MultiNest retrieval with two independent seeds | Pass |
| 2 | Posterior and evidence bias for native stride 1 versus candidate strides | Stride 1 approved; stride 2 rejected |
| 3 | Physical H-minus recovery in the shared LRS/HRS VMR state | Pass |
| 4 | Real multi-order preparation and full-covariance likelihood | Synthetic contract pass; real target/cross-order validation not done |
| 5 | Expanded pRT oracle grid, with VMR-to-mass-fraction boundary record | Pass for recorded grid |
| 6 | Release manifest, plots, CI, and archive checks | Pass |

The table uses measured compact JSON records. Keep the command, thresholds,
resource values, and report hash with every result.

## Measured LBL/HRS result

- Base K-band report: [JSON](../docs/data/lbl_kband_validation_20260829.json),
  SHA-256 `eaabde47bf4e9f8630f0b7882e53808e03fef302184551062d498a0826abd5fd`.
  Status pass. Peak RSS was 500,645,888 bytes. ROBERT versus pRT Gaussian
  R=100,000 RMS relative error was 0.00322002; maximum absolute relative
  error was 0.0216524.
- Science-grid report: [JSON](../docs/data/lbl_kband_science_grid_20260830.json),
  SHA-256 `f62f51904507c0bd479e8516a81ac33be01c06481de8c6e5ff495d80cd6f6537`.
  Status pass. Peak RSS was 928,579,584 bytes. Baseline RMS was 0.00325896
  (maximum 0.0210529); wider-band RMS was 0.00554096 (maximum 0.0468009);
  80-versus-160 layer RMS was 0.000118379.
- Full retrieval report: [JSON](../docs/data/combined_resolution_full_pymultinest_20260831.json),
  SHA-256 `3261d14d8ce48e5d4e6929a9ede6dcc2d413089f0b7a49c5f5704decd495a5b8`.
  The full four-parameter native recovery passed for both seeds. Native
  stride 1 is approved as the production default. Stride 2 is rejected: the
  paired signed ΔlnZ values (stride 2 minus stride 1) were -5.3645 and
  -5.4086, versus the 0.5 limit.
- H-minus report: [JSON](../docs/data/hminus_continuum_validation_20260831.json),
  SHA-256 `79b4c04181ec0c13bff3def1d0fa46c3af16159a808e4eaebfe06e637227e19c`.
  Physical H-minus LRS/HRS component checks and the two-seed `log10(H-)` VMR
  recovery passed. Peak RSS was 562,872,320 bytes.
- Observation report: [JSON](../docs/data/high_resolution_observation_pipeline_20260831.json),
  SHA-256 `9f790127e6eaf88be82555fa909334f9ab7a49ea489900e592dce1272b189c81`.
  The synthetic two-order response and covariance checks passed. This is not
  real target-data validation and does not include cross-order covariance.

## WASP-77Ab real-data comparison

The full source and method record is [review 52](../docs/review/52_wasp77ab_real_joint_retrieval.md).
The HRS source is Smith et al. (2024), [arXiv:2312.13069](https://arxiv.org/abs/2312.13069),
with supplementary files from [Zenodo 10382053](https://zenodo.org/records/10382053),
CC-BY-4.0. The LRS source is the August et al. (2023) [JWST/NIRSpec
paper](https://arxiv.org/abs/2305.07753) and its [MAST archive
record](https://doi.org/10.17909/3fmp-zj55). The source roles, official file
identifiers, and measured SHA-256 values are in
[`wasp77ab_smith2024_sources.json`](../docs/data/wasp77ab_smith2024_sources.json).

The local NIRSpec table is an August input, not a Smith-produced spectrum. It
has 150 rows and returns `nirspec_g395h_nrs1` (70 rows) and
`nirspec_g395h_nrs2` (80 rows). Their outer edges are 2.808--3.712 and
3.831--5.168 micron. The 0.119 micron detector gap remains explicit. The
public Smith text count `N=160` is retained as a documented discrepancy.

The PHOENIX stellar input for the shared-atmosphere runner is a separate
official STScI subset. Its source
manifest is [`wasp77ab_phoenix_sources.json`](../docs/data/wasp77ab_phoenix_sources.json)
and its base URL is
[`https://ssb.stsci.edu/cdbs/grid/phoenix/`](https://ssb.stsci.edu/cdbs/grid/phoenix/).
The manifest pins `catalog.fits` (1,497,600 bytes),
`phoenixm00/phoenixm00_5600.fits` (9,901,440 bytes), and
`phoenixm00/phoenixm00_5700.fits` (9,901,440 bytes) by SHA-256. The fetcher
installs them below an explicit `PYSYN_CDBS/grid/phoenix` root and the ROBERT
runner verifies all three identities before opacity preparation. Explicit
blackbody mode does not require these files. The fixed-template reference
uses the supplied Smith template stellar flux instead.

The real HRS method is order-wise SVD after the Smith preprocessing: frame
alignment, count normalisation, eight discarded orders, and 200 discarded
edge pixels per retained order. The primary pre-eclipse product removes four
principal components. The model chain is R=250,000 template, Gray rotation at
4.2 km/s, then Gaussian IGRINS LSF at R=45,000. The temporal likelihood
injects `(1 + Fp/Fs) * data_scale`, recomputes the same SVD, clips whole-order
residuals at 3 sigma, and uses the Smith non-relativistic velocity convention.
The deterministic HRS validator interpolates the convolved model onto real
cube pixels; the separate LBL benchmark also tests pixel-bin integration.

The current real-data checks are:

| Check | Status | Current result |
| --- | --- | --- |
| Smith pre-eclipse HRS validator | Pass | `(44, 79, 1848)` cube; 4 PCs; 6,423,648 active and 40,527 clipped samples; 630,226,944-byte peak RSS |
| August NIRSpec LRS validator | Pass | chi-square/N `0.7247545`; NRS1 `0.7425956`; NRS2 `0.7091435`; 211,304,448-byte peak RSS |
| Real-grid LBL stride check | Pass | selected HRS stride 2 and LRS stride 25; 1,663,959,040-byte peak RSS; one CPU; opacity below 1023 MiB |
| Fixed-template PyMultiNest reference | Pass | HRS-only, LRS-only, and joint two-seed reference measurements passed the declared evidence gate |
| Shared-VMR ROBERT PyMultiNest | Deferred to cluster | no laptop real-data sampler; current runner is an isothermal pilot |

The deterministic HRS report records raw summed CCF values of 26.0578 (full),
22.2704 (H2O-only), and 4.7053 (CO-only). Smith's map-normalised CCF SNR
anchors are 9.4, 9.2, and 3.6. The local raw sums do not claim to reproduce
those SNR values. The LBL report records HRS stride-2 RMS/max fractional
errors `4.0866e-5/3.1854e-4` and LRS stride-25 sigma RMS `0.0293389`.

ROBERT uses VMR only. The physical runner shares one atmosphere between HRS
and LRS, uses external pRT R=10^6 H2O POKAZATEL and CO HITEMP LBL tables,
NemesisPy H2--H2/H2--He CIA, and Rayleigh scattering. pRT mass fractions are
only a deterministic boundary conversion from the ROBERT VMR vector. They are
not a ROBERT state or retrieval parameters.

H-minus uses explicit H-, H, and e- VMR profiles. The bound-free cutoff is
1.6421 micron, so bound-free opacity is zero in the 1.90--5.17 micron target
window. Free-free strength depends on H and e-; the pilot fixes neutral H at
VMR 0.1 and retrieves electron VMR. H-minus VMR remains in the shared state
for shorter-wavelength LRS work but is expected to be prior-dominated in this
window. The current pilot is isothermal with 32 layers and a 2500--2900 K
temperature prior. Smith's comparison retrieval uses six non-isothermal P--T
parameters, so the pilot is not a literature reproduction.

The fixed-template reference report is
[`wasp77ab_smith2024_reference_pymultinest.json`](../docs/data/wasp77ab_smith2024_reference_pymultinest.json),
SHA-256
`7bdbf6df636ca1bb4cd9e01a87cc5898558e79142c0204305aea9368fd36f1d9`.
HRS-only, LRS-only, and joint evidence differences for seeds 24680/24681 are
`0.36144`, `0.06953`, and `0.28603`, with combined reported errors
`0.22754`, `0.09903`, and `0.07406`; all are below the `0.5` limit. Calls
were `1207/1230`, `534/547`, and `1835/2126`. Peak RSS was 1,261,125,632
bytes with the legacy maximum-three thread cap and one observed thread.
This is a fixed-template comparison, not a VMR retrieval and not an exact
Smith method reproduction.

The operator-level ROBERT injection validation is pass. The compact report is
[`wasp77ab_robert_injection_20260831.json`](../docs/data/wasp77ab_robert_injection_20260831.json),
15,080 bytes, SHA-256
`0e0efb30ba27b6a5fc0599ef6d5f8b839f13ae7693bd7a172a2a1436eed0afe9`. This is a
deterministic operator check on the real HRS/LRS data structures, not a
real-data posterior recovery and not a sampler result.

The final record uses a structured 19-order HRS noise calibration and a fixed
truth mapping. The HRS injection used fixed Gaussian noise seed `20260832`,
scaled to each prepared order's residual RMS. The LRS injection used fixed
Gaussian noise seed `20260831`; the joint case used the same HRS and LRS
products. Truth minus null log-likelihood differences were `+91.65259` (HRS-only), `+261859.82925`
(LRS-only), and `+261951.48184` (joint). The HRS truth-minus-wrong-RV result
was `+42.53990`. H-minus on/off truth differences were numerically zero or
negligible. This validates finite H-minus paths only; it is not an H-minus
detection. Bound-free opacity is zero beyond the exact `1.6421 micron` cutoff
in the target window, while free-free strength remains controlled by H and
electron VMR.

The injection passed with all seven numerical thread variables set to 6, one
MPI process, and peak RSS `1,754,120,192` bytes. This is below the strict
`1.9 GiB` limit (`2,040,109,465` bytes) and the hard `2 GiB` ceiling. Six
threads is a ceiling for new cluster jobs, not an automatic production choice.

All target sampler claims require PyMultiNest with the MultiNest backend, two
seeds 24680/24681, one MPI process, bounded live points/calls, and strict
process/opacity memory limits. The completed fixed-template reference uses
the legacy maximum-three numerical-thread cap. New shared-VMR ROBERT jobs
permit a maximum of six numerical threads, but six is only a ceiling. Choose
the production thread count only after a representative run at that setting
measures peak RSS strictly below 1.9 GiB. Full real-data HRS-only, LRS-only,
and joint ROBERT retrievals are deferred to a cluster. This is an explicit
scope decision, not a sampler pass:

```text
ROBERT_PYMULTINEST_MEASUREMENTS = DEFERRED_TO_CLUSTER
```

Use the six exact PyMultiNest/MultiNest commands in [review 52, section 7](../docs/review/52_wasp77ab_real_joint_retrieval.md#7-sampler-policy-and-cluster-deferral).
They run HRS-only, LRS-only, and joint modes with H-minus on and off, use
distinct output and report paths, set all seven numerical thread variables to
6, use one MPI process, and apply `--max-memory-gib 1.9`. Do not describe any
real-data ROBERT sampler as passed until both-seed evidence, posterior
intervals, residual statistics, input/output hashes, and resource gates are
recorded.

## Abundance rule

ROBERT uses VMR only. Use `log10_h2o_vmr`, `log10_co_vmr`, and equivalent VMR
fields in all ROBERT examples and reports. pRT may require mass fractions,
but the conversion is made only at the external oracle boundary from the same
ROBERT VMR state. Record molecular masses, background gas, species order,
normalization, and the VMR hash. Do not fit an independent pRT mass-fraction
state.

## No-external-data CI helper command

The following command uses generated fixtures only. It does not read pRT
tables or download data. CI sets every numerical thread variable to 3 before
the test process starts:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
NUMEXPR_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 NUMBA_NUM_THREADS=3 \
OMP_THREAD_LIMIT=3 python -m pytest -q \
  tests/test_hminus_continuum.py \
  tests/test_high_resolution_observation.py \
  tests/test_high_resolution_likelihoods.py \
  tests/test_correlated_likelihood.py \
  tests/test_mixed_multi_dataset_likelihood.py \
  tests/test_multi_dataset_injection.py \
  tests/test_validation_injection.py \
  tests/test_hminus_workflow.py \
  tests/test_public_high_resolution_api.py \
  tests/test_combined_resolution_full_pymultinest.py \
  tests/test_multinest.py
```

Use the `robert-exoplanets` Conda environment for local runs. The H-minus test
creates a small local HDF5 fixture. It does not need the external pRT tables.

## External pRT oracle command (authoritative)

Run the pRT oracle in the `petitradtrans-stable` Conda environment. The
external tables must exist under the ignored path shown below. Use a strict
process budget below 2 GiB. This is the only maintained pRT command; RFC 0002
links to this section.

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
NUMEXPR_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 NUMBA_NUM_THREADS=3 \
OMP_THREAD_LIMIT=3 conda run -n petitradtrans-stable python \
  examples/run_petitradtrans3_lbl_kband_science_grid.py \
  --input-data "$PWD/external_data/petitRADTRANS/input_data" \
  --output-dir "$PWD/external_data/petitRADTRANS/lbl_kband_outputs" \
  --max-memory-gib 1.9
```

The pRT grid command runs one oracle case at a time. Record the exact table
paths, checksums, licence, and VMR-to-mass-fraction boundary conversion in the
release record. The pRT and ROBERT solvers may use shared opacity tables.

## External PyMultiNest reproduction command

Run the full four-parameter retrieval only as an opt-in release job:

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

All numerical thread variables in the commands above are at most 3. Seeds,
live-point counts, and evidence settings are not thread limits. Before the
retrieval, verify that the `robert-exoplanets` environment provides
`pymultinest`, `mpi4py`, and the MultiNest shared library. This is an opt-in
external check and is not part of normal CI.

## Provenance and plots

For every release benchmark, record:

- ROBERT commit and dirty-tree state;
- Conda environment, Python/package versions, operating system, and CPU;
- exact command, seed, PyMultiNest settings, MPI process count, and all thread
  variables;
- VMR state and, for pRT, the deterministic mass-fraction conversion;
- table paths, sizes, licences, and SHA-256 values;
- peak RSS, wall time, likelihood calls, gate thresholds, and outputs;
- plot source script, plot command, input-report hash, and plot checksum.

Keep full tables, generated spectra, plots, chains, and posterior products
outside Git. Use `docs/data_policy.md` for the archive and checksum policy.
