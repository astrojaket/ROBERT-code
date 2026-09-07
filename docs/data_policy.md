# Scientific Data and Benchmark Artifacts

ROBERT's Git repository primarily contains code and human-readable
documentation. It may also contain lightweight, open-source reference inputs
that are required for the package or maintained examples to run: published
observations, optical constants, FastChem input tables, the packaged CIA
reference table, and the curated R=100 molecular-opacity starter set. Large
high-resolution opacity databases, generated figures, posterior products, and
numerical benchmark outputs must not be committed.

The LBL/HRS release records are complete for the measured scope. The
[release manifest](data/lbl_hrs_release_manifest_20260831.json) is the
authoritative summary. It records native stride 1 as approved, stride 2 as a
completed rejected candidate, and the synthetic multi-order covariance
contract as passed. The synthetic contract does not validate real target data
or cross-order covariance.

## Abundance representation

ROBERT uses volume mixing ratios (VMR) only. Atmospheric chemistry, retrieval
parameters, model metadata, tests, and reports must use VMR or `log10(VMR)`.
Do not add a mass-fraction abundance state to ROBERT.

The pRT oracle may require mass fractions. Convert the already-built ROBERT VMR
state only at that external boundary. Record the molecular masses, background
gas convention, species order, normalization, and a hash of the source VMR
vector. The pRT mass fractions must be reproducible from that VMR vector. They
must not be independently tuned.

## Storage policy

- Source code, tests, YAML configuration, Slurm scripts, Markdown, output-free
  tutorial notebooks, and lightweight required reference inputs belong in Git.
- R=100 K-tables may be packaged only for the six documented starter species
  when they include upstream and derived-product checksums, exact generation
  settings, attribution, and a compatible data licence.
- R=1000 and R=15000 K-tables, cross-section databases, generated arrays,
  plots, chains, and full benchmark products belong in external storage.
  Compact, human-readable benchmark summaries may stay in Git as test oracles.
- Tests should generate compact fixtures at runtime or validate a small,
  versioned JSON/CSV acceptance summary. Checks needing full archived outputs
  must be opt-in and accept an external path.
- Benchmark releases will be deposited on Zenodo. Until a ROBERT benchmark
  record is published, documentation must link to the upstream source and must
  not invent a ROBERT DOI.

Use `external_data/`, `opacity_data/`, or another location outside the checkout
for large local assets. These directories and common scientific binary formats
are ignored by Git. Paths in YAML files are explicit and should be changed for
each machine or cluster.

## External inputs currently used by examples

| Input | Upstream record | How ROBERT locates it |
| --- | --- | --- |
| R=100 H2O, CO, CO2, CH4, NH3, and HCN k-tables, 0.3–15 microns | ExoMolOP; DOI 10.1051/0004-6361/202038350; CC BY-SA 4.0 | Packaged under `src/robert_exoplanets/data/opacities/R100/`; selected automatically when YAML requests R100 without an external path. |
| NemesisPy v1.0.1 CIA table, `exocia_hitran12_200-3800K.tab` | NemesisPy v1.0.1, BSD-3-Clause | Packaged under `src/robert_exoplanets/data/cia/`. |
| WASP-69b Schlawin et al. (2024) spectrum | VizieR J/AJ/168/104; DOI 10.3847/1538-3881/ad58e0 | Versioned under `data/wasp69b_schlawin2024/`; pass a directory to the loader. |
| WASP-80b Wiser et al. spectrum | Zenodo 10.5281/zenodo.13146949 | Versioned under `data/wasp80b_wiser2025/`; pass a directory to the loader. |
| L 98-59 b Bello-Arufe et al. spectrum | Zenodo 10.5281/zenodo.14676143 | Versioned under `data/observations/`; pass a directory to the loader. |
| WASP-77Ab August et al. (2023) JWST/NIRSpec G395H eclipse spectrum | VizieR/MAST archive, bibcode 2023ApJ...953L..24A | Versioned under `data/jwst_emission_spectra/`; `load_august2023_wasp77ab` verifies the checked-in table hash and returns NRS1/NRS2 (70/80 points), preserving the 0.119 micron detector gap. This is an August et al. 2023 / Smith et al. 2024 comparison input, not Smith-produced data. |
| WASP-77Ab Smith et al. (2024) IGRINS HRS products | [Zenodo 10382053](https://zenodo.org/records/10382053), DOI 10.5281/zenodo.10382053, CC-BY-4.0 | External files under `external_data/wasp77ab_smith2024/`; the source manifest records official size/MD5 and measured SHA-256 for seven primary files. The pre-eclipse cube is loaded as `(order, frame, pixel)` HRS data; templates are comparison inputs, not ROBERT atmospheric state. |
| STScI PHOENIX minimal subset for WASP-77Ab | [STScI PHOENIX grid](https://ssb.stsci.edu/cdbs/grid/phoenix/) | The official three-file identity record is [`wasp77ab_phoenix_sources.json`](data/wasp77ab_phoenix_sources.json). Acquire it outside Git with `examples/fetch_wasp77ab_phoenix_subset.py`; install below an explicit `PYSYN_CDBS/grid/phoenix` root and verify exact size and SHA-256 before PHOENIX use. |
| FastChem input tables | FastChem / PyFastChem distribution, GPL-3.0 | Versioned under `data/chemistry/fastchem/`. |
| Exo-Skryer optical constants | Exo-Skryer source catalog, AGPL-3.0 | Versioned under `data/optical_constants/exo_skryer/`. |
| petitRADTRANS 3 line-by-line H2O POKAZATEL and CO HITEMP tables, native (R=10^6), 0.3--28 microns | petitRADTRANS input-data release; see the [pRT high-resolution documentation](https://petitradtrans.readthedocs.io/en/latest/content/notebooks/high_resolution_spectra.html) | Shared external opacity input for the ROBERT LBL provider and the independent pRT RT oracle under `external_data/petitRADTRANS/input_data/`; ignored by Git. Convert the same ROBERT VMR state to pRT mass fractions at the boundary. Record file size, SHA-256, and upstream licence in the compact LBL benchmark records. |

The pRT line-by-line tables are multi-gigabyte files. They are not package
inputs and must not be committed, copied into generated outputs, or loaded as
a complete cross-section cube during a laptop benchmark. The maintained ROBERT
examples read only an exact wavelength hyperslab and check the configured
opacity preparation budget. Legacy LBL/HRS reports retain their maximum-three
thread policy. New WASP-77Ab ROBERT jobs permit a maximum of six numerical
threads, but the production count must be selected only after a representative
run at that setting measures peak RSS strictly below 1.9 GiB. The laptop
process memory budget is strictly below 2 GiB. The compact
records [`lbl_kband_validation_20260829.json`](data/lbl_kband_validation_20260829.json),
[`lbl_kband_science_grid_20260830.json`](data/lbl_kband_science_grid_20260830.json),
[`combined_resolution_recovery_20260830.json`](data/combined_resolution_recovery_20260830.json),
[`hminus_continuum_validation_20260831.json`](data/hminus_continuum_validation_20260831.json),
[`combined_resolution_full_pymultinest_20260831.json`](data/combined_resolution_full_pymultinest_20260831.json),
and [`high_resolution_observation_pipeline_20260831.json`](data/high_resolution_observation_pipeline_20260831.json)
retain input checksums and resource fields. Their measured statuses are
recorded in the manifest. Keep the full report hash and paired evidence
values with any archived copy.

The narrow validation uses the narrow 2.3--2.3 micron H2O file and the full
0.3--28 micron CO file. The science-grid and combined-retrieval cases use the
full 0.3--28 micron H2O and CO files. The ROBERT model window is
2.2984--2.3042 micron; the narrow pRT comparison window is 2.2988--2.30378
micron; the combined synthetic observed window is 2.29885--2.30355 micron.
Reports must record the exact table variant and window.

The ROBERT H-minus implementation uses the documented pRT/Gray bound-free and
free-free convention with explicit H-, H, and e- VMR profiles. A comparison
with the John (1988) convention is a diagnostic only. It is not a pRT
acceptance gate.

The observation loaders retain expected upstream checksums in code and verify
external files by default. A checksum failure should be investigated, not
disabled for a science run.

## WASP-77Ab real-data comparison

The [target protocol](review/52_wasp77ab_real_joint_retrieval.md) owns the
source identities, reduction method, covariance assumptions, measured results,
and exact cluster commands. Keep those details there rather than copying them
into this storage policy. The compact
[validation manifest](data/wasp77ab_validation_manifest_20260831.json) records
the measured status and supporting reports.

## WASP-77Ab measured status and cluster-deferred runs

The completed operator checks and fixed-template reference are separate from
shared-VMR atmospheric retrievals. The full real-data HRS-only, LRS-only, and
joint ROBERT sampler runs remain `DEFERRED_TO_CLUSTER`. Do not report them as
passed until both seeds, posterior intervals, residual statistics, hashes, and
resource gates are recorded. See the
[sampler policy](review/52_wasp77ab_real_joint_retrieval.md#7-sampler-policy-and-cluster-deferral).

## Publishing a benchmark release

The six LBL/HRS release gates are:

1. full four-parameter PyMultiNest/MultiNest retrieval with two seeds;
2. retrieval bias from native stride choices;
3. physical H-minus recovery in the shared VMR state;
4. real multi-order response, fixed data/model filtering, and covariance;
5. expanded pRT oracle grid with the VMR-to-mass-fraction boundary record;
6. CI, provenance, plot, and archive checks.

The measured gate status is recorded in the compact reports and the release
manifest. Do not change a gate status without changing its report, checksum,
and provenance record.

## Measured LBL/HRS release

The [release manifest](data/lbl_hrs_release_manifest_20260831.json) and
[method review](review/51_line_by_line_high_resolution_retrieval_methods.md)
own the gate results. Native stride 1 passed the synthetic retrieval gate;
stride 2 was rejected. Synthetic order/covariance checks do not validate real
target data or cross-order covariance.

The pRT and ROBERT comparison shares opacity tables. It is independent at the
solver level, not the opacity-source level. The upstream table licence is not
yet recorded; verify it before archive publication. Keep the VMR boundary and
input hashes with every result.

For each release:

1. Run the maintained benchmark with a versioned ROBERT commit and record all
   external input versions and checksums.
2. Create a manifest containing the commit, dirty-tree state, environment,
   CPU, command, random seed, PyMultiNest settings, MPI process count, all
   thread controls, VMR state, pRT conversion, input checksums, output
   checksums, peak RSS, wall time, and likelihood calls.
3. Record the plot source script, plot command, input-report hash, and plot
   checksum.
4. Deposit the manifest and benchmark outputs on Zenodo.
5. Add the resulting DOI and a short interpretation to the relevant document
   under `docs/review/`; do not copy the deposited artifacts into Git.

History rewrites remove blobs from Git hosting, but they do not replace a
scientific archive. Zenodo is the permanent, citable location for released
benchmark products.
