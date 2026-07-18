# Emission intercomparison environment contract

Version 2 uses three isolated Conda environments; the historical Version-1
PICASO environment remains installed as a fourth, non-Version-2 environment.  Do not
install PICASO or petitRADTRANS into the ROBERT environment, and do not run one
code with another code's Python interpreter.  Process isolation is part of the
benchmark contract: codes exchange only versioned NPZ/JSON inputs and outputs.

| Code | Conda environment | Reproducible definition |
| --- | --- | --- |
| ROBERT | `robert-exoplanets` | `environment.yml` |
| PICASO 4.0 (Version 2) | `picaso-v4` | Dedicated frozen environment recorded below |
| PICASO 3.2.2 (Version 1 historical) | `picaso` | `environment-picaso.yml` |
| stable petitRADTRANS 3.3.3 | `petitradtrans-stable` | `environment-petitradtrans-stable.yml` |

Create or update the repository-defined ROBERT, stable-pRT, and historical
Version-1 PICASO environments from the repository root:

```bash
conda env create -f environment.yml
conda env create -f environment-picaso.yml
conda env create -f environment-petitradtrans-stable.yml
```

If an environment already exists, use `conda env update --prune -f FILE`
instead.  Do not use `environment-picaso.yml` to replace `picaso-v4`; its exact
frozen installed versions are recorded below.  A project-local ROBERT prefix
such as `.conda` is also valid when it
was created from `environment.yml`; invoke `.conda/bin/python` explicitly so
the interpreter remains unambiguous.

## Required local data

The current Version-2 work-laptop layout is:

- PICASO 4 reference data:
  `/Users/jaketaylor/Dropbox/picaso-v4/reference`
- PICASO 4 resort-rebin molecular tables:
  `/Users/jaketaylor/Dropbox/picaso/reference/opacities/resortrebin`
- petitRADTRANS input data:
  `/Users/jaketaylor/Dropbox/ROBERT-code/opacity_data/petitRADTRANS/input_data`

The historical Version-1 layout is:

- PICASO reference data:
  `opacity_data/picaso_official/reference_v3_2`
- PICASO opacity database:
  `opacity_data/picaso_official/reference/opacities/opacities_0.3_15_R15000.db`

Version 2 uses `/opt/miniconda3/envs/picaso-v4/bin/python`, with Python 3.11.15,
PICASO 4.0, numpy 2.4.6, scipy 1.17.1, h5py 3.16.0, numba 0.66.0, virga-exo
2.0.1, astropy 8.0.1, and pandas 3.0.3.  The official reference Version 4.0 at
`/Users/jaketaylor/Dropbox/picaso-v4/reference` came from PICASO Git commit
`0369089372f748609dd0233e6de9361af31a38cf`; its three frozen reference-file
checksums are in the Version-2 common contract.

An absolute interpreter invocation does not activate Conda configuration
variables.  Before importing PICASO, every worker must explicitly set
`picaso_refdata=/Users/jaketaylor/Dropbox/picaso-v4/reference` and writable
task-local `NUMBA_CACHE_DIR=<temp>/picaso-v4-numba-cache` and
`MPLCONFIGDIR=<temp>/picaso-v4-matplotlib`.

The fixed-VMR resort-rebin correlated-k smoke loaded exactly H2O, CO, CO2, and
CH4 with 661 bins and eight k points.  Forty layers produced `taugas` shape
`(40, 661, 8)`, and all 613 native bins over 0.3--12 micron had finite positive thermal
flux with scattering, Rayleigh, and delta-Eddington disabled.  The optional
Vega-spectrum warning is harmless because Version 2 supplies an explicit
blackbody star; record it without downloading stellar grids or suppressing it.

The historical `/opt/miniconda3/envs/picaso/bin/python` remains Python 3.10.20 /
PICASO 3.2.2 for Version 1 only.  Version-2 molecular work must not use it.

On managed machines PICASO's Numba and Matplotlib caches must be writable.  The
environment smoke test configures temporary cache directories automatically;
benchmark launchers must do the same or set `NUMBA_CACHE_DIR` and
`MPLCONFIGDIR` explicitly.

The benchmark launcher also gives the external pRT worker a private writable
`HOME` beneath the ignored output directory.  This lets pRT create its config
file without modifying the user's home directory; native runs still receive
the opacity location explicitly through `path_input_data`.

## Verification

Run each Version-2 check with its own interpreter.  PICASO variables must be
set explicitly before import; `<task-temp>` must be writable:

```bash
/opt/miniconda3/envs/robert-exoplanets/bin/python examples/check_emission_intercomparison_environment.py robert
env picaso_refdata=/Users/jaketaylor/Dropbox/picaso-v4/reference \
  NUMBA_CACHE_DIR=<task-temp>/picaso-v4-numba-cache \
  MPLCONFIGDIR=<task-temp>/picaso-v4-matplotlib \
  /opt/miniconda3/envs/picaso-v4/bin/python \
  examples/check_emission_intercomparison_environment.py picaso \
  --picaso-reference /Users/jaketaylor/Dropbox/picaso-v4/reference \
  --picaso-ck-directory /Users/jaketaylor/Dropbox/picaso/reference/opacities/resortrebin
/opt/miniconda3/envs/petitradtrans-stable/bin/python examples/check_emission_intercomparison_environment.py petitradtrans
```

For the work-laptop project-local ROBERT environment, use:

```bash
.conda/bin/python examples/check_emission_intercomparison_environment.py robert
```

All ROBERT benchmark orchestration and analysis runs in `robert-exoplanets`.
Only the external PICASO worker runs in `picaso-v4`, and only the external pRT
worker runs in `petitradtrans-stable`.  The worker metadata must record the
absolute Python executable and package version for every generated artifact.

Version-2 Stage 1 is reproduced with:

```bash
PYTHONPATH=src /opt/miniconda3/envs/robert-exoplanets/bin/python \
  examples/benchmark_emission_intercomparison_v2_stage_1.py
```

Version-2 Stage 2 is reproduced with the same process isolation using:

```bash
PYTHONPATH=src /opt/miniconda3/envs/robert-exoplanets/bin/python \
  examples/benchmark_emission_intercomparison_v2_stage_2.py
```

The Stage-2 launcher sets `picaso_refdata`, `NUMBA_CACHE_DIR`, and
`MPLCONFIGDIR` before every PICASO import, uses only the `picaso-v4`
interpreter for molecular work, and records the optional-Vega and exact-zero
cloud/Rayleigh divide warnings.  Its detailed raw workers remain ignored
beneath `examples/outputs/emission_intercomparison/version_2/stage_2/`.

Version-2 Stage 3 uses the same exact three interpreters and is reproduced with:

```bash
PYTHONPATH=src /opt/miniconda3/envs/robert-exoplanets/bin/python \
  examples/benchmark_emission_intercomparison_v2_stage_3.py
```

The Stage-3 launcher crosses the exact fixed-abundance H2O/CO/CO2/CH4 mixture
with the `2 x 2` H2--H2/H2--He CIA factorial for the isothermal and PG14
non-inverted profiles on 40/80/160 cells.  It sets `picaso_refdata`,
`NUMBA_CACHE_DIR`, and `MPLCONFIGDIR` before every PICASO import.  PICASO 4.0
resort-rebin correlated-k is the only active PICASO molecular representation;
opacity sampling is retired. The
launcher records the harmless optional-Vega and exact-zero cloud/Rayleigh
divide warnings without downloading or suppressing them.

Track A exchanges identical molecular-plus-selected-CIA mean optical-depth
arrays only between the compatible ROBERT and stable-pRT paths.  PICASO's
exact-`omega0=0` native shared-tensor path remains unsupported, and stable pRT's
supported native-flux interface still exposes no layer optical-depth tensor.
Native spectra and genuinely supported vertical diagnostics are retained under
their distinct definitions.  Detailed Stage-3 workers remain ignored beneath
`examples/outputs/emission_intercomparison/version_2/stage_3/`; the report,
integrity manifest, and committed array shards are under
`docs/data/emission_intercomparison/version_2/`.  Stage 3 supplies the frozen
pRT opacity path explicitly without overriding `HOME`.

The Stage-2 launcher invokes PICASO 4.0 and stable pRT only through the exact
interpreter paths above, sets all PICASO configuration/cache variables before
import, gives pRT a private ignored worker `HOME`, and records the optional
Vega warning rather than suppressing it.

Run Stages 4--6 from the ROBERT environment after the smoke tests:

```bash
conda activate robert-exoplanets
python examples/benchmark_emission_intercomparison_stage_4.py
python examples/benchmark_emission_intercomparison_stage_5.py
python examples/benchmark_emission_intercomparison_stage_6.py
python examples/benchmark_emission_intercomparison_stage_7.py
```

The launcher supplies 40/80/160 ROBERT cells, the matching cell edges as
PICASO levels, and the ROBERT geometric cell centres as the corresponding pRT
pressure nodes.  When opacity data live outside the fresh clone, pass
`--picaso-reference`, `--picaso-database`, and `--prt-input` explicitly.

Stage 5 preserves this grid contract and adds six localized temperature
perturbations from `1e-4` to `10 bar`.  Its launcher runs a
shared-optical-depth source/RT track and a native-temperature-dependent-opacity
track.  The latter recomputes opacity separately in every framework and
finite-difference case.  The launcher records all three absolute interpreter
paths and package versions in
`docs/data/emission_intercomparison/stage_5_report.json`.

Stage 6 preserves the same 40/80/160 vertical-grid and process-isolation
contracts.  It adds localized H2O, CO, CO2, and CH4 log10-VMR perturbations,
case-specific shared optical depths for Track A, native composition-dependent
opacity/CIA recomputation for Track B, and a primary-resolution finite-
difference audit at 0.05, 0.10, and 0.20 dex.  Its absolute interpreters,
package versions, warnings, contracts, checksums, and raw/summarized timings
are recorded in `docs/data/emission_intercomparison/stage_6_report.json`.

Stage 7 retains the same interpreters and vertical grids.  It runs a
representative 80-cell resource pilot before the complete absorbing-cloud
matrix, freezes cloud and Rayleigh scattering off, and records process-tree
peak resident memory in addition to raw and summarized timings.  Track A
exchanges only identical gas/cloud optical-depth NPZ contracts.  Track B uses
ROBERT layer extinction, PICASO native cloud tables, and pRT native `cm2/g`
additional absorption callbacks.  Its report and complete R=100 cloud arrays
are written under `docs/data/emission_intercomparison/`.

The completed Stage-7 run took `3666.4 s` for the full matrix plus `88.3 s`
for defect-correction reanalysis, with `5.57 GB` peak orchestrator RSS.  The
resource pilot authorized local execution, and the actual end-to-end launcher
time including its repeated pilot was `3783.7 s`.  Exact-`omega0` PICASO
shared calculations use the independently implemented Stage-1 absorbing
formal path because PICASO's low-level scattering routine returns NaNs at
exact zero; native PICASO cloud calculations remain on the official native
path.  This distinction and the unchanged failed full-domain gates are
recorded in the report.

The canonical continuation through absorbing clouds, cloud scattering, and
cloud-free plus cloudy cross-retrievals is documented in
`docs/emission_intercomparison_roadmap.md`.  Stage-9 implementation, contracts,
synthetic inputs, and smoke tests are prepared locally; the full posterior
matrix is then pulled from GitHub and executed on Glamdring, with DiRAC kept as
a fallback target.
