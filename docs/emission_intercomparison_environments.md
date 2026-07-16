# Emission intercomparison environment contract

The emission intercomparison uses three isolated Conda environments.  Do not
install PICASO or petitRADTRANS into the ROBERT environment, and do not run one
code with another code's Python interpreter.  Process isolation is part of the
benchmark contract: codes exchange only versioned NPZ/JSON inputs and outputs.

| Code | Conda environment | Reproducible definition |
| --- | --- | --- |
| ROBERT | `robert-exoplanets` | `environment.yml` |
| PICASO 3.2.2 | `picaso` | `environment-picaso.yml` |
| stable petitRADTRANS 3.3.3 | `petitradtrans-stable` | `environment-petitradtrans-stable.yml` |

Create or update the environments from the repository root:

```bash
conda env create -f environment.yml
conda env create -f environment-picaso.yml
conda env create -f environment-petitradtrans-stable.yml
```

If an environment already exists, use `conda env update --prune -f FILE`
instead.  A project-local ROBERT prefix such as `.conda` is also valid when it
was created from `environment.yml`; invoke `.conda/bin/python` explicitly so
the interpreter remains unambiguous.

## Required local data

The current work-laptop layout is:

- PICASO reference data:
  `opacity_data/picaso_official/reference_v3_2`
- PICASO opacity database:
  `opacity_data/picaso_official/reference/opacities/opacities_0.3_15_R15000.db`
- petitRADTRANS input data:
  `opacity_data/petitRADTRANS/input_data`

PICASO 3.2.2 currently uses reference-data configuration 3.2.1 and emits a
minor-version warning.  This is recorded rather than hidden.  Its environment
also pins SciPy 1.13.1 because PyMieScatt imports the removed
`scipy.integrate.trapz`, and setuptools 80.9.0 because PICASO/pysynphot still
use `pkg_resources`.

On managed machines PICASO's Numba and Matplotlib caches must be writable.  The
environment smoke test configures temporary cache directories automatically;
benchmark launchers must do the same or set `NUMBA_CACHE_DIR` and
`MPLCONFIGDIR` explicitly.

The benchmark launcher also gives the external pRT worker a private writable
`HOME` beneath the ignored output directory.  This lets pRT create its config
file without modifying the user's home directory; native runs still receive
the opacity location explicitly through `path_input_data`.

## Verification

Run each check with its own interpreter:

```bash
conda run -n robert-exoplanets python examples/check_emission_intercomparison_environment.py robert
conda run -n picaso python examples/check_emission_intercomparison_environment.py picaso
conda run -n petitradtrans-stable python examples/check_emission_intercomparison_environment.py petitradtrans
```

For the work-laptop project-local ROBERT environment, use:

```bash
.conda/bin/python examples/check_emission_intercomparison_environment.py robert
```

All ROBERT benchmark orchestration and analysis runs in `robert-exoplanets`.
Only the external PICASO worker runs in `picaso`, and only the external pRT
worker runs in `petitradtrans-stable`.  The worker metadata must record the
absolute Python executable and package version for every generated artifact.
