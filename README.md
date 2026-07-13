# ROBERT

ROBERT is a JWST exoplanet emission retrieval code with typed core domain
objects, modular atmosphere and chemistry components, opacity import/archive
helpers, RT-facing optical-depth objects, cloud-free and cloudy emission solvers,
optimal estimation, and an optional UltraNest adapter. The physical forward
model is ready for controlled science analyses within the validated cloud-free
and benchmarked cloudy-emission regimes described below. It is not yet a
general-purpose production retrieval: calibrated pipeline-product ingestion,
broader atmospheric parameterizations, independent end-to-end cloudy
validation, and long-run posterior validation remain planned.

The Python distribution name is `robert-exoplanets` to avoid colliding with the existing `robert` package on PyPI.

## Python Support

ROBERT supports Python 3.10 through 3.14. The reproducible full environment
uses Python 3.12, while CI tests every supported version.

## Architecture

The project architecture is governed by [RFC-0001: ROBERT Architectural Specification](docs/rfcs/0001-robert-architectural-specification.md), with companion documents listed in [docs/architecture](docs/architecture/README.md). Future substantial contributions should follow that document suite.

## Quick Start

```bash
conda env create --file environment.yml
conda activate robert-exoplanets
pytest
python examples/plot_blackbody_reference.py
python examples/plot_synthetic_tau_weighting.py
python examples/plot_cloud_scattering_reference.py
python examples/benchmark_cloud_scattering_picaso_virga.py
```

For an already-created environment, refresh the editable install with:

```bash
python -m pip install -e ".[dev]"
```

Install all optional runtime integrations without Conda with:

```bash
python -m pip install -e ".[complete]"
```

For the runnable Jupyter examples, include the notebook environment:

```bash
python -m pip install -e ".[dev,notebooks,opacity,retrieval]"
```

Science runs use a strict, versioned YAML file rather than an edited Python
workflow. Validate the supplied WASP-69b NIRCam+MIRI example without loading
its external data or opacity:

```bash
python run_retrieval.py \
  --config configurations/wasp69b_cloud_free_R1000.yaml \
  --validate-only
```

See [Configuring and running ROBERT](docs/configuration.md) for portable paths,
directory initialization, opacity preparation, forward modelling, retrievals,
and 64-rank Slurm submission.

The maintained [forward-model benchmark suite](examples/BENCHMARKS.md) uses
PICASO and petitRADTRANS as independent gold-standard comparisons. Superseded
pre-YAML checks are retained under `examples/Depreciated_Benchmarks/` for
historical reference only.

The plotting example writes local figures under `examples/outputs/`, which is
ignored by git.

## What Exists Today

- Strict, versioned YAML configuration for reproducible forward-model and
  retrieval runs, backed by typed planet, star, atmosphere, observation, and
  instrument data models.
- Correlated-k opacity preparation and evaluation for ExoMolOP/exo_k and
  ROBERT-native archives, including multi-gas mixing, CIA, Rayleigh scattering,
  spectral rebinning, interpolation, coverage checks, and provenance tracking.
- Configurable pressure-temperature profiles and equilibrium or prescribed
  chemistry, evaluated consistently within each likelihood call.
- Cloud-free and cloudy thermal-emission radiative transfer, including disk
  integration, hydrostatic geometry, Mie cloud optics, Toon two-stream, and
  SH4 multiple-scattering solvers.
- Shared-atmosphere, multi-instrument forward modelling with instrument-aware
  binning and Gaussian likelihood support for masks, offsets, and jitter.
- Optimal-estimation and optional UltraNest inference through a common
  retrieval interface, with MPI-compatible execution for larger runs.
- Versioned run manifests and portable result products containing the inputs,
  opacity identifiers, code and runtime provenance, settings, seeds, spectra,
  and inference outputs.
- Unit, integration, regression, injection-recovery, and scientific benchmark
  tests, including independent forward-model comparisons with pRT and PICASO.

## What Comes Later

- Calibrated JWST pipeline-product ingestion and multi-instrument covariance.
- Production atmospheric parameterizations and broader sampler support.
- End-to-end PICASO/Virga parity using independent science molecular-opacity
  databases, plus high-stream validation beyond the matched SH4 closure.
- Long-run posterior diagnostics and science-grade validation suites.
