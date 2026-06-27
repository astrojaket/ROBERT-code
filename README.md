# ROBERT

ROBERT is a starter repository for a JWST exoplanet emission retrieval code. It currently provides the package layout, typed core domain objects, tests, and a stubbed end-to-end example. Full retrieval physics is not implemented yet.

The Python distribution name is `robert-exoplanets` to avoid colliding with the existing `robert` package on PyPI.

## Architecture

The project architecture is governed by [RFC-0001: ROBERT Architectural Specification](docs/rfcs/0001-robert-architectural-specification.md), with companion documents listed in [docs/architecture](docs/architecture/README.md). Future substantial contributions should follow that document suite.

## Quick Start

```bash
python -m pip install -e ".[dev]"
pytest
python examples/stub_emission_retrieval.py
```

## What Exists Today

- A minimal `robert_exoplanets` Python package.
- Core grid, spectrum, planet, star, and observation containers.
- Retrieval configuration containers.
- A placeholder emission model.
- A stub retrieval runner that returns deterministic mock results.
- Tests that lock in the intended skeleton behavior.

## What Comes Later

- Real JWST data ingestion.
- Atmospheric parameterization.
- Radiative transfer and opacity handling.
- Likelihood evaluation.
- Sampler integration.
