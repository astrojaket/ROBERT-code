# AGENTS.md

Guidance for coding agents working in the ROBERT repository.

## Project Intent

ROBERT is a Python package skeleton for JWST exoplanet emission retrievals. The current repository is intentionally lightweight: it defines the seams of the retrieval workflow, testable core data containers, and a stubbed end-to-end example without implementing full atmospheric physics, radiative transfer, instrument models, or samplers.

The Python distribution name is `robert-exoplanets`; avoid introducing packaging or documentation that assumes the distribution is named `robert`.

## Working Guidelines

- Keep physics-facing APIs explicit and typed so later scientific implementations can replace stubs without changing user-facing examples.
- Prefer small, well-tested modules over broad framework code.
- Do not add numerical approximations that look like real retrieval physics unless they are clearly labeled as placeholders.
- Keep examples runnable with the default development dependencies.
- Add tests for public behavior whenever you add or change an API.
- Follow `docs/rfcs/0001-robert-architectural-specification.md` and the companion docs in `docs/architecture/` for package boundaries, dependency rules, data models, plugin design, configuration, and testing philosophy.

## Repository Layout

- `src/robert_exoplanets/`: Python package source.
- `src/robert_exoplanets/core/`: Core grids, spectra, exceptions, and logging helpers.
- `src/robert_exoplanets/bodies/`: Planet and star domain objects.
- `src/robert_exoplanets/instruments/`: Observation and future instrument objects.
- `src/robert_exoplanets/retrieval/`: Retrieval configuration, model placeholders, and orchestration.
- `examples/`: Runnable example scripts.
- `tests/`: Pytest suite.

## Useful Commands

```bash
python -m pip install -e ".[dev]"
pytest
python examples/stub_emission_retrieval.py
```

## Current Scope Boundaries

The following are deliberately out of scope for the initial skeleton:

- JWST pipeline calibration products.
- Real opacity tables or chemistry.
- Radiative transfer.
- Bayesian/nested sampling.
- Instrument throughput or line spread functions.
