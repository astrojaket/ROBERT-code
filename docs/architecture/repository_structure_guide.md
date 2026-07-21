# Repository Structure Guide

This document describes the target repository layout and explains why each
directory exists.

## 1. Target Layout

```text
ROBERT-code/
  AGENTS.md
  README.md
  pyproject.toml
  CHANGELOG.md
  CITATION.cff
  LICENSE
  docs/
  examples/
  schemas/
  src/
  tests/
  benchmarks/
  scripts/
  .github/
```

The current repository is intentionally smaller. Directories should be created
when they have real content, not as empty architecture theater.

## 2. Root Files

| File | Purpose |
| --- | --- |
| `AGENTS.md` | Instructions for coding agents and contributors. |
| `README.md` | User-facing project overview and quick start. |
| `pyproject.toml` | Packaging, dependencies, build backend, tool config. |
| `CHANGELOG.md` | Release notes and migration notes. |
| `CITATION.cff` | Citation metadata. |
| `LICENSE` | License text. |

Rules:

- The distribution name is `robert-exoplanets`.
- The target import namespace is `robert_exoplanets`.
- Root files should be short and point to detailed docs.

## 3. `docs/`

Purpose:

- Store all project documentation.

Target layout:

```text
docs/
  rfcs/
  architecture/
  api/
  tutorials/
  how_to/
  theory/
  developer/
  review/
```

Subdirectory responsibilities:

| Path | Responsibility |
| --- | --- |
| `docs/rfcs/` | Accepted and proposed architectural RFCs. |
| `docs/architecture/` | Normative architecture documents. |
| `docs/api/` | Generated or curated public API docs. |
| `docs/tutorials/` | End-to-end guided examples. |
| `docs/how_to/` | Task-specific guides. |
| `docs/theory/` | Equations, assumptions, and citations. |
| `docs/developer/` | Contribution, release, and plugin authoring guidance. |
| `docs/review/` | External framework reviews and design evidence. |

Rules:

- Review docs are evidence, not final architecture, unless promoted by RFC.
- Architecture docs change in the same PR as architecture-changing code.
- Tutorials use public APIs only.

## 4. `examples/`

Purpose:

- Provide runnable examples that demonstrate public APIs.

Target layout:

```text
examples/
  configs/
  data/
  scripts/
  notebooks/
```

Rules:

- Example data must be lightweight, redistributable, and required to run a
  maintained example; large opacity inputs remain external.
- Examples must not import private modules.
- Examples must be runnable from a fresh editable install.
- Notebooks are educational, not the production API, and are committed without
  execution output.

## 5. `schemas/`

Purpose:

- Store versioned config schemas.

Rules:

- Each schema version has a stable file.
- Schema changes require migration notes.
- Generated schemas may be committed only if they are reviewed and stable.

## 6. `src/`

Purpose:

- Store installable source code.

Target layout:

```text
src/
  robert_exoplanets/
```

Rules:

- No large data files.
- No notebooks.
- No generated outputs.
- Optional backends are isolated behind adapter modules.

## 7. `tests/`

Purpose:

- Verify behavior.

Target layout:

```text
tests/
  unit/
  integration/
  regression/
  scientific/
  fixtures/
```

Rules:

- Unit tests are fast and do not need network or large data.
- Scientific tests use explicit tolerances and provenance.
- Fixtures are small and versioned.
- Slow tests are marked.

## 8. `benchmarks/`

Purpose:

- Track performance and memory.

Rules:

- Benchmarks are not correctness tests.
- Benchmarks record machine/backend metadata.
- Performance regressions are reviewed, but clarity and correctness come first.

## 9. `scripts/`

Purpose:

- Store developer utilities.

Rules:

- Scripts are not public APIs.
- Scripts should have clear help text.
- Scripts should not be required for normal package use unless documented.

## 10. `.github/workflows/`

Purpose:

- Continuous integration, docs, releases.

Required future workflows:

- Unit and integration tests.
- Lint/type checks when configured.
- Docs build.
- Optional dependency matrix.
- Release packaging check.

## 11. Files That Should Not Be Committed

- `.DS_Store`.
- `__pycache__/`.
- `.pytest_cache/`.
- Large opacity databases.
- Raw JWST products unless tiny and explicitly licensed.
- Generated retrieval outputs except deliberate golden fixtures.
- Local environment files containing paths or secrets.
