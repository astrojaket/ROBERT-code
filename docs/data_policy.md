# Scientific Data and Benchmark Artifacts

ROBERT's Git repository primarily contains code and human-readable
documentation. It may also contain lightweight, open-source reference inputs
that are required for the package or maintained examples to run: published
JWST observations, optical constants, FastChem input tables, and the packaged
CIA reference table. Large opacity databases, generated figures, posterior
products, and numerical benchmark outputs must not be committed.

## Storage policy

- Source code, tests, YAML configuration, Slurm scripts, Markdown, output-free
  tutorial notebooks, and lightweight required reference inputs belong in Git.
- K-tables, cross-section databases, generated arrays, plots, chains, and full
  benchmark products belong in external storage. Compact, human-readable
  benchmark summaries may stay in Git as test oracles.
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
| NemesisPy v1.0.1 CIA table, `exocia_hitran12_200-3800K.tab` | NemesisPy v1.0.1, BSD-3-Clause | Packaged under `src/robert_exoplanets/data/cia/`. |
| WASP-69b Schlawin et al. (2024) spectrum | VizieR J/AJ/168/104; DOI 10.3847/1538-3881/ad58e0 | Versioned under `data/wasp69b_schlawin2024/`; pass a directory to the loader. |
| WASP-80b Wiser et al. spectrum | Zenodo 10.5281/zenodo.13146949 | Versioned under `data/wasp80b_wiser2025/`; pass a directory to the loader. |
| L 98-59 b Bello-Arufe et al. spectrum | Zenodo 10.5281/zenodo.14676143 | Versioned under `data/observations/`; pass a directory to the loader. |
| FastChem input tables | FastChem / PyFastChem distribution, GPL-3.0 | Versioned under `data/chemistry/fastchem/`. |
| Exo-Skryer optical constants | Exo-Skryer source catalog, AGPL-3.0 | Versioned under `data/optical_constants/exo_skryer/`. |

The observation loaders retain expected upstream checksums in code and verify
external files by default. A checksum failure should be investigated, not
disabled for a science run.

## Publishing a benchmark release

1. Run the maintained benchmark with a versioned ROBERT commit and record all
   external input versions and checksums.
2. Create a manifest containing the commit, environment, command, random seed,
   input checksums, and output checksums.
3. Deposit the manifest and benchmark outputs on Zenodo.
4. Add the resulting DOI and a short interpretation to the relevant document
   under `docs/review/`; do not copy the deposited artifacts into Git.

History rewrites remove blobs from Git hosting, but they do not replace a
scientific archive. Zenodo is the permanent, citable location for released
benchmark products.
