# Opacity Data

ROBERT treats opacity data as a prepared scientific input, not as something
radiative-transfer kernels discover from file paths. The opacity package now
separates:

- source database, such as ExoMol, ExoMolOP, HITRAN, HITEMP, or NEMESIS,
- storage format, such as ExoMol line-list files, HITRAN `.par`, HITRAN CIA,
  NEMESIS `.kta`, or a future ROBERT compressed archive,
- numerical mode, such as correlated-k, opacity sampling, line-by-line, or CIA,
- spectral, pressure, temperature, species, and checksum metadata.

This matters because the same physical source can appear in several storage
formats. The local HAT-P-32b k-tables are ExoMol-derived and binned with
`exo_k`, but the on-disk product is a NEMESIS-style `.kta` correlated-k file.
ROBERT records that as:

```python
source = "exomol_op"
storage_format = "nemesis_kta"
mode = "correlated_k"
```

## Current Scope

The current opacity implementation is metadata and coverage scaffolding. It can:

- describe opacity products with typed metadata,
- inspect ExoMol/ExoMolOP-style directories by suffix,
- inspect ExoMolOP or exo_k-generated `.kta` files as correlated-k products,
  including header-derived spectral, pressure, temperature, g-ordinate, and
  native-shape metadata,
- read `.kta` k-coefficients into ROBERT's native
  `(pressure, temperature, wavelength, g)` axis order,
- optionally floor non-finite `.kta` k-coefficients in memory for incomplete
  opacity products, recording replacement metadata while leaving source files
  unchanged,
- convert `.kta` products into ROBERT native archives,
- evaluate correlated-k coefficients by exact native-grid lookup for
  benchmark and validation cases,
- evaluate correlated-k coefficients with optional log-pressure, linear
  temperature, log-k interpolation while keeping spectral points on the native
  opacity grid,
- run a local HAT-P-32b opacity benchmark that checks exact evaluator slices
  against native `.kta` values, records non-finite k-coefficient locations, and
  plots wavelength and pressure-temperature opacity diagnostics with the same
  explicit runtime floor policy available to future RT calls,
- inspect HITRAN `.par` files enough to infer line-center coverage,
- inspect HITRAN CIA headers enough to infer pair, spectral range, and
  temperature range,
- read a ROBERT `.npz` archive manifest without loading large arrays,
- validate whether an atmosphere request is covered by known metadata.

It does not yet interpolate or rebin k-tables onto arbitrary off-grid spectral
grids, evaluate CIA coefficients, or perform radiative transfer.

## Native ROBERT Archive Candidates

The NEMESIS `.kta` format is useful for compatibility, especially for
ExoMolOP/exo_k products, but it is not a good long-term ROBERT working format:
it is opaque, difficult to inspect, and awkward to benchmark without specialized
readers.

ROBERT's first native fast-read candidate is therefore:

```text
*.robert-opacity/
  manifest.json
  kcoeff.npy
  pressure_bar.npy
  temperature_K.npy
  wavenumber_cm-1.npy
  g_weights.npy
```

This format keeps metadata human-readable while arrays remain native NumPy
files. It supports memory mapping, which is important for large opacity tables
and repeated retrieval calls.

ROBERT also supports `.npz` archives:

- uncompressed `.npz` is a convenient single-file exchange format,
- compressed `.npz` can reduce disk use but may read more slowly,
- neither should be assumed to be the fastest runtime format until benchmarked
  against real opacity arrays.

HDF5 or Zarr may still become the long-term default if benchmarks show a clear
benefit for chunked multi-species access, compression, or cloud/distributed
workflows. The current implementation keeps that decision open by separating
metadata, archive format, and RT-facing prepared opacity state.

## Design Direction

Correlated-k is the first target because it is the practical retrieval mode used
by the HAT-P-32b benchmark workflow. The public metadata layer already has
separate modes for opacity sampling and line-by-line so POSEIDON-style
opacity-sampling and high-resolution HITRAN/HITEMP/ExoMol line-by-line support
can be added without changing the RT-facing contract.

The next opacity increments should be:

- validated `.kta` benchmark conversion for the HAT-P-32b ExoMolOP/exo_k files,
- source-specific provenance checks for incomplete `.kta` products traced back
  to ExoMol, HITRAN, or other upstream databases,
- HITRAN CIA table reader for H2-H2 and H2-He first,
- ROBERT native archive conversion from validated external opacity inputs,
- prepared-opacity cache keys that include checksums, coverage, quadrature, and
  source-table identity,
- validated spectral-grid resampling/rebinning policy for correlated-k tables,
- opacity benchmark reports that compare absorption or k-coefficients as a
  function of wavelength, pressure, and temperature,
- NumPy reference kernels for optical-depth assembly before any compiled
  acceleration.

## Benchmarking Requirement

Opacity readers are not accepted just because they parse files. Each real reader
needs a benchmark that checks the numerical opacity values against a trusted
reference on a small grid:

- wavelength or wavenumber,
- pressure,
- temperature,
- species,
- and, for correlated-k, g-ordinate and quadrature weights.

The benchmark should report at least maximum absolute difference, maximum
relative difference, median relative difference, the location of the worst
point, and whether the result passes a documented tolerance. This will let us
verify that ExoMolOP/exo_k `.kta`, future ROBERT compressed archives, HITRAN CIA
tables, and line-by-line products preserve the expected absorption behavior
before any RT backend consumes them.
