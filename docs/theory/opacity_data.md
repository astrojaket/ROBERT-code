# Opacity Data

ROBERT treats opacity data as a prepared scientific input, not as something
radiative-transfer kernels discover from file paths. The opacity package now
separates:

- source database, such as ExoMol, ExoMolOP, HITRAN, HITEMP, or an unknown
  upstream source,
- storage format, such as ExoMol line-list files, HITRAN `.par`, HITRAN CIA,
  `.kta` binary tables, or a future ROBERT compressed archive,
- numerical mode, such as correlated-k, opacity sampling, line-by-line, or CIA,
- spectral, pressure, temperature, species, and checksum metadata.

This matters because the same physical source can appear in several storage
formats. ROBERT's bundled R=100 k-tables are derived from ExoMolOP R=1000
tables and binned with `exo_k`; the on-disk products are `.kta` correlated-k
binary tables. ROBERT records them as:

```python
source = "exomol_op"
storage_format = "kta_binary"
mode = "correlated_k"
```

## Current Scope

ROBERT includes R=100 molecular k-tables for H2O, CO, CO2, CH4, NH3, and HCN
over 0.3–15 microns. These compact tables are the default for quick forward
models and resolution-appropriate data such as HST/WFC3. They retain all 22
pressure points, all 27 temperature points, and eight g-points. Their source
URLs, input and output checksums, transformation settings, attribution, and
licence are recorded in
checked-in [provenance record](../../src/robert_exoplanets/data/opacities/R100/provenance.json),
which is also used by the downloader.
The [ExoMol data index](https://exomol.com/data/) provides the upstream
molecule and dataset navigation.

The opacity implementation can:

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
- load ExoMolOP/TauREx pressure-temperature cross-section HDF5 grids with
  `OpacitySamplingProvider.from_exomol_paths`, retain an explicit shared subset
  of physical wavelength samples, and interpolate log cross section in
  log-pressure and temperature,
- prepare either correlated-k or opacity-sampling providers behind the same
  forward-model contract; sampled species optical depths are summed directly
  and never passed through random overlap,
- optionally clamp pressure and temperature to the nearest native boundary with
  the explicit `log_pressure_temperature_log_k_clip` comparison policy,
- bin KTA-backed correlated-k distributions onto explicit observation bins
  with `exo_k.Ktable.bin_down_cp`, preserving correlated-k recompression rather
  than wavelength-interpolating individual coefficients,
- discover arbitrary molecular KTA products in an ExoMol/exo_k directory with
  `CorrelatedKOpacityProvider.from_exomol_kta_directory`, while requiring an
  explicit path when multiple resolutions or isotopologues map to one species,
- load other precomputed correlated-k formats supported by exo_k, including
  ExoMol HDF5 products, with `CorrelatedKOpacityProvider.from_exok_paths`,
- run a local HAT-P-32b opacity benchmark that checks exact evaluator slices
  against native `.kta` values, records non-finite k-coefficient locations, and
  plots wavelength and pressure-temperature opacity diagnostics with the same
  explicit runtime floor policy available to future RT calls,
- inspect HITRAN `.par` files enough to infer line-center coverage,
- inspect HITRAN CIA headers enough to infer pair, spectral range, and
  temperature range,
- read a ROBERT `.npz` archive manifest without loading large arrays,
- validate whether an atmosphere request is covered by known metadata.

It does not invent an independent wavelength interpolator for correlated-k
tables. Off-grid preparation requires explicit bin edges and the optional
`exo_k` dependency. CIA evaluation and radiative transfer are implemented as
separate RT-layer components with explicit coverage policies.

Strict pressure/temperature coverage remains the default. The clipping policy
exists to reproduce NemesisPy's documented implementation behavior at table
boundaries and is included in prepared-opacity cache identity and provenance.

## Reusing external R=1000 tables

R=1000 tables are large external inputs. Download them once into a shared
source collection, such as `/shared/ROBERT-data/ktables_exomol`, and point each
run to that collection. The path may be absolute and read-only. Keep the run
directory, prepared opacity cache, outputs, and scratch files outside the
checkout, for example under `/shared/ROBERT-runs/wasp69b-clear`.

For a new collection, use one directory for all selected gases:

```text
/shared/ROBERT-data/ktables_exomol/
└── R1000/
    ├── CO2_R1000.kta
    ├── H2O_R1000.kta
    ├── H2S_R1000.kta
    ├── SO2_R1000.kta
    └── ...
```

`CorrelatedKOpacityProvider.from_exomol_kta_directory` first uses
`<root>/R1000` when that directory exists. An existing flat collection is also
valid when it has no `R1000/` child, and a path ending in `R1000` is valid. Do
not split one run between these layouts: once `<root>/R1000` exists, files in
the flat root are not searched. Every selected file must end in
`_<resolution>.kta`; for R=1000 the canonical names are `H2O_R1000.kta`,
`SO2_R1000.kta`, and so on. The loader takes the species name from the text
before the first underscore, so save or rename long upstream ExoMol names to
these canonical names.

Use the shared source root in each configuration. In a generated external run
directory, keep the project path local to that run:

```yaml
paths:
  project_directory: .
  k_table_directory: /shared/ROBERT-data/ktables_exomol

opacity:
  format: exomol_kta
  resolution: R1000
```

`--prepare-opacity` reads the source files and writes derived tables and a
manifest to the configured opacity cache, normally below
`project_directory/opacity_cache/R1000`. It does not modify the shared source
files. A new cache is needed when species, resolution, target grid, or other
preparation settings change.

The six tables supported by `robert-opacity-download` have the following
public source files. The URLs are the ExoMolOP NEMESIS
R=1000 products. The exact SHA-256 values and source metadata are in the
checked-in [provenance record](../../src/robert_exoplanets/data/opacities/R100/provenance.json),
which is also used by the downloader.

| Species | Save as | ExoMolOP R=1000 KTA |
| --- | --- | --- |
| H2O | `H2O_R1000.kta` | [download](https://www.exomol.com/db/H2O/1H2-16O/POKAZATEL/1H2-16O__POKAZATEL__R1000_0.3-50mu.ktable.NEMESIS.kta) |
| CO | `CO_R1000.kta` | [download](https://www.exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.R1000_0.3-50mu.ktable.NEMESIS.kta) |
| CO2 | `CO2_R1000.kta` | [download](https://www.exomol.com/db/CO2/12C-16O2/UCL-4000/12C-16O2__UCL-4000.R1000_0.3-50mu.ktable.NEMESIS.kta) |
| CH4 | `CH4_R1000.kta` | [download](https://www.exomol.com/db/CH4/12C-1H4/YT34to10/12C-1H4__YT34to10.R1000_0.3-50mu.ktable.NEMESIS.kta) |
| NH3 | `NH3_R1000.kta` | [download](https://www.exomol.com/db/NH3/14N-1H3/CoYuTe/14N-1H3__CoYuTe.R1000_0.3-50mu.ktable.NEMESIS.kta) |
| HCN | `HCN_R1000.kta` | [download](https://www.exomol.com/db/HCN/1H-12C-14N/Harris/1H-12C-14N__Harris.R1000_0.3-50mu.ktable.NEMESIS.kta) |

Run the checksum-pinned downloader for these six files:

```bash
conda run -n robert-exoplanets robert-opacity-download \
  --directory /shared/ROBERT-data/ktables_exomol
```

The downloader creates `R1000/` when the selected root is not already named
`R1000`. If an existing collection is flat, use it as a flat collection or
use a clean source root for this command. Running the command in a flat root
creates a partial `R1000/` directory and can hide the other flat files from
the loader.

The maintained hot-Jupiter configurations also use SO2, and the L 98-59 b CLR
configuration uses SO2 and H2S. These two additional R=1000 products are not
included in the downloader. Their public ExoMol source pages and direct KTA
files are:

| Species | Save as | Source page | Direct KTA file | Public release check |
| --- | --- | --- | --- | --- |
| SO2 | `SO2_R1000.kta` | [ExoAmes](https://www.exomol.com/data/molecules/SO2/32S-16O2/ExoAmes/) | [download](https://www.exomol.com/db/SO2/32S-16O2/ExoAmes/32S-16O2__ExoAmes.R1000_0.3-50mu.ktable.NEMESIS.kta) | [Zenodo 5716834](https://zenodo.org/records/5716834) MD5 `f2e4ec8cc8bcd5cf8310f276527def0a`; SHA-256 `dcbde137f1b7e9ed8bb79e0a72a964acfef9b8bfd1163dd2c705227da80a088d` |
| H2S | `H2S_R1000.kta` | [AYT2](https://www.exomol.com/data/molecules/H2S/1H2-32S/) | [download](https://www.exomol.com/db/H2S/1H2-32S/AYT2/1H2-32S__AYT2.R1000_0.3-50mu.ktable.NEMESIS.kta) | [Zenodo 5716825](https://zenodo.org/records/5716825) MD5 `83eee902df6f71334561153bdeab78c1`; SHA-256 `b2d7107e60c1c2bd7c69853975616d143d3c4812b6ffc32b3f06566e0ffb47dc` |

The ExoMol/Zenodo records provide MD5 for these large files. The SHA-256
values above were measured from files that match those published MD5 values. Compute and record a
SHA-256 value after every download. Save the files with the canonical names in
the same selected directory as the six downloader files.
For example:

```bash
mkdir -p /shared/ROBERT-data/ktables_exomol/R1000
curl -L --fail \
  --output /shared/ROBERT-data/ktables_exomol/R1000/SO2_R1000.kta \
  'https://www.exomol.com/db/SO2/32S-16O2/ExoAmes/32S-16O2__ExoAmes.R1000_0.3-50mu.ktable.NEMESIS.kta'
shasum -a 256 /shared/ROBERT-data/ktables_exomol/R1000/SO2_R1000.kta
```

## Selecting opacity sampling

Opacity sampling is currently a **beta** backend: it works and is covered by
numerical parity tests, but its sampling strategy is not yet validated broadly
enough for production retrievals. Correlated-k remains the default.

Downloaded cross sections stay outside Git in a shared source collection such
as `/shared/ROBERT-data/exomol_xsec`. R=15000 ExoMolOP cross sections are a
separate opacity-sampling input; they are not R=1000 KTA files and are not
true line-by-line data. A provider and an explicit sampling grid can be
constructed with:

```python
from pathlib import Path

from robert_exoplanets import OpacitySamplingProvider

species = ("H2O", "CO", "CO2", "CH4", "NH3", "HCN")
root = Path("/shared/ROBERT-data/exomol_xsec")
provider = OpacitySamplingProvider.from_exomol_paths(
    {name: root / f"{name}.h5" for name in species}
)
spectral_grid = provider.native_spectral_grid(
    sampling=3,
    wavelength_bounds_micron=(1.0, 12.0),
)
```

Pass this provider and grid to the same emission factory used for correlated-k.
`ExoMolOpacitySamplingSource` is the equivalent typed factory source. Exo-k
binning is ignored for this provider because its grid is already an explicit
selection of physical samples. The default strict interpolation policy raises
outside the ExoMol pressure or temperature grid; append `_clip` to the provider
interpolation policy only for a documented comparison requiring boundary
clamping.

The public downloader for this backend obtains the same six species and writes
`<species>.h5` files:

```bash
conda run -n robert-exoplanets python \
  examples/download_exomol_opacity_sampling.py \
  /shared/ROBERT-data/exomol_xsec
```

Set `opacity.format: exomol_cross_section_hdf` and point
`paths.k_table_directory` to `/shared/ROBERT-data/exomol_xsec` when using these
files in a configured run. That configured path converts the HDF5 cross
sections to correlated-k tables during preparation; it is not native opacity
sampling. Use the `OpacitySamplingProvider.from_exomol_paths` example above for
the native sampling API. In both cases, keep the source collection shared and
read-only and write derived data to the run's cache.

### True line-by-line inputs

The R=15000 files above use the ExoMolOP/TauREx HDF5 format and the ROBERT
opacity-sampling provider. They must not be renamed as petitRADTRANS
line-by-line files. For true petitRADTRANS line-by-line work, follow the
[petitRADTRANS high-resolution guide](https://petitradtrans.readthedocs.io/en/latest/content/notebooks/high_resolution_spectra.html)
and its [opacity input guide](https://petitradtrans.readthedocs.io/en/latest/content/adding_opacities.html).
The standard files are R=1,000,000 `.xsec.petitRADTRANS.h5` files under the
petitRADTRANS input-data tree, for example:

```text
input_data/opacities/lines/line_by_line/H2O/1H2-16O/
└── 1H2-16O__POKAZATEL.R1e6_0.3-28mu.xsec.petitRADTRANS.h5
input_data/opacities/lines/line_by_line/CO/12C-16O/
└── 12C-16O__HITEMP.R1e6_0.3-28mu.xsec.petitRADTRANS.h5
```

The optional independent petitRADTRANS oracle defaults to a narrow H2O file.
Give it a shared input-data root, override H2O with the full-range file shown
above, and write its result outside the checkout:

```bash
conda run -n petitradtrans-stable python \
  examples/run_petitradtrans3_lbl_kband_reference.py \
  --input-data /shared/ROBERT-data/petitRADTRANS/input_data \
  --h2o-table opacities/lines/line_by_line/H2O/1H2-16O/1H2-16O__POKAZATEL.R1e6_0.3-28mu.xsec.petitRADTRANS.h5 \
  --co-table opacities/lines/line_by_line/CO/12C-16O/12C-16O__HITEMP.R1e6_0.3-28mu.xsec.petitRADTRANS.h5 \
  --output /shared/ROBERT-runs/lbl-reference/reference.npz
```

The LBL examples in this repository expect these external files and do not
download them. Keep the input-data root shared and read-only, and record the
source paths and checksums with the resulting oracle. A ROBERT
`LineByLineOpacityProvider` can read the same pRT files directly; running the
independent pRT oracle is optional.

For observed spectra, `StratifiedSamplingObservationResponse` can select a
fixed number of native opacity samples inside every observation bin and then
average the resulting spectrum back onto those bins. This is deterministic and
keeps the model grid tied to the data, but it is an opt-in convergence tool—not
a validated replacement for correlated-k. On the WASP-69b native bins, sparse
sampling was faster only at errors larger than the observational uncertainty;
the sampling density needed to approach the full-grid result removed the speed
advantage. See `docs/review/30_wasp69b_fast_rt_and_sampling.md`.

Incomplete precomputed molecular tables may contain NaN/Inf values or exact
zeros where absorption was unavailable. ROBERT applies the caller-selected
non-finite policy first. During spectral preparation it then uses
`exo_k.Ktable.remove_zeros` before `bin_down_cp`, allowing exo_k to choose its
standard small positive zero replacement. Both replacement counts and floor
values are retained in provenance metadata.

The loaders read precomputed ExoMol/ExoMolOP correlated-k products for any
molecule; they are not raw ExoMol line-list calculators. Raw `.states`/`.trans`
line lists must first be converted to cross sections or k-tables by a dedicated
line-opacity tool or supported exo_k workflow.

## Native ROBERT Archive Candidates

The `.kta` binary format is useful for compatibility, especially for
ExoMolOP/exo_k products, but it is not a good long-term ROBERT working format:
it is opaque, difficult to inspect, and awkward to benchmark without
specialized readers.

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

Correlated-k remains the validated default for retrieval workflows. ExoMolOP
cross-section opacity sampling is now an alternative provider using the same
RT-facing contract. The sampling stride is an explicit accuracy/performance
choice: on the current six-species benchmark, retaining every third R=15,000
point (effective R about 5,000) is close to correlated-k accuracy at R=100 and
remains faster, while every fifteenth point is much faster but materially less
accurate. The WASP-69b native-bin benchmark demonstrates that this conclusion
does not automatically transfer to a different bin layout. The benchmark must
be repeated for each target wavelength range and data resolution before
changing a retrieval default.

The next opacity increments should be:

- validated `.kta` benchmark conversion for the HAT-P-32b ExoMolOP/exo_k files,
- source-specific provenance checks for incomplete `.kta` products traced back
  to ExoMol, HITRAN, or other upstream databases,
- HITRAN CIA table reader for H2-H2 and H2-He first,
- ROBERT native archive conversion from validated external opacity inputs,
- prepared-opacity cache keys that include checksums, coverage, quadrature, and
  source-table identity,
- broader validation of `exo_k` binning settings against trusted spectra,
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
