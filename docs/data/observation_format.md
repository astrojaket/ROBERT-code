# Portable observation format

ROBERT's portable spectral observation is a compressed NumPy archive with
schema identifier `robert-observation-v1`. The same format represents emission,
transmission, and relative-flux observations.

Required arrays are:

- `wavelength`: one-dimensional wavelength-bin centers;
- `data`: one-dimensional observed values; and
- `err`: positive symmetric one-sigma uncertainties.

Optional arrays are `wavelength_bin_edges` with length `N + 1` and a boolean
`mask` with length `N`. The archive also stores `wavelength_unit`, `flux_unit`,
`observable`, `instrument`, and JSON metadata. Arrays must have matching
lengths and strictly monotonic wavelengths.

Convert a named-column table:

```bash
python scripts/convert_observation_to_robert.py spectrum.csv observation.npz \
  --delimiter comma \
  --wavelength-column wavelength_um \
  --flux-column depth_ppm \
  --uncertainty-column error_ppm \
  --wavelength-unit micron \
  --flux-unit ppm \
  --observable transit_depth \
  --instrument JWST/NIRSpec-G395H
```

Use `--observable eclipse_depth` for emission,
`--observable transit_depth` for transmission, or
`--observable relative_flux` for a normalized spectrum. Add
`--bin-low-column` and `--bin-high-column` when the source gives bin bounds.
Otherwise ROBERT infers contiguous midpoint edges.

Python callers use:

```python
from robert_exoplanets import (
    convert_observation_table,
    load_observation_npz,
    load_observation_table,
    save_observation_npz,
)
```

Legacy NPZ archives with different array keys can be read by passing
`wavelength_key`, `flux_key`, `uncertainty_key`, units, observable, and
instrument to `load_observation_npz`.

The checked-in WASP-77Ab NIRSpec/G395H archive table can be loaded with
`load_august2023_wasp77ab`. The loader accepts the archive root or the `.tbl`
file, verifies the August et al. (2023) SHA-256 by default, converts percent
eclipse depth to a dimensionless value, and stores both published asymmetric
error columns in metadata. It returns the two detector datasets
`nirspec_g395h_nrs1` (70 points) and `nirspec_g395h_nrs2` (80 points), with the
0.119 micron detector gap kept out of either grid. Bin edges use the published
bandwidth at each segment's outer edges and midpoint boundaries inside each
segment. The metadata labels this as an `August et al. 2023 / Smith et al. 2024
input`; it is not Smith-produced data.

## Time-resolved IGRINS high-resolution data

The Smith et al. (2024) IGRINS product is a separate, external HRS format. It
is loaded with `load_smith2024_wasp77ab_hrs` into a
`TimeResolvedHighResolutionObservation`. The primary pre-eclipse product has:

- `order_wavelengths`: shape `(n_order, n_pixel)`, with monotonic wavelength
  coordinates for each order;
- `flux` and `mask`: shape `(n_order, n_frame, n_pixel)`;
- `phase`, `time_bjd`, and `fixed_velocity_km_s`: one value per frame; and
- optional `airmass`, `humidity_percent`, and `median_snr` arrays, plus source,
  checksum, and velocity-frame metadata.

For the verified `cube14v3.pic` product, the shape is `(44, 79, 1848)` and
the local source SHA-256 is
`999623a7ba4460aa72ba9bbe284de92e7b5a811ebe09e51d70667084ae592b26`.
The matching `20201214_info.csv` SHA-256 is
`68f5a80f930e5f1f99cd1861678754f824bf665972d675f617a7d66c4fdad11e`. Both
files are checked against the official Zenodo size and MD5 values before a
science run. The `.pic` file is an opaque external input; the acquisition
helper does not deserialize it.

The public cube is already aligned, normalised, order-cleaned, and trimmed by
the Smith reduction. Smith discarded eight orders and 200 low-throughput edge
pixels, then saved the reduced cube, a four-component SVD scaling matrix, and
frame times. The ROBERT adapter preserves the order/frame/pixel axes and
recomputes the configured order-wise SVD when it injects a model. The primary
pre-eclipse comparison uses four components. Post-eclipse sensitivity work
uses three components as the preferred case and records 2, 3, 4, 6, and 8 as
alternatives.

The deterministic HRS response is Gray rotation at `v sin(i) = 4.2 km/s`,
then a Gaussian IGRINS LSF at `R = 45,000`, then interpolation onto the real
cube pixels. The local HRS validator does not pixel-bin this model. The
separate real-grid LBL benchmark also tests pixel-bin integration on the 19
complete orders within 1.90--2.45 micron.

The current public HRS product has no published covariance matrix used by the
adapter. Named orders and frames are therefore independent likelihood terms.
This is an explicit assumption, not a declaration that cross-order covariance
is zero. If a covariance or PCA/SysRem filter is added, apply the same linear
operator to data and model, propagate `C_out = F C_in F.T`, and carry the
retained covariance rank into the likelihood degrees of freedom. Never replace
such a covariance silently with independent diagonal errors.

ROBERT abundances associated with this HRS format are VMRs or `log10(VMR)`.
Mass fractions belong only to a deterministic external pRT boundary
conversion and are not part of the ROBERT observation or retrieval state.
