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
