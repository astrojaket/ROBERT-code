# ROBERT emission-observation format

ROBERT's portable observation product is a compressed NumPy archive (`.npz`)
with schema identifier `robert-emission-observation-v1`. It contains these
required arrays:

- `wavelength`: one-dimensional wavelength-bin centres, in micron.
- `data`: one-dimensional fractional eclipse depths.
- `err`: positive symmetric 1-sigma uncertainties in fractional eclipse depth.

Optional arrays are `wavelength_bin_edges` (length `N + 1`) and `mask` (boolean,
length `N`). The writer also records units, observable, instrument, and JSON
metadata. Arrays must have matching lengths and wavelength must be monotonic.

Convert a named-column table with:

```bash
python scripts/convert_observation_to_robert.py unity_spectrum.csv unity.npz \
  --delimiter comma \
  --wavelength-column wavelength_um \
  --flux-column eclipse_ppm \
  --uncertainty-column error_ppm \
  --wavelength-unit micron \
  --flux-unit ppm \
  --instrument JWST/NIRSpec-G395H
```

Add `--bin-low-column` and `--bin-high-column` when the published product gives
bin bounds. Without them, ROBERT infers contiguous edges midway between bin
centres. Use `--help` for every supported input unit and option.

Python callers can use `load_emission_observation_table`,
`save_emission_observation_npz`, and `convert_emission_observation_table` from
`robert_exoplanets`.
