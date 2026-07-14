# PHOENIX stellar-spectrum implementation and G-star benchmark

## Decision

ROBERT now uses an interpolated STScI PHOENIX stellar-atmosphere spectrum by
default for emission contrast. A blackbody is retained as an explicit model
choice. Stellar spectra are prepared once on each forward-model grid and are
not read or interpolated inside likelihood calls.

The implementation uses `stsynphot.grid_to_spec("phoenix", T_eff, [M/H],
log(g))`. `PYSYN_CDBS` must point at an STScI Synphot reference-data root
containing `grid/phoenix/catalog.fits`. The maintained STScI packages are
`synphot` and `stsynphot`; legacy `pysynphot.Icat` is not used.

## Conventions

- The STScI PHOENIX FITS tables provide wavelength in Angstrom and surface
  flux in FLAM (`erg s-1 cm-2 Angstrom-1`).
- The finite-grid bolometric flux is normalized to `sigma * T_eff**4`, as in
  petitRADTRANS's PHOENIX table path.
- Surface flux is divided by pi to obtain the disk-averaged stellar radiance
  used with ROBERT's disk-integrated planetary radiance. This is also the
  convention visible in petitRADTRANS's stellar-irradiation implementation.
- When target bin edges exist, the native PHOENIX spectrum is integrated over
  each bin. Point evaluation is used only for grids without edges.
- Requested wavelengths outside the PHOENIX atlas raise a coverage error;
  extrapolation is never silent.

## PICASO and petitRADTRANS comparison

PICASO's documented stellar workflow delegates atmosphere-grid selection to a
Synphot catalog using effective temperature, metallicity, and log(g), then
rebins the stellar flux to its opacity wavenumber grid. Current PICASO
documentation recommends PHOENIX for climate calculations; the locally
installed older implementation still calls legacy `pysynphot.Icat`, while the
current STScI replacement is `stsynphot.grid_to_spec`.

petitRADTRANS uses a packaged HDF5 PHOENIX table, interpolates in log effective
temperature, normalizes the interpolated spectrum to `sigma * T_eff**4`,
rebins to the requested wavelengths, divides surface flux by pi, and caches
prepared stellar spectra. Its table does not expose the full STScI
temperature/metallicity/log(g) interpolation used here.

ROBERT adopts the common preparation, rebinning, bolometric normalization, and
`F/pi` conventions, while retaining the full stellar-parameter dependence of
the STScI atlas and explicit immutable `Spectrum` metadata.

## Sun-like G-star benchmark

`examples/benchmark_g_star_stellar_spectrum.py` was run with the supplied
1.7 GB atlas at
`/Users/jaketaylor/Dropbox/NemesisPy-Docker/grp/redcat/trds/grid/phoenix`.
The benchmark used `T_eff=5778 K`, `log(g)=4.44`, `[M/H]=0`, 1000 logarithmic
bins from 0.3 to 20 micron, and a 1500 K Jupiter-radius planet around a
solar-radius star.

Results:

- PHOENIX preparation, including first atlas load and binning: 4.78 s;
- bolometric normalization factor: 1.027561;
- PHOENIX/blackbody range over 0.5-15 micron: 0.84554-1.25319;
- RMS fractional profile difference over 0.5-15 micron: 0.10667;
- median absolute eclipse-depth change over 2.4-12 micron: 122.29 ppm;
- maximum absolute eclipse-depth change over 2.4-12 micron: 333.73 ppm at
  11.906 micron.

These differences are large relative to JWST eclipse uncertainties and confirm
that a blackbody stellar denominator is not an adequate default even for a
Sun-like G star.
