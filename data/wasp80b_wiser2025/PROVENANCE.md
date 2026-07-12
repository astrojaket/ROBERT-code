# WASP-80b Wiser et al. panchromatic emission spectrum

The three ECSV tables in this directory are the unmodified fiducial Eureka!
eclipse spectra from `FinalEurekaResults.zip`, Zenodo record
[10.5281/zenodo.13146949](https://doi.org/10.5281/zenodo.13146949), downloaded
on 2026-07-12. The archive describes the NIRCam/F322W2, NIRCam/F444W, and
MIRI/LRS reductions used by Wiser et al., *A precise metallicity and
carbon-to-oxygen ratio for a warm giant exoplanet from its panchromatic JWST
emission spectrum* (PNAS, DOI 10.1073/pnas.2504085122).

Archive MD5 (published by Zenodo): `2721ca17667927ed1deda9953a9e83bb`.
Downloaded archive SHA-256:
`6d1f65959a830ae5be375b021eceb58224b6dc6390af7995dd9a5adcc1fcf07e`.

The `bin_width` column is a half-width: adjacent NIRCam bin centres are
0.015 micron apart while `bin_width` is 0.0075 micron; adjacent LRS centres are
0.25 micron apart while it is 0.125 micron. The loader therefore constructs
edges as `wavelength +/- bin_width`. ROBERT currently uses a symmetric Gaussian
likelihood, so the loader explicitly averages the published negative and
positive uncertainties and preserves both original arrays in metadata.
