# Bundled R=100 molecular opacity

This directory contains ROBERT's ready-to-use correlated-k tables for H2O, CO,
CO2, CH4, NH3, and HCN. The tables cover 0.3–15 microns at constant resolving
power R=100, with 22 pressure points, 27 temperature points, and eight
Gauss-Legendre ordinates.

The products were derived from the corresponding ExoMolOP R=1000 NEMESIS KTA
tables using `exo_k.Ktable.bin_down_cp`. Exact inputs, checksums, numerical
settings, output checksums, and source URLs are recorded in `provenance.json`.

The tables are intended for introductory forward modelling, rapid model
checks, and observations such as HST/WFC3 that do not require higher opacity
resolution. They are not a substitute for a convergence-tested R=1000 or
R=15000 calculation.

## Licence and attribution

ExoMol releases its data under the Creative Commons
Attribution-ShareAlike 4.0 International licence. These adapted opacity tables
are distributed under the same licence and are not covered by ROBERT's MIT
software licence.

- Licence: https://creativecommons.org/licenses/by-sa/4.0/
- ExoMol licence statement: https://www.exomol.com/data/licence/
- ExoMolOP: Chubb et al. (2021), DOI 10.1051/0004-6361/202038350

ROBERT changed the source products by truncating the wavelength coverage from
0.3–50 to 0.3–15 microns, recompressing the spectral grid from R=1000 to
R=100, and recompressing the correlated-k distributions from 20 to eight
Gauss points.
