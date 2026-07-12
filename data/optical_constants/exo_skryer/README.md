# Exo Skryer optical-constant catalogue

This directory contains a verbatim snapshot of the `nk_data/*.txt` tables from
[`ELeeAstro/Exo_Skryer`](https://github.com/ELeeAstro/Exo_Skryer), pinned at
commit `abde13e1e39753f0528e271227decec503211c64` (retrieved 2026-07-12).

The snapshot contains 45 source tables. ROBERT's `OpticalConstantsCatalog`
exposes 44 physical/reference materials by name; it deliberately hides the
artificial `White` pseudo-species. `Vacuum` remains available as a numerical
reference. Several source tables join measurements at repeated wavelengths;
the ROBERT reader records and averages those duplicate rows at load time while
leaving these source files unchanged. The `TiO2.txt` declared row count differs
from its numeric row count by one; that mismatch is retained and recorded in
loaded metadata rather than silently rewriting the table.

Each table's header identifies the material state and literature/laboratory
source where supplied. Scientific use must cite those original measurements,
not only Exo Skryer or ROBERT. The source repository is AGPL-3.0; its licence is
included as `LICENSE-AGPL-3.0.txt`. These files are separated from ROBERT's
MIT-licensed Python source and are not declared as Python package data.

Example:

```python
from robert_exoplanets import OpticalConstantsCatalog

catalog = OpticalConstantsCatalog("data/optical_constants/exo_skryer")
print(catalog.materials)
mg_silicate = catalog.load("MgSiO3")
```
