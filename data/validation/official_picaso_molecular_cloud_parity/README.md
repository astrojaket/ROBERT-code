# Official PICASO Molecular-Opacity Cloud Reference

This directory contains the compact, versioned reference for the independently
evaluated ROBERT versus PICASO/Virga cloud experiment using the official PICASO
opacity database, Zenodo DOI `10.5281/zenodo.14861730`.

The reference calculation uses 72 atmospheric layers, 192 cloud-optics
wavelengths, 36 particle-radius bins, all 37,273-37,275 native opacity samples
from 1-12 micron, and 240 common output bins (R approximately 97). ROBERT uses
independent ExoMolOP molecular cross sections plus its vendored CIA table;
PICASO queries its official SQLite molecular and continuum database. No opacity,
optical depth, or spectrum is shared.

The planet radius is anchored at exactly 10 bar in both frameworks. Both paths
use inverse-square gravity above that reference level; their independently
integrated top radii agree within 0.77 km across the full `1e-5`-10 bar column.

Files:

- `official_picaso_molecular_cloud_parity.json`: provenance and science metrics;
- `official_picaso_molecular_cloud_parity_compact.npz`: common-grid molecular
  optical-depth, emission, and transmission arrays;
- `official_picaso_molecular_cloud_parity.png`: paper diagnostic figure;
- `sampling_convergence.json`: stride 2 and stride 5 errors relative to the full
  native opacity grids;
- `checksums.json`: SHA-256 hashes for all versioned products.

The 7.34 GB PICASO database and native intermediate arrays are intentionally
not committed. Place the database under
`opacity_data/picaso_official/reference/opacities/` as described in the review
report before reproducing the benchmark.
