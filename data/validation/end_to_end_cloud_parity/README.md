# End-to-End Cloud Parity Reference

This directory is the versioned 2026-07-13 reference output for
`examples/benchmark_end_to_end_cloud_parity.py` at 72 pressure layers, 96
wavelengths, 36 particle-radius bins, six Mie subradii per bin, and six disk
angles.

Files:

- `shared_physical_contract.npz`: the only input shared between frameworks;
  it intentionally contains no opacity, optical depth, or spectrum;
- `robert_independent_output.npz`: ROBERT Mie, gas/cloud optical-depth, and
  SH4 products;
- `picaso_virga_independent_output.npz`: external Virga/PICASO products;
- `end_to_end_cloud_parity.json`: provenance, metrics, and acceptance gates;
- `end_to_end_cloud_parity.png`: paper-review diagnostic figure.
- `checksums.json`: SHA-256 integrity hashes for the versioned products.

The analytic gas opacity is a validation fixture, not science opacity. See
`docs/review/35_end_to_end_picaso_virga_cloud_parity.md` for the full scope,
convergence study, interpretation, and remaining science-opacity benchmark.
