# Molecular Emission Comparison with PICASO

## Purpose

This benchmark moves the ROBERT–PICASO thermal comparison from grey optical
depths to real molecular correlated-k opacity without mixing opacity-database
differences into the RT test.

Both codes receive the same layer-by-wavelength-by-g optical-depth cube and the
same correlated-k quadrature weights. The cube is assembled by ROBERT from the
bundled HAT-P-32b ExoMolOP/exo_k opacity archives and then passed directly to
PICASO's low-level Toon thermal solver.

## Controlled setup

- Molecules: H2O, CO, CO2, CH4, NH3, and HCN.
- Opacity: bundled ExoMolOP tables prepared with exo_k.
- Spectral grid: 117 bins from 2.886 to 5.177 micron.
- Correlated-k quadrature: 20 g ordinates.
- Atmosphere: 80 layers from `1e-5` to 100 bar.
- Temperature: controlled non-isothermal profile from 800 to 1800 K.
- Geometry: six-point Gauss-Legendre thermal disc quadrature.
- Gas combination: random overlap with rank rebinning.
- Boundary conditions: no incident thermal radiation at the top and a black
  lower boundary.
- Scattering, CIA, and Rayleigh opacity: deliberately excluded from this
  molecular-absorption RT isolation test.

The fixed volume mixing ratios are recorded in the benchmark JSON. They are
not intended as a retrieved HAT-P-32b chemistry solution; they provide strong,
recognizable molecular structure while keeping chemistry outside this test.

## Results

| Metric | Result |
|---|---:|
| Maximum absolute disk relative difference | `5.073e-4` (0.0507%) |
| RMS disk relative difference | `1.515e-4` (0.0151%) |
| Median disk relative difference | `-7.113e-5` |
| Maximum absolute point-radiance difference | `2.802e-3` (0.280%) |

The residual is smooth with wavelength rather than following molecular lines.
This is evidence that molecular optical-depth assembly, random-overlap g
integration, and spectral ordering agree. ROBERT now uses the exact formal
integral for a Planck source linear in optical depth between physical pressure
edges. This reduced the RMS disk residual from 0.0445% to 0.0151% and moved the
median close to zero. The largest point residual occurs at the smallest
emission cosine, where PICASO's source-function angular approximation and
cutoffs are most exposed; it has little disk weight.

## Reproduction

```bash
python examples/compare_molecular_emission_picaso.py
```

Outputs:

- `examples/outputs/molecular_emission_picaso/molecular_emission_picaso_comparison.json`
- `examples/outputs/molecular_emission_picaso/molecular_emission_picaso_comparison.png`
- `examples/outputs/molecular_emission_picaso/molecular_emission_shared_tau.npz`

The shared atmosphere and opacity construction should be reused for the first
transmission benchmark. Transmission must recompute the path through the same
layer extinction using spherical chord geometry; it must not interpret the
vertical emission optical depth as a transit slant optical depth directly.
