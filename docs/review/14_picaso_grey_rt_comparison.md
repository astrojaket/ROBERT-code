# Controlled Grey Thermal RT Comparison with PICASO

Date: 2026-07-11

## Purpose

This comparison isolates radiative-transfer physics from opacity, chemistry,
and cloud-microphysics differences. ROBERT and PICASO receive the same pressure
levels, temperature profile, wavelength grid, per-layer grey extinction,
single-scattering albedo, asymmetry factor, emission angles, and black lower
boundary. Gas optical depth is zero.

PICASO 3.2.2 is run as an external reference process through its low-level
Toon (1989) thermal source-function solver. This avoids regridding through a
molecular opacity database and permits a direct angle-by-angle comparison.

## Reproduction

```bash
ROBERT_PICASO_PYTHON=/path/to/picaso/bin/python \
python examples/compare_grey_cloud_rt_picaso.py
```

The harness writes the complete shared inputs, raw PICASO outputs, a JSON
comparison report, and a spectrum/residual plot under
`examples/outputs/grey_cloud_rt_picaso/`. Generated products are ignored by
git.

## Initial Results

The comparison uses 160 layers, 80 wavelengths from 1 to 12 microns, six
outgoing-angle quadrature points, a column cloud optical depth of 2, and a
black lower boundary.

| Case | Maximum disk-relative difference | Status |
| --- | ---: | --- |
| Isothermal, absorbing | 0.00048% | Pass |
| Temperature gradient, absorbing | 0.84% | Pass at 1% |
| Gradient, omega0=0.5, g=0.0 | 0.72% | Pass at 5% |
| Gradient, omega0=0.9, g=0.0 | 0.58% | Pass at 5% |
| Gradient, omega0=0.9, g=0.6 | 0.66% | Pass at 5% |

The isothermal result validates the wavelength-density unit conversion,
Planck normalization, angular normalization, and black lower boundary. The
gradient absorption residual decreases from 3.33% at 40 layers to 0.84% at
160 layers, consistent with convergence between ROBERT's piecewise-constant
layer source and PICASO's optical-depth-linear Planck source.

Replacing the former effective-extinction shortcut with the coupled
hemispheric-mean source-function solver reduces the disk-integrated scattering
differences below 0.72% after matching the top boundary. PICASO's low-level
routine normally extrapolates a small downward thermal field above its first
positive pressure; the controlled runner sets that extrapolated optical depth
to zero so both codes enforce no incident radiation at `tau=0`.

Angle-resolved agreement is below 1.5% for isotropic scattering but the
`omega0=0.9`, `g=0.6` case still differs by as much as 35% at individual
quadrature angles despite its accurate disk integral. The two-stream angular
reconstruction is therefore not yet validated for phase-resolved science.

## Required Next Physics Change

Validate the implemented thermal two-stream source-function solver against a
higher-fidelity SH4 or discrete-ordinates/matrix-operator backend that:

- resolves enough angular moments for forward-scattering phase functions;
- reproduces the Rutten formal solution and Taylor et al. (2021) limits;
- tests optically thin through semi-infinite columns;
- establishes error envelopes over `omega0`, `g`, wavelength, and viewing angle;
- determines where retrievals may use two stream and where they must switch to
  the higher-order backend.

The three scattering cases in this benchmark are the first acceptance tests
for that solver. Molecular ExoMol opacity should be added only after these
controlled grey cases pass, using a documented user opacity-data directory.
