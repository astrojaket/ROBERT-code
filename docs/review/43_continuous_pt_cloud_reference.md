# Continuous non-isothermal P/T-opacity and cloud-boundary benchmark

## Scope

This benchmark extends the continuous transmission reference in three ways:

1. temperature varies smoothly from 1600 K at 10 bar to 800 K at 1e-9 bar,
   with an additional 250 K mid-atmosphere temperature minimum;
2. each analytic absorption band has its own pressure and temperature
   exponent; and
3. a sharp aerosol boundary is fixed at 2e-3 bar and evaluated on both an
   edge-aligned grid and a standard log-pressure grid that cuts through the
   cloud-top layer.

The inverse-square hydrostatic pressure-radius relation is integrated
analytically over the prescribed temperature function. The continuous
reference evaluates the full P/T-dependent extinction along each chord. Its
line-of-sight and impact-parameter integrals are split exactly at the cloud
radius, so the reference does not inherit a finite-grid cloud-boundary error.

Both ROBERT and the external PICASO 3.2.2 `get_transit_1d` kernel receive the
same pressure levels, exact radii, temperatures, and exact layer-integrated
vertical optical depths. For each layer count the aligned and misaligned grids
represent the same physical atmosphere and the same 2e-3 bar cloud top.

## Continuous-reference precision

The production reference uses 1024-point Gauss--Legendre integration. The
512-point check differs by 1.93e-9 ppm RMS in the clear spectrum, 3.08e-5 ppm
in the cloudy spectrum, and 3.08e-5 ppm in cloud effect. These uncertainties
are negligible compared with all reported finite-layer residuals.

## Clear non-isothermal P/T-opacity result

At 80 layers, ROBERT's clear-spectrum RMS error is 1.13--1.14 ppm depending on
the nearly uniform pressure-grid phase. PICASO's is 6.62--6.65 ppm. The
isothermal conclusion therefore survives smooth thermal structure and
pressure/temperature-dependent opacity: ROBERT's uniform-shell Gaussian
solver remains closer to the continuous atmosphere at this resolution.

## Sharp cloud boundary

Cloud-effect RMS errors are:

| Layers | ROBERT aligned | ROBERT misaligned | PICASO aligned | PICASO misaligned |
|---:|---:|---:|---:|---:|
| 20 | 3.875 ppm | 19.135 ppm | 6.953 ppm | 30.165 ppm |
| 40 | 0.851 ppm | 2.856 ppm | 2.385 ppm | 6.211 ppm |
| 80 | 0.202 ppm | 3.495 ppm | 0.868 ppm | 4.641 ppm |
| 160 | 0.046 ppm | 3.035 ppm | 0.316 ppm | 3.873 ppm |
| 320 | 0.009 ppm | 1.270 ppm | 0.121 ppm | 1.543 ppm |

With the cloud top on an edge, ROBERT converges at fitted order 2.17 and
PICASO at order 1.46. At 80 layers ROBERT's aligned cloud-effect maximum error
is 1.12 ppm, compared with 4.54 ppm for PICASO.

The misaligned results are non-monotonic because the cloud top moves through
different fractions of the intersected layer as resolution changes. Exact
vertical optical-depth conservation is not enough: distributing that
partial-layer cloud optical depth uniformly across the whole spherical shell
misplaces extinction along tangent chords. At 80 layers this raises ROBERT's
cloud-effect RMS error from 0.20 to 3.49 ppm and PICASO's from 0.87 to
4.64 ppm.

## Consequence for ROBERT

The current edge-aware cloud model correctly conserves vertical optical
depth, but a retrieved cloud pressure will almost never coincide with a fixed
pressure edge. Dynamically rebuilding the full opacity grid at every
likelihood evaluation would be costly and can introduce non-smooth posterior
behaviour.

Cloud-aware RT sub-layering is a documented future optimization, not the next
production feature. A possible implementation would split only the intersected
shell geometrically at the exact cloud pressure, reuse the parent layer's gas
properties, and place aerosol extinction only in the cloudy sub-shell. This
would preserve a fixed gas/opacity grid while giving the spherical chord solver
the correct cloud boundary.

The optimization is deliberately gated. The next benchmark is an end-to-end,
independently PICASO-generated JWST-like retrieval evaluated by ROBERT without
changing its current forward model. Sub-layering should move ahead only if that
retrieval shows a scientifically material cloud-pressure or aerosol bias that
can be localized to the partial cloud-top shell. The continuous benchmark then
provides its numerical acceptance target: at 80 layers, a sub-layered result
should approach the 0.20 ppm aligned-grid RMS rather than the current 3.49 ppm
misaligned-grid RMS. A generic cross-code opacity mismatch, unconstrained cloud
parameter, or acceptable end-to-end fit is not sufficient justification for
adding the extra production complexity.

## Limitations

The analytic opacity is monochromatic rather than correlated-k, the cloud is
pure extinction with a fixed spectral slope, and refraction and scattered
light return remain excluded. These choices isolate the geometry and
discretization question; they do not replace the separate database, k-mixing,
Mie, or forward-scattering validations.

## Artifacts

- `examples/outputs/continuous_pt_cloud_reference/continuous_pt_cloud_reference.json`
- `examples/outputs/continuous_pt_cloud_reference/continuous_pt_cloud_reference.png`
- `examples/outputs/continuous_pt_cloud_reference/continuous_pt_cloud_reference_spectra.npz`
- `examples/outputs/continuous_pt_cloud_reference/picaso_continuous_pt_cloud.npz`
