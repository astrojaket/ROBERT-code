# Shared aerosol and Mie benchmarks

## Scope

This review validates the shared cloud interface without treating another
forward model as ground truth. It separates four questions:

1. Does the cloud parameterization conserve its own physical contract under
   vertical refinement?
2. Do emission and transmission receive identical layer optical properties?
3. What changes when ROBERT and an external framework use independent gas and
   continuum opacity databases?
4. What remains when the evaluated layer optical depths are matched exactly?

All ROBERT calculations used the `robert-exoplanets` conda environment.
External references used PICASO 3.2.2 with its official R=15000 database and
petitRADTRANS 3.3.3 with its native R=1000 correlated-k tables.

## Finite deck and power-law haze

The four-state matrix is clear, deck, haze, and deck+haze, evaluated in both
emission and transmission. The shared physical parameters are a deck top of
1e-3 bar, integrated vertical deck optical depth 0.3, and a well-mixed
isotropically scattering haze with 1e-3 cm2/g extinction at 1 micron and slope
-4.

The first convergence run exposed layer-center snapping at the sharp deck top.
ROBERT now allocates deck optical depth by fractional overlap in log-pressure,
while conserving the requested integrated optical depth exactly. This is an
edge-aware ROBERT discretization, not a reproduction of pRT's pressure-node
continuum integration.

Relative to the 160-layer result, ROBERT's 80-layer numerical changes are:

| Cloud effect | Emission RMS | Transmission RMS |
|---|---:|---:|
| Deck | 0.020 ppm | 0.561 ppm |
| Haze | 0.001 ppm | 0.077 ppm |
| Deck+haze | 0.020 ppm | 0.583 ppm |

At 80 layers, the official-PICASO independent-opacity cloud-effect differences
are:

| Cloud effect | Emission RMS | Transmission RMS |
|---|---:|---:|
| Deck | 3.44 ppm | 29.34 ppm |
| Haze | 0.33 ppm | 8.59 ppm |
| Deck+haze | 3.47 ppm | 30.15 ppm |

The larger transmission deck difference should not be interpreted as a ROBERT
accuracy failure. The matched-layer-optical-depth diagnostic feeds PICASO's
evaluated gas and cloud optical depths directly to ROBERT's spherical solver.
At native opacity stride 5 this reduces the 80-layer total spectrum difference
to 1.68--3.74 ppm RMS and the aerosol-effect difference to 3.31--6.33 ppm RMS. The
remaining difference is consistent with annulus discretization and
pressure-radius integration conventions. The much larger independent-opacity
difference is therefore predominantly a nonlinear opacity/cloud interaction.

The native-pRT comparison is intentionally not a matched-deck test. pRT uses a
continuous mass opacity integrated between pressure nodes, while ROBERT uses a
finite integrated optical depth distributed over edge-defined layers. The
haze definitions are much closer: at 80 layers the differential differences
are 0.43 ppm in emission and 12.15 ppm in transmission. The native deck
differences quantify the consequence of the different parameterizations rather
than a target ROBERT should be tuned to reproduce.

## Shared Mie particles

`ParameterizedMieCloudModel` now implements the geometry-independent cloud
protocol. Fixed catalogue and retrieved nodal n/log10(k) modes both return the
same layer extinction, single-scattering albedo, asymmetry, and Mie phase
moments to emission and transmission. Configured Mie models are now accepted
for either geometry; the older emission-specific class remains as a
compatibility API.

The shared-protocol benchmark uses MgSiO3, a 0.3 micron effective radius,
lognormal width 1.6, 24 quadrature points, and a 1.2e-5 condensate mass
fraction. Compared with an explicit lower-level Mie construction:

- maximum layer extinction difference: 2.13e-14;
- maximum single-scattering-albedo difference: 1.03e-15;
- maximum phase-moment difference: 9.64e-14;
- maximum emission spectrum difference: 1.79e-9 ppm;
- maximum transmission spectrum difference: 3.47e-12 ppm.

The independent native-R=15000 PICASO/Virga comparison remains the external
physics benchmark. Its Mie mass-extinction RMS relative difference is
1.58e-6, while the independently evaluated cloud effects differ by 7.89 ppm in
emission and 17.84 ppm in transmission.

## Interpretation and next diagnostic

The benchmarks support the shared-cloud architecture and show that ROBERT's
edge-aware deck is numerically well behaved. They do not establish that every
external difference belongs to the other framework. The next focused test
should vary the pressure-radius anchor, annulus quadrature, and cloud-top
location while holding evaluated layer optical depths fixed. That will map the
remaining few-ppm matched-optical-depth transmission residual without changing
ROBERT to imitate pRT or PICASO.

Spherical shell geometry remains the correct baseline for transmission. The
observable is the stellar area blocked after integrating chord transmission
over impact parameter, so the solver must traverse curved slant paths through
concentric pressure shells. A plane-parallel vertical solution can help with
local approximations, but it is not the transit-depth integral. The remaining
model assumptions to validate are the 1D terminator, hydrostatic
pressure-radius mapping, reference-radius anchor, annulus quadrature,
refraction, and whether forward-scattered light can safely be treated as lost
from the direct beam.

## Artifacts

- `examples/outputs/shared_deck_haze_picaso/`
- `examples/outputs/shared_deck_haze_external_parity/`
- `examples/outputs/shared_mie_protocol/`
- `examples/outputs/official_picaso_molecular_cloud_parity/`
